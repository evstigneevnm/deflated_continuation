#ifndef __STABILITY_CONTINUATION_HPP__
#define __STABILITY_CONTINUATION_HPP__
/**
 * Postprocesses saved deflation-continuation curves into a stability diagram.
 * The eigensolver is supplied as a structured adapter; this class owns only
 * branch traversal and the Newton solve used to refine stability transitions.
 */

#include <string>
#include <vector>
#include <deque>
#include <stdexcept>
#include <utility>
#include <cstdint>

#include <common/vector_snapshot_queue.h>

#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <containers/bifurcation_diagram/diagram_archive.h>
#include <deflation/solution_storage.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <containers/stability_diagram.h>

#include <stability/stability_analysis.hpp>



namespace main_classes
{

template<
    class VectorOperations,
    class VectorFileOperations,
    class Log,
    class Monitor,
    class NonlinearOperations,
    class LinearOperator,
    class Preconditioner,
    template<class, class, class, class, class> class LinearSolver,
    template<class, class, class, class> class SystemOperator,
    class Parameters,
    class EigensolverAdapter,
    class StabilityLinearizationProvider = NonlinearOperations>
class stability_continuation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;
    
    //linear solver
    using lin_slv_t = LinearSolver<LinearOperator, Preconditioner, VectorOperations, Monitor, Log>;

    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;

    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        lin_slv_t
        > system_operator_t;
    
    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;
    


    using stability_t = stability::stability_analysis<
        VectorOperations,   
        NonlinearOperations, 
        Log, 
        newton_t,
        EigensolverAdapter,
        StabilityLinearizationProvider>;


    typedef container::curve_helper_container<VectorOperations> container_helper_t;

    typedef deflation::solution_storage<VectorOperations> sol_storage_def_t;   


    typedef container::bifurcation_diagram_curve<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        container_helper_t
        > bif_diag_curve_t;

    typedef container::bifurcation_diagram<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        bif_diag_curve_t,
        container_helper_t
        > bif_diag_t;

    typedef container::stability_diagram<
        VectorOperations, 
        VectorFileOperations, 
        Log
        > stability_diagram_t;



    typedef common::vector_snapshot_queue<VectorOperations> queue_pointer_t;
    typedef std::deque<T> queue_lambda_t;
    typedef std::deque<std::pair<int, int>> queue_dims_t;
    typedef std::deque<uint64_t> queue_source_indices_t;

public:
    using stability_result_type =
        typename stability_t::analysis_result_type;
    using stability_transition_result_type =
        typename stability_t::transition_result_type;

    stability_continuation(
        VectorOperations* vec_ops_,
        VectorFileOperations* file_ops_,
        Log* log_,
        Log* log_linsolver_,
        NonlinearOperations* nonlin_op_,
        Parameters* parameters_,
        EigensolverAdapter* eigensolver_adapter_,
        StabilityLinearizationProvider*
            stability_linearization_provider_ = nullptr):
            vec_ops(vec_ops_),
            file_ops(file_ops_),
            log(log_),
            nonlin_op(nonlin_op_),
            log_linsolver(log_linsolver_),
            parameters(parameters_),
            eigensolver_adapter(eigensolver_adapter_),
            stability_linearization_provider(
                stability_linearization_provider_)

    {
        if(eigensolver_adapter == nullptr)
            throw std::invalid_argument(
                "stability_continuation: eigensolver adapter is null");
        
        project_dir = parameters->path_to_project;
        skip_files = parameters->deflation_continuation.skip_files;

        //set project directory the same way as in deflation_continuation
        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';

        lin_op = new LinearOperator(nonlin_op);  
        prec = new Preconditioner(nonlin_op);  

        lin_slv = new lin_slv_t(vec_ops, log_linsolver);
        lin_slv->set_preconditioner(prec);

        convergence_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, lin_slv);
        newton = new newton_t(vec_ops, system_operator, convergence_newton);

        stab = new stability_t(
            vec_ops,
            log,
            nonlin_op,
            newton,
            eigensolver_adapter,
            stability_linearization_provider);


        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files);

        stability_diagram = new stability_diagram_t(vec_ops, file_ops, log, project_dir);

        queue_pointer = new queue_pointer_t(vec_ops, queue_size);
        queue_lambda = new queue_lambda_t();
        queue_dims = new queue_dims_t();
        queue_source_indices = new queue_source_indices_t();

        vec_ops->init_vector(x_p);
        vec_ops->start_use_vector(x_p);
        vec_ops->init_vector(x_transition_upper);
        vec_ops->start_use_vector(x_transition_upper);
        
    }
    ~stability_continuation()
    {
        delete queue_source_indices;
        delete queue_dims;
        delete queue_lambda;
        delete queue_pointer;
        delete stability_diagram;
        delete bif_diag;
        delete stab;        
        delete newton;
        delete system_operator;
        delete convergence_newton;
        delete lin_slv;
        delete prec;
        delete lin_op;
        


        vec_ops->stop_use_vector(x_transition_upper);
        vec_ops->free_vector(x_transition_upper);
        vec_ops->stop_use_vector(x_p);
        vec_ops->free_vector(x_p);

    }
   

    void set_parameters()
    {
        set_linsolver();
        set_newton();
        set_stability_classifier();
        set_linear_operator_stable_eigenvalues_halfplane();
        set_stability_transition_refinement();
    }

    template<class Aligner>
    void set_transition_state_aligner(Aligner* aligner)
    {
        stab->set_transition_state_aligner(aligner);
    }

    void reset_transition_state_aligner()
    {
        stab->reset_transition_state_aligner();
    }

    void set_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true*/    
    {
        //setup linear system:
        mon = &lin_slv->monitor();
    
        T lin_solver_tol = parameters->stability_continuation.linear_solver.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->stability_continuation.linear_solver.lin_solver_max_it;
        bool save_convergence_history_ = parameters->stability_continuation.linear_solver.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->stability_continuation.linear_solver.divide_out_norms_by_rel_base;
        bool verbose_ = parameters->stability_continuation.linear_solver.verbose;
        int use_precond_resid = parameters->stability_continuation.linear_solver.use_precond_resid;
        int resid_recalc_freq = parameters->stability_continuation.linear_solver.resid_recalc_freq;
        int basis_sz = parameters->stability_continuation.linear_solver.basis_size;

        mon->init(lin_solver_tol, T(0.0), lin_solver_max_it);
        mon->set_save_convergence_history(save_convergence_history_);
        mon->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon->set_verbose(verbose_);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            lin_slv->set_use_precond_resid(use_precond_resid);
        if(resid_recalc_freq >= 0)
            lin_slv->set_resid_recalc_freq(resid_recalc_freq);
        if(basis_sz > 0)
            lin_slv->set_basis_size(basis_sz);  

    }

    void set_newton()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        T tolerance_ = parameters->stability_continuation.newton.tolerance;
        unsigned int maximum_iterations_ = parameters->stability_continuation.newton.newton_max_it;
        T newton_wight_ = parameters->stability_continuation.newton.newton_wight;

        bool store_norms_history_ = parameters->stability_continuation.newton.store_norms_history;
        bool verbose_ = parameters->stability_continuation.newton.verbose;        

        convergence_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);
    }

    void set_linear_operator_stable_eigenvalues_halfplane()
/*const T sign_*/    
    {
        bool left_hp = parameters->stability_continuation.linear_operator_stable_eigenvalues_left_halfplane;
        
        if(left_hp)
        {
            stab->set_linear_operator_stable_eigenvalues_halfplane(T(-1.0));
            log->info(
                "stability classifier uses the left stable half-plane");
        }
        else
        {
            stab->set_linear_operator_stable_eigenvalues_halfplane(T(1.0));
            log->info(
                "stability classifier uses the right stable half-plane");
        }
    }

    void set_stability_classifier()
    {
        typename stability_t::classifier_options_type options =
            stab->classifier_options();
        options.stability_boundary_tolerance =
            parameters->stability_continuation.
                stability_boundary_tolerance;
        options.real_eigenvalue_tolerance =
            parameters->stability_continuation.
                real_eigenvalue_tolerance;
        options.conjugate_pair_tolerance =
            parameters->stability_continuation.
                conjugate_pair_tolerance;
        options.require_converged_eigenpairs =
            parameters->stability_continuation.
                require_converged_eigenpairs;
        options.require_nonempty_spectrum =
            parameters->stability_continuation.
                require_nonempty_spectrum;
        options.require_complete_scan_coverage =
            parameters->stability_continuation.
                require_complete_scan_coverage;
        stab->set_classifier_options(options);
        stab->set_classification_retry_count(
            parameters->stability_continuation.
                spectrum_classification_retries);
        stab->set_classification_confirmation_count(
            parameters->stability_continuation.
                transition_classification_confirmations);
    }

    void set_stability_transition_refinement()
    {
        const bool correct_with_newton =
            parameters->stability_continuation.
                correct_stability_transitions_with_newton;
        stab->set_transition_refinement_uses_fixed_parameter_newton(
            correct_with_newton);
        stab->set_transition_refinement_maximum_iterations(
            parameters->stability_continuation.
                transition_refinement_maximum_iterations);
        stab->set_transition_refinement_maximum_subdivisions(
            parameters->stability_continuation.
                transition_refinement_maximum_subdivisions);
        stab->set_transition_refinement_parameter_tolerance(
            parameters->stability_continuation.
                transition_refinement_parameter_tolerance);
        log->info(
            correct_with_newton
                ? "stability transitions use fixed-parameter Newton refinement"
                : "stability transitions use continuation-secant refinement");
        log->info_f(
            "stability transition refinement: maximum iterations = %u, "
            "parameter tolerance = %.3e, maximum subdivisions = %u, "
            "classification confirmations = %u, symmetry endpoint "
            "guard = %u source points",
            parameters->stability_continuation.
                transition_refinement_maximum_iterations,
            double(
                parameters->stability_continuation.
                    transition_refinement_parameter_tolerance),
            parameters->stability_continuation.
                transition_refinement_maximum_subdivisions,
            parameters->stability_continuation.
                transition_classification_confirmations,
            parameters->stability_continuation.
                symmetry_endpoint_guard_source_points);
    }


    bool execute_single_curve(int curve_number_)
    {
        typedef std::pair<bool, bool> bool2;
        using curve_point_type =
            typename bif_diag_curve_t::values_t;

        int container_index = 0;
        int requested_curve = curve_number_;
        const std::vector<curve_point_type> curve_points =
            bif_diag->get_curve_points_vector(requested_curve);
        std::size_t symmetry_endpoint_count = 0;
        for(const auto& point: curve_points)
        {
            if(
                point.endpoint_reason ==
                container::curve_endpoint_reason::
                    symmetry_intersection)
            {
                ++symmetry_endpoint_count;
            }
        }
        if(symmetry_endpoint_count != 0)
        {
            log->info_f(
                "stability traversal: loaded %zu saved symmetry "
                "endpoint(s) for curve %i",
                symmetry_endpoint_count,
                curve_number_);
        }

        bool2 res_read = bool2(true, true);
        
        queue_pointer->clear();
        queue_lambda->clear();
        queue_dims->clear();
        queue_source_indices->clear();
        bool curve_succeeded = true;
        curve_point_type previous_point;
        bool previous_point_available = false;

        while(res_read.second)
        { 
            bool curve_break = false;
            curve_point_type current_point;
            bool metadata_available = false;
            res_read = bif_diag->get_solutoin_from_curve(
                requested_curve,
                container_index,
                lambda_p,
                x_p,
                current_point,
                metadata_available);

            if(!res_read.first)
            {
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
                queue_source_indices->clear();
                break;
            }
            if(!res_read.second)
            {
                break;
            }
            T x_p_norm = vec_ops->norm_l2(x_p);
            
            if(container_index == 1)
            {
                x_p_start_norm = x_p_norm;
                lambda_p_start = lambda_p;
            }
            else if(
                metadata_available &&
                previous_point_available)
            {
                curve_break =
                    container::starts_new_curve_traversal_segment(
                        previous_point,
                        current_point);
            }
            else if(!metadata_available)
            {
                if (
                    (std::abs(lambda_p_start-lambda_p)<T(1.0e-6)) &&
                    (std::abs(x_p_start_norm - x_p_norm)<T(1.0e-6)) )
                {
                    curve_break = true;
                }

            }
            if(
                metadata_available &&
                container::is_stability_refinement_barrier(
                    current_point.endpoint_reason))
            {
                curve_break = true;
                log->info_f(
                    "stability traversal: starting a new stability "
                    "segment at %s endpoint source point %llu, "
                    "lambda = %.16le; the endpoint is classified "
                    "without smooth-branch transition refinement",
                    container::to_string(
                        current_point.endpoint_reason),
                    static_cast<unsigned long long>(
                        current_point.point_index),
                    double(lambda_p));
            }

            log->info_f(
                "curve number = %i, index = %i, lambda = %.3lf, "
                "||x|| = %.3le, curve_break = %d, segment = %llu, "
                "semicurve = %llu, endpoint = %s",
                curve_number_,
                container_index,
                lambda_p,
                x_p_norm,
                curve_break,
                static_cast<unsigned long long>(
                    current_point.segment_id),
                static_cast<unsigned long long>(
                    current_point.semicurve_id),
                container::to_string(
                    current_point.endpoint_reason));

            if( curve_break )
            {
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
                queue_source_indices->clear();
            }

            const curve_point_type* following_symmetry_endpoint =
                metadata_available
                ? container::
                      following_symmetry_endpoint_within_source_points(
                          curve_points,
                          current_point,
                          parameters->stability_continuation.
                              symmetry_endpoint_guard_source_points)
                : nullptr;
            if(following_symmetry_endpoint != nullptr)
            {
                log->info_f(
                    "stability traversal: omitting source point %llu at "
                    "lambda = %.16le within the guard neighborhood of "
                    "symmetry endpoint source point %llu at %.16le",
                    static_cast<unsigned long long>(
                        current_point.point_index),
                    double(lambda_p),
                    static_cast<unsigned long long>(
                        following_symmetry_endpoint->point_index),
                    double(following_symmetry_endpoint->lambda));
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
                queue_source_indices->clear();
                previous_point = current_point;
                previous_point_available = true;
                continue;
            }


            queue_pointer->push(x_p); 
            push_queue(*queue_lambda, lambda_p);
            
            try
            {
                const auto stability_point = stab->analyze(x_p, lambda_p);
                if(!stability_point.succeeded())
                {
                    throw std::runtime_error(
                        std::string("stability classification failed: ") +
                        stability_point.diagnostic);
                }
                std::pair<int, int> unstable_dim_p =
                    stability_point.unstable_dimension_pair();
                const uint64_t source_point_index =
                    metadata_available
                    ? current_point.point_index
                    : (
                          container_index > 0
                          ? static_cast<uint64_t>(
                                container_index - 1)
                          : uint64_t(0));
                push_queue(*queue_dims, unstable_dim_p);
                push_queue(
                    *queue_source_indices,
                    source_point_index);
                std::size_t transition_count = 0;
                if( queue_pointer->is_queue_filled() )
                {
                    if(
                        stability::analysis::
                            unstable_subspace_dimension(
                                queue_dims->at(0)) !=
                        stability::analysis::
                            unstable_subspace_dimension(
                                queue_dims->at(1)))
                    {
                        log->info_f("container::stability_diagram: (lambda0,||x0||) = (%lf;%lf), (lambda1,||x1||) = (%lf;%lf).", double(queue_lambda->at(0)), vec_ops->norm(queue_pointer->at(0)), double(queue_lambda->at(1)), vec_ops->norm(queue_pointer->at(1)) );
                        const auto lower_confirmation =
                            stab->analyze_confirmed(
                                queue_pointer->at(0),
                                queue_lambda->at(0));
                        const auto upper_confirmation =
                            stab->analyze_confirmed(
                                queue_pointer->at(1),
                                queue_lambda->at(1));
                        if(!lower_confirmation.succeeded())
                        {
                            throw std::runtime_error(
                                "lower transition endpoint "
                                "confirmation failed: " +
                                lower_confirmation.diagnostic);
                        }
                        if(!upper_confirmation.succeeded())
                        {
                            throw std::runtime_error(
                                "upper transition endpoint "
                                "confirmation failed: " +
                                upper_confirmation.diagnostic);
                        }

                        const auto lower_confirmed_dimension =
                            lower_confirmation.
                                unstable_dimension_pair();
                        const auto upper_confirmed_dimension =
                            upper_confirmation.
                                unstable_dimension_pair();
                        if(
                            lower_confirmed_dimension !=
                                queue_dims->at(0) ||
                            upper_confirmed_dimension !=
                                queue_dims->at(1))
                        {
                            log->warning_f(
                                "stability transition confirmation "
                                "revised endpoint signatures from "
                                "(%i,%i)->(%i,%i) to "
                                "(%i,%i)->(%i,%i)",
                                queue_dims->at(0).first,
                                queue_dims->at(0).second,
                                queue_dims->at(1).first,
                                queue_dims->at(1).second,
                                lower_confirmed_dimension.first,
                                lower_confirmed_dimension.second,
                                upper_confirmed_dimension.first,
                                upper_confirmed_dimension.second);
                        }
                        if(
                            lower_confirmed_dimension !=
                            queue_dims->at(0))
                        {
                            stability_diagram->
                                update_regular_point_dimension(
                                    queue_source_indices->at(0),
                                    lower_confirmed_dimension);
                        }
                        queue_dims->at(0) =
                            lower_confirmed_dimension;
                        queue_dims->at(1) =
                            upper_confirmed_dimension;
                        unstable_dim_p =
                            upper_confirmed_dimension;

                        if(
                            stability::analysis::
                                unstable_subspace_dimension(
                                    lower_confirmed_dimension) !=
                            stability::analysis::
                                unstable_subspace_dimension(
                                    upper_confirmed_dimension))
                        {
                            transition_count =
                                stab->refine_transition_sequence_confirmed(
                                    queue_pointer->at(0),
                                    queue_lambda->at(0),
                                    lower_confirmation,
                                    queue_pointer->at(1),
                                    queue_lambda->at(1),
                                    upper_confirmation,
                                    [this, source_point_index](
                                    const stability_transition_result_type&
                                        transition,
                                    const T_vec& transition_state)
                                {
                                    const auto before_dimension =
                                        transition.before_stability.
                                            unstable_dimension_pair();
                                    const auto after_dimension =
                                        transition.after_stability.
                                            unstable_dimension_pair();
                                    std::vector<T> transition_norms;
                                    calculate_plot_norms(
                                        transition_state,
                                        transition_norms);
                                    stability_diagram->
                                        add_with_plot_data(
                                            transition.parameter,
                                            after_dimension.first,
                                            after_dimension.second,
                                            source_point_index,
                                            transition_norms,
                                            before_dimension,
                                            after_dimension,
                                            transition_state);
                                    });
                        }
                    }

                }

                if(transition_count == 0)
                {
                    std::vector<T> plot_norms;
                    calculate_plot_norms(x_p, plot_norms);
                    stability_diagram->add_with_plot_data(
                        lambda_p,
                        unstable_dim_p.first,
                        unstable_dim_p.second,
                        source_point_index,
                        plot_norms,
                        unstable_dim_p,
                        unstable_dim_p);
                }

            }
            catch(const std::exception& e)
            {
                log->warning_f("container::stability_diagram: %s. Point was not added.", e.what());
                log->warning_f("container::stability_diagram: failed for lambda_p = %lf; curve transaction is aborted.", lambda_p);
                queue_pointer->clear();
                queue_lambda->clear();
                queue_dims->clear();
                queue_source_indices->clear();
                curve_succeeded = false;
                break;
            }

            previous_point = current_point;
            previous_point_available = metadata_available;
            
        }


        return res_read.first && curve_succeeded;

    }

    stability_result_type execute_single_state(
        const std::string& state_file,
        T parameter,
        bool confirmed = false)
    {
        file_ops->read_vector(state_file, x_p);
        const stability_result_type result =
            confirmed
            ? stab->analyze_confirmed(x_p, parameter)
            : stab->analyze(x_p, parameter);
        log->info_f(
            "single-state stability replay: file = %s, "
            "lambda = %.16le, status = %s, confirmed = %i",
            state_file.c_str(),
            double(parameter),
            result.succeeded() ? "success" : "failure",
            confirmed ? 1 : 0);
        return result;
    }

    stability_transition_result_type execute_single_transition(
        const std::string& lower_state_file,
        T lower_parameter,
        const std::string& upper_state_file,
        T upper_parameter,
        bool confirmed = false)
    {
        file_ops->read_vector(lower_state_file, x_p);
        file_ops->read_vector(
            upper_state_file,
            x_transition_upper);
        const stability_transition_result_type result =
            confirmed
            ? stab->refine_transition_confirmed(
                  x_p,
                  lower_parameter,
                  x_transition_upper,
                  upper_parameter,
                  x_p)
            : stab->refine_transition(
                  x_p,
                  lower_parameter,
                  x_transition_upper,
                  upper_parameter,
                  x_p);
        log->info_f(
            "two-state stability transition replay: "
            "lower = %s at %.16le, upper = %s at %.16le, "
            "status = %s, refined lambda = %.16le, confirmed = %i",
            lower_state_file.c_str(),
            double(lower_parameter),
            upper_state_file.c_str(),
            double(upper_parameter),
            stability::analysis::
                stability_transition_status_name(result.status),
            double(result.parameter),
            confirmed ? 1 : 0);
        return result;
    }


    void edit()
    {
        std::string file_name_stability_ = parameters->stability_diagram_file_name;
        
        bool stability_data = load_stability_data(file_name_stability_);  
        if(stability_data)
        {
            std::cout << "entering interactive edit mode" << std::endl;
            std::cout << "enter 'd' to pop_back() the curve or 'q' to quit." << std::endl;
            char c = 'c';
            while(c != 'q')
            {
                std::cout << "file " << file_name_stability_ << " contains:" << std::endl;
                stability_diagram->print_curves_status();
                c = std::cin.get();
                if(c=='d')
                {
                    stability_diagram->pop_back_curve();
                }
            }
            c = std::cin.get();
            std::cout << "save file(y/n)>>>";
            c = std::cin.get();
            if(c == 'y')
                save_stability_data(file_name_stability_);

        }
        else
        {
            log->warning_f("MAIN:stability_diagram: file %s doesn't exist; called edit with no file provided!", file_name_stability_.c_str());
        }

    }


    void execute()
    {
        std::string file_name_diagram_ = parameters->bifurcaiton_diagram_file_name;
        std::string file_name_stability_ = parameters->stability_diagram_file_name;

        bool file_exists = load_diagram_data(file_name_diagram_);
        if(file_exists)
        {
            bool stability_data = load_stability_data(file_name_stability_);
            std::size_t curve_number = 0;
            if(stability_data)
            {
                curve_number = stability_diagram->current_curve();
            }
            const std::size_t available_curves =
                bif_diag->curve_count();
            if(curve_number > available_curves)
            {
                throw std::runtime_error(
                    "stability diagram contains more curves than the "
                    "bifurcation diagram");
            }
            while(curve_number < available_curves)
            {
                log->info_f(
                    "executing curve = %zu/%zu",
                    curve_number,
                    available_curves);
                stability_diagram->open_curve(
                    static_cast<int>(curve_number));
                const bool curve_processed =
                    execute_single_curve(
                        static_cast<int>(curve_number));
                if(!curve_processed)
                {
                    stability_diagram->abandon_curve();
                    throw std::runtime_error(
                        "failed to traverse bifurcation curve " +
                        std::to_string(curve_number));
                }
                stability_diagram->close_curve();
                save_stability_data(file_name_stability_);
                ++curve_number;
            }
        }

    }


private:
    void calculate_plot_norms(
        const T_vec& state,
        std::vector<T>& norms) const
    {
        nonlin_op->norm_bifurcation_diagram(state, norms);
        if(norms.empty())
            norms.push_back(vec_ops->norm_l2(state));
    }

    template<class Queue, class Value>
    void push_queue(Queue& queue, Value&& value) const
    {
        if(queue.size() == queue_size)
        {
            queue.pop_front();
        }
        queue.push_back(std::forward<Value>(value));
    }

    static constexpr std::size_t queue_size = 2;

    VectorOperations* vec_ops; 
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    Parameters* parameters;
    EigensolverAdapter* eigensolver_adapter;
    StabilityLinearizationProvider*
        stability_linearization_provider;
    std::string project_dir;
    unsigned int skip_files;
//created locally:
    LinearOperator* lin_op = nullptr;
    Preconditioner* prec = nullptr;
    lin_slv_t* lin_slv = nullptr;
    monitor_t* mon = nullptr;
    newton_t* newton = nullptr;
    convergence_newton_t* convergence_newton = nullptr;
    system_operator_t* system_operator = nullptr;
    stability_t* stab = nullptr;
    bif_diag_t* bif_diag = nullptr;
    queue_pointer_t* queue_pointer = nullptr;
    queue_lambda_t* queue_lambda = nullptr;
    queue_dims_t* queue_dims = nullptr;
    queue_source_indices_t* queue_source_indices = nullptr;
    stability_diagram_t* stability_diagram = nullptr;

//  to detect curve break during analysis
    T x_p_start_norm;
    T lambda_p_start;

    T_vec x_p;
    T_vec x_transition_upper;
    T lambda_p;


    bool load_diagram_data(const std::string file_name_ = {})
    {
        if(file_name_.empty())
        {
            return false;
        }
        const std::string path = project_dir + file_name_;
        const auto result =
            container::load_diagram_archive(path, *bif_diag);
        if(!result.succeeded())
        {
            log->warning_f(
                "MAIN:stability_continuation: failed to load "
                "bifurcation diagram %s: %s",
                path.c_str(),
                result.message.c_str());
            return false;
        }
        log->info_f(
            "MAIN:stability_continuation: read bifurcation diagram "
            "from %s",
            path.c_str());
        return true;
    }

    bool load_stability_data(const std::string file_name_ = {})
    {
        if(file_name_.empty())
        {
            return false;
        }
        const std::string path = project_dir + file_name_;
        const auto result =
            container::load_diagram_archive(
                path,
                *stability_diagram);
        if(!result.succeeded())
        {
            log->warning_f(
                "MAIN:stability_continuation: failed to load "
                "stability diagram %s: %s",
                path.c_str(),
                result.message.c_str());
            return false;
        }
        log->info_f(
            "MAIN:stability_continuation: read stability diagram "
            "from %s",
            path.c_str());
        return true;
    }


    void save_stability_data(const std::string& file_name_ = {})
    {
        if(!file_name_.empty())
        {
            const std::string path = project_dir + file_name_;
            const auto result =
                container::save_diagram_archive(
                    path,
                    *stability_diagram);
            if(!result.succeeded())
            {
                throw std::runtime_error(
                    "failed to save stability diagram " + path +
                    ": " + result.message);
            }
            log->info_f(
                "MAIN:stability_continuation: saved stability "
                "diagram in %s",
                path.c_str());
        }        
    }

};



}

#endif // __STABILITY_CONTINUATION_HPP__
