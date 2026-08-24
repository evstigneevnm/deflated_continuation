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
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
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
#include <stability/analysis/source_parameter_path.h>
#include <stability/analysis/source_parameter_state_marcher.h>
#include <stability/persistence/classification_uncertainty_registry.h>



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

    using curve_point_type = typename bif_diag_curve_t::values_t;



    typedef common::vector_snapshot_queue<VectorOperations> queue_pointer_t;
    typedef std::deque<T> queue_lambda_t;
    typedef std::deque<std::pair<int, int>> queue_dims_t;
    typedef std::deque<uint64_t> queue_source_indices_t;
    using uncertainty_registry_t =
        stability::persistence::
            classification_uncertainty_registry<T>;
    using uncertainty_stage_t =
        stability::persistence::classification_uncertainty_stage;
    using source_path_marcher_t =
        stability::analysis::source_parameter_state_marcher<
            VectorOperations>;
    using source_path_sample_t =
        stability::analysis::source_parameter_sample<T>;
    using vector_workspace_t =
        stability::analysis::detail::vector_workspace<VectorOperations>;

public:
    using stability_result_type =
        typename stability_t::analysis_result_type;
    using stability_transition_result_type =
        typename stability_t::transition_result_type;
    struct newton_correction_replay_result
    {
        bool converged = false;
        bool used_fallback = false;
        T source_residual = T{};
        T initial_target_residual = T{};
        T final_target_residual = T{};
        std::string diagnostic;
    };
    struct source_path_topology_split_result
    {
        std::uint64_t left_source_point = 0;
        std::uint64_t right_source_point = 0;
        T left_parameter = T{};
        T right_parameter = T{};
        T state_parameter = T{};
        T aligned_relative_distance = T{};
        stability_result_type left_stability;
        stability_result_type right_stability;
        std::string diagnostic;
    };

private:
    struct pending_topology_uncertainty
    {
        std::size_t curve_number = 0;
        std::uint64_t lower_source_point = 0;
        std::uint64_t upper_source_point = 0;
        T lower_parameter = T{};
        T upper_parameter = T{};
        T failed_parameter = T{};
        std::vector<
            stability::analysis::unstable_dimension_observation>
            observations;
        std::string diagnostic;
    };

public:

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

        typename uncertainty_registry_t::options
            uncertainty_options;
        const auto& configured_uncertainty =
            parameters->stability_continuation.
                classification_uncertainty_registry;
        uncertainty_options.enabled =
            configured_uncertainty.enabled;
        std::filesystem::path uncertainty_file =
            configured_uncertainty.file_name;
        if(
            !uncertainty_file.empty() &&
            uncertainty_file.is_relative())
        {
            uncertainty_file =
                std::filesystem::path(project_dir)/uncertainty_file;
        }
        uncertainty_options.file_name =
            std::move(uncertainty_file);
        uncertainty_options.maximum_diagnostic_length =
            configured_uncertainty.maximum_diagnostic_length;
        uncertainty_options.retain_resolved =
            configured_uncertainty.retain_resolved;
        uncertainty_registry =
            new uncertainty_registry_t(
                std::move(uncertainty_options));
        if(uncertainty_registry->unresolved_count() != 0)
        {
            log->warning_f(
                "stability traversal: loaded %zu unresolved "
                "classification uncertainty record(s)",
                uncertainty_registry->unresolved_count());
        }

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

        source_path_marcher = new source_path_marcher_t(vec_ops);


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
        delete uncertainty_registry;
        delete queue_source_indices;
        delete queue_dims;
        delete queue_lambda;
        delete queue_pointer;
        delete stability_diagram;
        delete bif_diag;
        delete source_path_marcher;
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
        source_path_marcher->set_state_aligner(aligner);
    }

    void reset_transition_state_aligner()
    {
        stab->reset_transition_state_aligner();
        source_path_marcher->reset_state_aligner();
    }

    template<class FallbackNewton>
    void set_transition_fallback_newton(
        FallbackNewton* fallback_newton)
    {
        stab->set_transition_fallback_newton(
            fallback_newton);
    }

    void reset_transition_fallback_newton()
    {
        stab->reset_transition_fallback_newton();
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
        const bool recover_failed_classification =
            parameters->stability_continuation.
                recover_failed_transition_classification_with_newton;
        stab->
            set_failed_transition_classification_uses_fixed_parameter_newton(
                recover_failed_classification);
        const bool recover_failed_newton_with_homotopy =
            parameters->stability_continuation.
                recover_failed_transition_newton_with_parameter_homotopy;
        stab->set_failed_transition_newton_uses_parameter_homotopy(
            recover_failed_newton_with_homotopy);
        stab->set_transition_newton_homotopy_maximum_subdivisions(
            parameters->stability_continuation.
                transition_newton_homotopy_maximum_subdivisions);
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
            "classification confirmations = %u, failed-classification "
            "Newton recovery = %i, failed-Newton parameter homotopy = "
            "%i (maximum subdivisions = %u), symmetry endpoint guard "
            "= %u source points",
            parameters->stability_continuation.
                transition_refinement_maximum_iterations,
            double(
                parameters->stability_continuation.
                    transition_refinement_parameter_tolerance),
            parameters->stability_continuation.
                transition_refinement_maximum_subdivisions,
            parameters->stability_continuation.
                transition_classification_confirmations,
            recover_failed_classification ? 1 : 0,
            recover_failed_newton_with_homotopy ? 1 : 0,
            parameters->stability_continuation.
                transition_newton_homotopy_maximum_subdivisions,
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
        pending_topology_uncertainties.clear();
        stab->reset_recycled_subspace();
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
            const uint64_t source_point_index =
                metadata_available
                ? current_point.point_index
                : (
                      container_index > 0
                      ? static_cast<uint64_t>(
                            container_index - 1)
                      : uint64_t(0));
            
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
            std::string terminal_boundary_diagnostic;
            const bool discontinuous_terminal_boundary =
                metadata_available &&
                previous_point_available &&
                source_path_has_discontinuous_terminal_boundary(
                    curve_points,
                    previous_point,
                    current_point,
                    terminal_boundary_diagnostic);
            if(discontinuous_terminal_boundary)
            {
                curve_break = true;
                resolve_uncertainty(
                    curve_number_,
                    previous_point.point_index,
                    current_point.point_index,
                    uncertainty_stage_t::endpoint_confirmation);
                resolve_uncertainty(
                    curve_number_,
                    previous_point.point_index,
                    current_point.point_index,
                    uncertainty_stage_t::transition_refinement);
                log->warning_f(
                    "stability traversal: isolating discontinuous %s "
                    "endpoint source point %llu at lambda = %.16le "
                    "from source point %llu at %.16le; %s",
                    container::to_string(
                        current_point.endpoint_reason),
                    static_cast<unsigned long long>(
                        current_point.point_index),
                    double(current_point.lambda),
                    static_cast<unsigned long long>(
                        previous_point.point_index),
                    double(previous_point.lambda),
                    terminal_boundary_diagnostic.c_str());
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
                stab->reset_recycled_subspace();
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
                stab->reset_recycled_subspace();
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
            uncertainty_stage_t failure_stage =
                uncertainty_stage_t::point_classification;
            uint64_t failure_lower_source = source_point_index;
            uint64_t failure_upper_source = source_point_index;
            T failure_lower_parameter = lambda_p;
            T failure_upper_parameter = lambda_p;
            T failed_parameter = lambda_p;
            std::vector<
                stability::analysis::
                    unstable_dimension_observation>
                failure_observations;
            
            try
            {
                const auto stability_point =
                    parameters->stability_continuation.
                        confirm_regular_stability_points
                    ? stab->analyze_confirmed(x_p, lambda_p)
                    : stab->analyze(x_p, lambda_p);
                if(!stability_point.succeeded())
                {
                    failure_observations =
                        stability_point.
                            observed_unstable_dimensions;
                    throw std::runtime_error(
                        std::string("stability classification failed: ") +
                        stability_point.diagnostic);
                }
                std::pair<int, int> unstable_dim_p =
                    stability_point.unstable_dimension_pair();
                resolve_uncertainty(
                    curve_number_,
                    source_point_index,
                    source_point_index,
                    uncertainty_stage_t::point_classification);
                push_queue(*queue_dims, unstable_dim_p);
                push_queue(
                    *queue_source_indices,
                    source_point_index);
                std::size_t transition_count = 0;
                bool topology_split_detected = false;
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
                        failure_stage =
                            uncertainty_stage_t::endpoint_confirmation;
                        failure_lower_source =
                            queue_source_indices->at(0);
                        failure_upper_source =
                            queue_source_indices->at(1);
                        failure_lower_parameter =
                            queue_lambda->at(0);
                        failure_upper_parameter =
                            queue_lambda->at(1);
                        failed_parameter =
                            failure_lower_parameter;
                        const auto lower_confirmation =
                            stab->analyze_confirmed_independent(
                                queue_pointer->at(0),
                                queue_lambda->at(0));
                        const auto upper_confirmation =
                            stab->analyze_confirmed_independent(
                                queue_pointer->at(1),
                                queue_lambda->at(1));
                        if(!lower_confirmation.succeeded())
                        {
                            failure_observations =
                                lower_confirmation.
                                    observed_unstable_dimensions;
                            throw std::runtime_error(
                                "lower transition endpoint "
                                "confirmation failed: " +
                                lower_confirmation.diagnostic);
                        }
                        if(!upper_confirmation.succeeded())
                        {
                            failed_parameter =
                                failure_upper_parameter;
                            failure_observations =
                                upper_confirmation.
                                    observed_unstable_dimensions;
                            throw std::runtime_error(
                                "upper transition endpoint "
                                "confirmation failed: " +
                                upper_confirmation.diagnostic);
                        }
                        resolve_uncertainty(
                            curve_number_,
                            failure_lower_source,
                            failure_upper_source,
                            uncertainty_stage_t::endpoint_confirmation);

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
                            failure_stage =
                                uncertainty_stage_t::
                                    transition_refinement;
                            failed_parameter =
                                failure_lower_parameter +
                                T(0.5)*(
                                    failure_upper_parameter -
                                    failure_lower_parameter);
                            failure_observations.clear();
                            transition_count =
                                refine_transition_sequence_on_source_path(
                                    curve_number_,
                                    curve_points,
                                    queue_source_indices->at(0),
                                    queue_pointer->at(0),
                                    queue_lambda->at(0),
                                    lower_confirmation,
                                    queue_source_indices->at(1),
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
                                    },
                                    [
                                        this,
                                        curve_number_,
                                        &topology_split_detected
                                    ](
                                        const source_path_topology_split_result&
                                            split,
                                        const T_vec& left_state,
                                        const T_vec& right_state)
                                    {
                                        topology_split_detected = true;
                                        const auto before_dimension =
                                            split.left_stability.
                                                unstable_dimension_pair();
                                        const auto after_dimension =
                                            split.right_stability.
                                                unstable_dimension_pair();
                                        std::vector<T> topology_norms;
                                        calculate_plot_norms(
                                            right_state,
                                            topology_norms);
                                        stability_diagram->
                                            add_topology_break_with_plot_data(
                                                split.state_parameter,
                                                split.left_source_point,
                                                split.right_source_point,
                                                split.left_parameter,
                                                split.right_parameter,
                                                split.
                                                    aligned_relative_distance,
                                                topology_norms,
                                                before_dimension,
                                                after_dimension,
                                                left_state,
                                                right_state);

                                        std::vector<
                                            stability::analysis::
                                                unstable_dimension_observation>
                                            observations{
                                                {
                                                    split.left_stability.
                                                        unstable,
                                                    std::size_t(1)},
                                                {
                                                    split.right_stability.
                                                        unstable,
                                                    std::size_t(1)}};
                                        pending_topology_uncertainty pending;
                                        pending.curve_number =
                                            static_cast<std::size_t>(
                                                curve_number_);
                                        pending.lower_source_point =
                                            split.left_source_point;
                                        pending.upper_source_point =
                                            split.right_source_point;
                                        pending.lower_parameter =
                                            split.left_parameter;
                                        pending.upper_parameter =
                                            split.right_parameter;
                                        pending.failed_parameter =
                                            split.state_parameter;
                                        pending.observations =
                                            std::move(observations);
                                        pending.diagnostic = split.diagnostic;
                                        pending_topology_uncertainties.push_back(
                                            std::move(pending));
                                        stab->reset_recycled_subspace();
                                    });
                            stab->reset_recycled_ritz_subspace();
                        }
                        resolve_uncertainty(
                            curve_number_,
                            failure_lower_source,
                            failure_upper_source,
                            uncertainty_stage_t::
                                transition_refinement);
                    }

                }

                if(transition_count == 0 || topology_split_detected)
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
                const auto* transition_error = dynamic_cast<const
                    stability::analysis::stability_transition_error<T>*>(
                        &e);
                if(transition_error != nullptr)
                {
                    failure_observations =
                        transition_error->result().stability.
                            observed_unstable_dimensions;
                    failed_parameter =
                        transition_error->result().parameter;
                }
                if(
                    uncertainty_registry->enabled() &&
                    !uncertainty_registry->record_failure(
                        static_cast<std::size_t>(curve_number_),
                        failure_lower_source,
                        failure_lower_parameter,
                        failure_upper_source,
                        failure_upper_parameter,
                        failed_parameter,
                        failure_stage,
                        failure_observations,
                        e.what()))
                {
                    log->warning(
                        "container::stability_diagram: failed to persist "
                        "the classification uncertainty record");
                }
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

    std::vector<stability_transition_result_type>
    execute_single_transition_sequence(
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

        const stability_result_type lower_result =
            confirmed
            ? stab->analyze_confirmed_independent(
                  x_p,
                  lower_parameter)
            : stab->analyze_independent(
                  x_p,
                  lower_parameter);
        const stability_result_type upper_result =
            confirmed
            ? stab->analyze_confirmed_independent(
                  x_transition_upper,
                  upper_parameter)
            : stab->analyze_independent(
                  x_transition_upper,
                  upper_parameter);
        if(!lower_result.succeeded())
        {
            throw std::runtime_error(
                "lower transition-sequence endpoint classification "
                "failed: " + lower_result.diagnostic);
        }
        if(!upper_result.succeeded())
        {
            throw std::runtime_error(
                "upper transition-sequence endpoint classification "
                "failed: " + upper_result.diagnostic);
        }

        std::vector<stability_transition_result_type> transitions;
        const auto collect_transition =
            [&transitions](
                const stability_transition_result_type& transition,
                const T_vec&)
            {
                transitions.push_back(transition);
            };
        const std::size_t transition_count =
            confirmed
            ? stab->refine_transition_sequence_confirmed(
                  x_p,
                  lower_parameter,
                  lower_result,
                  x_transition_upper,
                  upper_parameter,
                  upper_result,
                  collect_transition)
            : stab->refine_transition_sequence_known(
                  x_p,
                  lower_parameter,
                  lower_result.unstable_dimension_pair(),
                  x_transition_upper,
                  upper_parameter,
                  upper_result.unstable_dimension_pair(),
                  collect_transition);
        if(transition_count != transitions.size())
        {
            throw std::logic_error(
                "transition-sequence callback count mismatch");
        }
        log->info_f(
            "two-state stability transition-sequence replay: "
            "lower = %s at %.16le, upper = %s at %.16le, "
            "events = %zu, confirmed = %i",
            lower_state_file.c_str(),
            double(lower_parameter),
            upper_state_file.c_str(),
            double(upper_parameter),
            transition_count,
            confirmed ? 1 : 0);
        return transitions;
    }

    std::vector<stability_transition_result_type>
    execute_curve_transition_sequence(
        int curve_number,
        std::uint64_t lower_source_point,
        std::uint64_t upper_source_point)
    {
        if(!load_diagram_data(
            parameters->bifurcaiton_diagram_file_name))
        {
            throw std::runtime_error(
                "failed to load the bifurcation diagram for "
                "curve-transition replay");
        }

        const std::vector<curve_point_type> curve_points =
            bif_diag->get_curve_points_vector(curve_number);
        if(curve_points.empty())
        {
            throw std::runtime_error(
                "curve-transition replay requested an empty or "
                "unavailable curve");
        }

        const curve_point_type* lower_point = nullptr;
        const curve_point_type* upper_point = nullptr;
        for(const auto& point: curve_points)
        {
            if(point.point_index == lower_source_point)
                lower_point = &point;
            if(point.point_index == upper_source_point)
                upper_point = &point;
        }
        if(lower_point == nullptr || upper_point == nullptr)
        {
            throw std::runtime_error(
                "curve-transition replay source-point metadata is "
                "missing");
        }
        std::string terminal_boundary_diagnostic;
        if(source_path_has_discontinuous_terminal_boundary(
            curve_points,
            *lower_point,
            *upper_point,
            terminal_boundary_diagnostic))
        {
            log->warning_f(
                "archive curve-transition replay: source interval "
                "[%llu,%llu] ends at a discontinuous %s endpoint and "
                "is a stability-refinement barrier; %s",
                static_cast<unsigned long long>(lower_source_point),
                static_cast<unsigned long long>(upper_source_point),
                container::to_string(upper_point->endpoint_reason),
                terminal_boundary_diagnostic.c_str());
            return {};
        }
        if(!bif_diag->read_saved_solution_from_curve(
            curve_number,
            lower_source_point,
            x_p))
        {
            throw std::runtime_error(
                "curve-transition replay lower source point has no "
                "saved state");
        }
        if(!bif_diag->read_saved_solution_from_curve(
            curve_number,
            upper_source_point,
            x_transition_upper))
        {
            throw std::runtime_error(
                "curve-transition replay upper source point has no "
                "saved state");
        }

        log->info_f(
            "archive curve-transition replay: confirming curve %i "
            "endpoint source %llu at %.16le",
            curve_number,
            static_cast<unsigned long long>(lower_source_point),
            double(lower_point->lambda));
        const stability_result_type lower_result =
            stab->analyze_confirmed_independent(
                x_p,
                lower_point->lambda);
        log->info_f(
            "archive curve-transition replay: confirming curve %i "
            "endpoint source %llu at %.16le",
            curve_number,
            static_cast<unsigned long long>(upper_source_point),
            double(upper_point->lambda));
        const stability_result_type upper_result =
            stab->analyze_confirmed_independent(
                x_transition_upper,
                upper_point->lambda);
        if(!lower_result.succeeded())
        {
            throw std::runtime_error(
                "curve-transition replay lower endpoint "
                "classification failed: " + lower_result.diagnostic);
        }
        if(!upper_result.succeeded())
        {
            throw std::runtime_error(
                "curve-transition replay upper endpoint "
                "classification failed: " + upper_result.diagnostic);
        }

        std::vector<stability_transition_result_type> transitions;
        const auto collect_transition =
            [&transitions](
                const stability_transition_result_type& transition,
                const T_vec&)
            {
                transitions.push_back(transition);
            };
        std::size_t topology_split_count = 0;
        const auto collect_topology_split =
            [this, curve_number, &topology_split_count](
                const source_path_topology_split_result& split,
                const T_vec&,
                const T_vec&)
            {
                ++topology_split_count;
                log->warning_f(
                    "archive curve-transition replay: curve %i has an "
                    "unresolved source-path topology split across sources "
                    "[%llu,%llu] at lambda = %.16le, aligned distance = "
                    "%.6e",
                    curve_number,
                    static_cast<unsigned long long>(
                        split.left_source_point),
                    static_cast<unsigned long long>(
                        split.right_source_point),
                    double(split.state_parameter),
                    double(split.aligned_relative_distance));
            };
        const std::size_t transition_count =
            refine_transition_sequence_on_source_path(
                curve_number,
                curve_points,
                lower_source_point,
                x_p,
                lower_point->lambda,
                lower_result,
                upper_source_point,
                x_transition_upper,
                upper_point->lambda,
                upper_result,
                collect_transition,
                collect_topology_split);
        if(transition_count != transitions.size())
        {
            throw std::logic_error(
                "curve-transition replay callback count mismatch");
        }
        log->info_f(
            "archive curve-transition replay: curve = %i, source "
            "interval = [%llu,%llu], lambda interval = "
            "[%.16le,%.16le], events = %zu, topology splits = %zu",
            curve_number,
            static_cast<unsigned long long>(lower_source_point),
            static_cast<unsigned long long>(upper_source_point),
            double(lower_point->lambda),
            double(upper_point->lambda),
            transition_count,
            topology_split_count);
        return transitions;
    }

    newton_correction_replay_result
    execute_single_newton_correction(
        const std::string& state_file,
        T source_parameter,
        T target_parameter)
    {
        file_ops->read_vector(state_file, x_p);
        newton_correction_replay_result result;
        nonlin_op->F(x_p, source_parameter, x_transition_upper);
        result.source_residual = vec_ops->norm(x_transition_upper);
        nonlin_op->F(x_p, target_parameter, x_transition_upper);
        result.initial_target_residual =
            vec_ops->norm(x_transition_upper);
        const auto correction =
            stab->correct_fixed_parameter_state(
            x_p,
            target_parameter);
        result.converged = correction.succeeded;
        result.used_fallback = correction.used_fallback;
        result.diagnostic = correction.diagnostic;
        nonlin_op->F(x_p, target_parameter, x_transition_upper);
        result.final_target_residual =
            vec_ops->norm(x_transition_upper);
        log->info_f(
            "fixed-parameter Newton replay: state = %s, source "
            "lambda = %.16le, target lambda = %.16le, source "
            "residual = %.6e, initial target residual = %.6e, "
            "final target residual = %.6e, converged = %i, fallback "
            "= %i, diagnostic = %s",
            state_file.c_str(),
            double(source_parameter),
            double(target_parameter),
            double(result.source_residual),
            double(result.initial_target_residual),
            double(result.final_target_residual),
            result.converged ? 1 : 0,
            result.used_fallback ? 1 : 0,
            result.diagnostic.c_str());
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
                    pending_topology_uncertainties.clear();
                    throw std::runtime_error(
                        "failed to traverse bifurcation curve " +
                        std::to_string(curve_number));
                }
                stability_diagram->close_curve();
                save_stability_data(file_name_stability_);
                persist_pending_topology_uncertainties();
                ++curve_number;
            }
        }

    }


private:
    struct source_path_interval
    {
        std::vector<source_path_sample_t> samples;
        stability::analysis::source_parameter_path_result<T> geometry;
    };

    source_path_interval make_source_path_interval(
        const std::vector<curve_point_type>& curve_points,
        std::uint64_t lower_source_point,
        std::uint64_t upper_source_point) const
    {
        source_path_interval result;
        if(lower_source_point >= upper_source_point)
        {
            throw std::invalid_argument(
                "stability source path requires increasing source "
                "point indices");
        }
        const curve_point_type* lower = nullptr;
        const curve_point_type* upper = nullptr;
        for(const auto& point: curve_points)
        {
            if(point.point_index == lower_source_point)
                lower = &point;
            if(point.point_index == upper_source_point)
                upper = &point;
            if(
                point.point_index >= lower_source_point &&
                point.point_index <= upper_source_point)
            {
                result.samples.push_back(
                    {point.point_index, point.lambda});
            }
        }
        if(lower == nullptr || upper == nullptr)
        {
            throw std::runtime_error(
                "stability source path endpoint metadata is missing");
        }
        if(
            lower->segment_id != upper->segment_id ||
            lower->semicurve_id != upper->semicurve_id)
        {
            throw std::runtime_error(
                "stability source path crosses a continuation segment "
                "boundary");
        }
        result.geometry =
            stability::analysis::analyze_source_parameter_path(
                result.samples);
        if(!result.geometry.valid)
        {
            throw std::runtime_error(
                "invalid stability source path: " +
                result.geometry.diagnostic);
        }
        return result;
    }

    bool source_path_has_discontinuous_terminal_boundary(
        const std::vector<curve_point_type>& curve_points,
        const curve_point_type& lower,
        const curve_point_type& upper,
        std::string& diagnostic) const
    {
        diagnostic.clear();
        if(
            !container::is_parameter_boundary_endpoint(
                upper.endpoint_reason) ||
            lower.point_index >= upper.point_index ||
            lower.segment_id != upper.segment_id ||
            lower.semicurve_id != upper.semicurve_id)
        {
            return false;
        }

        const source_path_interval path = make_source_path_interval(
            curve_points,
            lower.point_index,
            upper.point_index);
        const auto terminal =
            stability::analysis::analyze_source_parameter_terminal_step(
                path.samples);
        if(!terminal.valid || !terminal.discontinuous)
            return false;

        std::ostringstream message;
        message
            << terminal.diagnostic
            << ", terminal step = " << terminal.terminal_step
            << ", maximum preceding step = "
            << terminal.reference_step
            << ", ratio = " << terminal.step_ratio;
        diagnostic = message.str();
        return true;
    }

    typename source_path_marcher_t::correction_result
    correct_source_path_state(T_vec& state, T parameter)
    {
        const auto corrected =
            stab->correct_fixed_parameter_state(state, parameter);
        return {
            corrected.succeeded,
            corrected.used_fallback,
            corrected.diagnostic};
    }

    template<class EventCallback, class TopologyCallback>
    std::size_t
    refine_multiple_turn_transition_sequence_on_source_path(
        int curve_number,
        const source_path_interval& path,
        std::uint64_t lower_source_point,
        const T_vec& lower_state,
        T lower_parameter,
        const stability_result_type& lower_stability,
        std::uint64_t upper_source_point,
        const T_vec& upper_state,
        T upper_parameter,
        const stability_result_type& upper_stability,
        EventCallback& on_event,
        TopologyCallback& on_topology_split)
    {
        vector_workspace_t current_state_workspace(vec_ops);
        vector_workspace_t reconstructed_upper_workspace(vec_ops);
        vector_workspace_t aligned_upper_workspace(vec_ops);
        vector_workspace_t difference_workspace(vec_ops);
        vector_workspace_t topology_left_workspace(vec_ops);
        vector_workspace_t topology_right_workspace(vec_ops);
        T_vec& current_state = current_state_workspace.get();
        T_vec& reconstructed_upper = reconstructed_upper_workspace.get();
        T_vec& aligned_upper = aligned_upper_workspace.get();
        T_vec& difference = difference_workspace.get();
        T_vec& topology_left_state = topology_left_workspace.get();
        T_vec& topology_right_state = topology_right_workspace.get();
        vec_ops->assign(lower_state, current_state);
        T current_parameter = lower_parameter;
        stability_result_type current_stability = lower_stability;
        std::size_t current_source_position = 0;
        std::size_t event_count = 0;
        const std::size_t turning_guard_source_points =
            parameters->stability_continuation.
                turning_point_guard_source_points;

        const auto correction =
            [this](T_vec& state, T parameter)
            {
                return correct_source_path_state(state, parameter);
            };
        const auto process_turning_point =
            [
                this,
                curve_number,
                &path,
                &current_state,
                &current_parameter,
                &current_stability,
                &current_source_position,
                &event_count,
                &on_event,
                &correction,
                turning_guard_source_points
            ](
                std::size_t turning_index,
                const stability::analysis::
                    source_parameter_turning_estimate<T>& turning,
                const T_vec& left_guard_state,
                const T_vec& right_guard_state,
                const typename source_path_marcher_t::result_type&
                    approach_result,
                const typename source_path_marcher_t::result_type&
                    crossing_result)
            {
                const std::size_t left_classification_position =
                    std::max(
                        current_source_position,
                        turning.left_position >
                                turning_guard_source_points
                            ? turning.left_position -
                                turning_guard_source_points
                            : std::size_t(0));
                std::size_t right_classification_limit =
                    path.samples.size() - 1;
                if(
                    turning_index + 1 <
                        path.geometry.turning_positions.size())
                {
                    const auto next_turning = stability::analysis::
                        estimate_source_parameter_turning_point(
                            path.samples,
                            path.geometry.turning_positions[
                                turning_index + 1]);
                    if(!next_turning.valid)
                    {
                        throw std::runtime_error(
                            "failed to estimate the next source-path "
                            "turning point while selecting guards: " +
                            next_turning.diagnostic);
                    }
                    right_classification_limit =
                        next_turning.left_position;
                }
                const std::size_t right_classification_position =
                    std::min(
                        right_classification_limit,
                        turning.right_position +
                            turning_guard_source_points);
                vector_workspace_t left_classification_workspace(
                    vec_ops);
                vector_workspace_t right_classification_workspace(
                    vec_ops);
                T_vec& left_classification_state =
                    left_classification_workspace.get();
                T_vec& right_classification_state =
                    right_classification_workspace.get();
                typename source_path_marcher_t::result_type
                    left_guard_march;
                typename source_path_marcher_t::result_type
                    right_guard_march;
                if(
                    left_classification_position ==
                        turning.left_position)
                {
                    vec_ops->assign(
                        left_guard_state,
                        left_classification_state);
                    left_guard_march.succeeded = true;
                }
                else
                {
                    left_guard_march = source_path_marcher->march(
                        current_state,
                        path.samples,
                        current_source_position,
                        left_classification_position,
                        correction,
                        left_classification_state);
                }
                if(!left_guard_march.succeeded)
                {
                    throw std::runtime_error(
                        "failed to reconstruct the outward left "
                        "turning-point classification guard: " +
                        left_guard_march.diagnostic);
                }
                if(
                    right_classification_position ==
                        turning.right_position)
                {
                    vec_ops->assign(
                        right_guard_state,
                        right_classification_state);
                    right_guard_march.succeeded = true;
                }
                else
                {
                    right_guard_march = source_path_marcher->march(
                        right_guard_state,
                        path.samples,
                        turning.right_position,
                        right_classification_position,
                        correction,
                        right_classification_state);
                }
                if(!right_guard_march.succeeded)
                {
                    throw std::runtime_error(
                        "failed to reconstruct the outward right "
                        "turning-point classification guard: " +
                        right_guard_march.diagnostic);
                }

                const T left_parameter = path.samples[
                    left_classification_position].parameter;
                const T right_parameter = path.samples[
                    right_classification_position].parameter;
                log->info_f(
                    "stability source path turn %zu/%zu on curve %i: "
                    "crossing sources [%llu,%llu], classification "
                    "sources [%llu,%llu], classification lambdas = "
                    "[%.16le,%.16le], estimated turning lambda = "
                    "%.16le; approach = %zu step(s), crossing "
                    "fallbacks = %u",
                    turning_index + 1,
                    path.geometry.turning_positions.size(),
                    curve_number,
                    static_cast<unsigned long long>(
                        path.samples[turning.left_position].source_index),
                    static_cast<unsigned long long>(
                        path.samples[turning.right_position].source_index),
                    static_cast<unsigned long long>(
                        path.samples[left_classification_position].
                            source_index),
                    static_cast<unsigned long long>(
                        path.samples[right_classification_position].
                            source_index),
                    double(left_parameter),
                    double(right_parameter),
                    double(turning.parameter),
                    approach_result.completed_steps,
                    crossing_result.fallback_recoveries);

                const auto left_stability =
                    stab->analyze_confirmed_independent(
                        left_classification_state,
                        left_parameter);
                if(!left_stability.succeeded())
                {
                    throw std::runtime_error(
                        "stability classification failed at "
                        "reconstructed left guard of turning point " +
                        std::to_string(turning_index) + ": " +
                        left_stability.diagnostic);
                }
                const auto right_stability =
                    stab->analyze_confirmed_independent(
                        right_classification_state,
                        right_parameter);
                if(!right_stability.succeeded())
                {
                    throw std::runtime_error(
                        "stability classification failed at "
                        "reconstructed right guard of turning point " +
                        std::to_string(turning_index) + ": " +
                        right_stability.diagnostic);
                }

                const int current_dimension =
                    current_stability.unstable.
                        real_subspace_dimension();
                const int left_dimension =
                    left_stability.unstable.real_subspace_dimension();
                const int right_dimension =
                    right_stability.unstable.real_subspace_dimension();
                if(current_dimension != left_dimension)
                {
                    event_count +=
                        stab->refine_transition_sequence_confirmed(
                            current_state,
                            current_parameter,
                            current_stability,
                            left_classification_state,
                            left_parameter,
                            left_stability,
                            on_event);
                }
                if(left_dimension != right_dimension)
                {
                    stability_transition_result_type fold_transition;
                    fold_transition.status = stability::analysis::
                        stability_transition_status::success;
                    fold_transition.parameter = turning.parameter;
                    fold_transition.stability = right_stability;
                    fold_transition.before_stability = left_stability;
                    fold_transition.after_stability = right_stability;
                    fold_transition.diagnostic =
                        "stability transition localized at a "
                        "source-path turning point without parameter "
                        "interpolation";
                    vector_workspace_t event_state_workspace(vec_ops);
                    T_vec& event_state = event_state_workspace.get();
                    vec_ops->assign_mul(
                        T(1) - turning.right_fraction,
                        left_guard_state,
                        turning.right_fraction,
                        right_guard_state,
                        event_state);
                    on_event(fold_transition, event_state);
                    ++event_count;
                    log->info_f(
                        "stability source-path turning event on curve "
                        "%i at lambda = %.16le, dim(U): before = "
                        "(%i,%i), after = (%i,%i)",
                        curve_number,
                        double(turning.parameter),
                        left_stability.unstable.real,
                        left_stability.unstable.complex_pairs,
                        right_stability.unstable.real,
                        right_stability.unstable.complex_pairs);
                }

                vec_ops->assign(
                    right_classification_state,
                    current_state);
                current_parameter = right_parameter;
                current_stability = right_stability;
                current_source_position =
                    right_classification_position;
            };

        log->info_f(
            "stability source path decomposes interval [%llu,%llu] "
            "into %zu monotone spans separated by %zu turning points",
            static_cast<unsigned long long>(lower_source_point),
            static_cast<unsigned long long>(upper_source_point),
            path.geometry.monotone_spans.size(),
            path.geometry.turning_positions.size());
        const T matching_tolerance =
            T(32)*std::sqrt(std::numeric_limits<T>::epsilon());
        const auto march_result =
            source_path_marcher->march_through_turning_points(
                lower_state,
                path.samples,
                0,
                path.samples.size() - 1,
                path.geometry.turning_positions,
                correction,
                process_turning_point,
                reconstructed_upper,
                &upper_state,
                matching_tolerance,
                parameters->stability_continuation.
                        allow_source_path_topology_splits
                    ? &topology_left_state
                    : nullptr,
                parameters->stability_continuation.
                        allow_source_path_topology_splits
                    ? &topology_right_state
                    : nullptr);
        if(!march_result.succeeded)
        {
            throw std::runtime_error(
                "failed to reconstruct a multiple-turn stability "
                "source path: " + march_result.diagnostic);
        }
        if(march_result.used_bidirectional_recovery)
        {
            log->warning_f(
                "stability multiple-turn source path used saved-anchor "
                "bidirectional recovery at source %llu with relative "
                "aligned join distance %.6e; forward steps = %zu, "
                "reverse steps = %zu",
                static_cast<unsigned long long>(
                    march_result.bidirectional_join_source_index),
                double(
                    march_result.bidirectional_join_relative_distance),
                march_result.completed_steps -
                    march_result.reverse_completed_steps,
                march_result.reverse_completed_steps);
        }
        if(march_result.topology_split_detected)
        {
            const T topology_parameter =
                path.samples[
                    march_result.topology_split_right_position].parameter;
            const auto left_topology_stability =
                stab->analyze_confirmed_independent(
                    topology_left_state,
                    topology_parameter);
            if(!left_topology_stability.succeeded())
            {
                throw std::runtime_error(
                    "stability classification failed on the left side "
                    "of a source-path topology split: " +
                    left_topology_stability.diagnostic);
            }
            const auto right_topology_stability =
                stab->analyze_confirmed_independent(
                    topology_right_state,
                    topology_parameter);
            if(!right_topology_stability.succeeded())
            {
                throw std::runtime_error(
                    "stability classification failed on the right side "
                    "of a source-path topology split: " +
                    right_topology_stability.diagnostic);
            }

            if(
                current_stability.unstable.real_subspace_dimension() !=
                left_topology_stability.unstable.
                    real_subspace_dimension())
            {
                event_count += stab->refine_transition_sequence_confirmed(
                    current_state,
                    current_parameter,
                    current_stability,
                    topology_left_state,
                    topology_parameter,
                    left_topology_stability,
                    on_event);
            }

            source_path_topology_split_result split;
            split.left_source_point =
                march_result.topology_split_left_source_index;
            split.right_source_point =
                march_result.topology_split_right_source_index;
            split.left_parameter =
                march_result.topology_split_left_parameter;
            split.right_parameter =
                march_result.topology_split_right_parameter;
            split.state_parameter = topology_parameter;
            split.aligned_relative_distance =
                march_result.bidirectional_join_relative_distance;
            split.left_stability = left_topology_stability;
            split.right_stability = right_topology_stability;
            split.diagnostic = march_result.diagnostic;
            on_topology_split(
                split,
                topology_left_state,
                topology_right_state);

            stab->align_transition_state(
                topology_right_state,
                upper_state,
                aligned_upper);
            if(
                right_topology_stability.unstable.
                        real_subspace_dimension() !=
                    upper_stability.unstable.real_subspace_dimension())
            {
                event_count += stab->refine_transition_sequence_confirmed(
                    topology_right_state,
                    topology_parameter,
                    right_topology_stability,
                    aligned_upper,
                    upper_parameter,
                    upper_stability,
                    on_event);
            }

            log->warning_f(
                "stability source path split curve %i across archive "
                "sources [%llu,%llu] at lambda = %.16le; aligned state "
                "distance = %.6e, dim(U): left = (%i,%i), right = "
                "(%i,%i)",
                curve_number,
                static_cast<unsigned long long>(
                    split.left_source_point),
                static_cast<unsigned long long>(
                    split.right_source_point),
                double(split.state_parameter),
                double(split.aligned_relative_distance),
                split.left_stability.unstable.real,
                split.left_stability.unstable.complex_pairs,
                split.right_stability.unstable.real,
                split.right_stability.unstable.complex_pairs);
            return event_count;
        }

        stab->align_transition_state(
            reconstructed_upper,
            upper_state,
            aligned_upper);
        vec_ops->assign_mul(
            T(1),
            reconstructed_upper,
            T(-1),
            aligned_upper,
            difference);
        const T endpoint_scale = T(1) + std::max(
            vec_ops->norm(reconstructed_upper),
            vec_ops->norm(aligned_upper));
        const T endpoint_relative_distance =
            vec_ops->norm(difference)/endpoint_scale;
        log->info_f(
            "stability multiple-turn source path reached saved upper "
            "anchor with relative state distance %.6e after %zu "
            "source step(s)",
            double(endpoint_relative_distance),
            march_result.completed_steps);
        if(endpoint_relative_distance > matching_tolerance)
        {
            throw std::runtime_error(
                "multiple-turn source-path reconstruction reached a "
                "different terminal branch representative");
        }

        const int current_dimension =
            current_stability.unstable.real_subspace_dimension();
        const int upper_dimension =
            upper_stability.unstable.real_subspace_dimension();
        if(current_dimension != upper_dimension)
        {
            event_count += stab->refine_transition_sequence_confirmed(
                current_state,
                current_parameter,
                current_stability,
                aligned_upper,
                upper_parameter,
                upper_stability,
                on_event);
        }
        if(
            event_count == 0 &&
            lower_stability.unstable.real_subspace_dimension() !=
                upper_dimension)
        {
            throw std::runtime_error(
                "multiple-turn source-path subdivision lost the "
                "endpoint dimension change");
        }
        return event_count;
    }

    template<class EventCallback, class TopologyCallback>
    std::size_t refine_transition_sequence_on_source_path(
        int curve_number,
        const std::vector<curve_point_type>& curve_points,
        std::uint64_t lower_source_point,
        const T_vec& lower_state,
        T lower_parameter,
        const stability_result_type& lower_stability,
        std::uint64_t upper_source_point,
        const T_vec& upper_state,
        T upper_parameter,
        const stability_result_type& upper_stability,
        EventCallback&& on_event,
        TopologyCallback&& on_topology_split)
    {
        const source_path_interval path = make_source_path_interval(
            curve_points,
            lower_source_point,
            upper_source_point);
        if(path.geometry.monotone)
        {
            return stab->refine_transition_sequence_confirmed(
                lower_state,
                lower_parameter,
                lower_stability,
                upper_state,
                upper_parameter,
                upper_stability,
                std::forward<EventCallback>(on_event));
        }
        if(path.geometry.turning_positions.size() > 1)
        {
            return refine_multiple_turn_transition_sequence_on_source_path(
                curve_number,
                path,
                lower_source_point,
                lower_state,
                lower_parameter,
                lower_stability,
                upper_source_point,
                upper_state,
                upper_parameter,
                upper_stability,
                on_event,
                on_topology_split);
        }

        const std::size_t turning_position =
            path.geometry.turning_positions.front();
        const auto turning =
            stability::analysis::estimate_source_parameter_turning_point(
                path.samples,
                turning_position);
        if(!turning.valid)
        {
            throw std::runtime_error(
                "failed to estimate the stability source-path turning "
                "point: " + turning.diagnostic);
        }
        const std::size_t left_guard_position = turning.left_position;
        const std::size_t right_guard_position = turning.right_position;
        const T left_guard_parameter =
            path.samples[left_guard_position].parameter;
        const T right_guard_parameter =
            path.samples[right_guard_position].parameter;
        vector_workspace_t left_workspace(vec_ops);
        vector_workspace_t right_workspace(vec_ops);
        T_vec& x_path_left = left_workspace.get();
        T_vec& x_path_right = right_workspace.get();
        typename source_path_marcher_t::result_type left_march;
        typename source_path_marcher_t::result_type right_march;
        bool turning_topology_split = false;
        T turning_guard_relative_distance = T{};
        T forward_cross_relative_distance =
            std::numeric_limits<T>::infinity();
        T reverse_cross_relative_distance =
            std::numeric_limits<T>::infinity();
        std::string turning_topology_diagnostic;
        {
            vector_workspace_t left_predecessor_workspace(vec_ops);
            vector_workspace_t right_predecessor_workspace(vec_ops);
            vector_workspace_t cross_workspace(vec_ops);
            vector_workspace_t difference_workspace(vec_ops);
            T_vec& x_path_left_predecessor =
                left_predecessor_workspace.get();
            T_vec& x_path_right_predecessor =
                right_predecessor_workspace.get();
            T_vec& x_path_cross = cross_workspace.get();
            T_vec& x_path_difference = difference_workspace.get();
            const auto correction =
                [this](T_vec& state, T parameter)
                {
                    return correct_source_path_state(state, parameter);
                };
            left_march = source_path_marcher->march(
                lower_state,
                path.samples,
                0,
                left_guard_position,
                correction,
                x_path_left,
                &x_path_left_predecessor);
            right_march = source_path_marcher->march(
                upper_state,
                path.samples,
                path.samples.size() - 1,
                right_guard_position,
                correction,
                x_path_right,
                &x_path_right_predecessor);
            if(!left_march.succeeded && !right_march.succeeded)
            {
                throw std::runtime_error(
                    "failed to reconstruct either one-sided stability "
                    "path turning guard: left={" +
                    left_march.diagnostic + "}; right={" +
                    right_march.diagnostic + "}");
            }

            typename source_path_marcher_t::result_type forward_cross;
            typename source_path_marcher_t::result_type reverse_cross;
            bool forward_matches = false;
            bool reverse_matches = false;
            const T matching_tolerance =
                T(32)*std::sqrt(std::numeric_limits<T>::epsilon());
            if(
                left_march.succeeded &&
                left_march.terminal_predecessor_available)
            {
                forward_cross =
                    source_path_marcher->advance_with_secant_predictor(
                        x_path_left_predecessor,
                        x_path_left,
                        path.samples[right_guard_position],
                        correction,
                        x_path_cross);
                if(forward_cross.succeeded && right_march.succeeded)
                {
                    stab->align_transition_state(
                        x_path_right,
                        x_path_cross,
                        x_path_cross);
                    vec_ops->assign_mul(
                        T(1),
                        x_path_right,
                        T(-1),
                        x_path_cross,
                        x_path_difference);
                    const T scale = T(1) + std::max(
                        vec_ops->norm(x_path_right),
                        vec_ops->norm(x_path_cross));
                    forward_cross_relative_distance =
                        vec_ops->norm(x_path_difference)/scale;
                    forward_matches =
                        forward_cross_relative_distance <=
                            matching_tolerance;
                }
                else if(forward_cross.succeeded)
                {
                    vec_ops->assign(x_path_cross, x_path_right);
                    forward_cross_relative_distance = T{};
                    forward_matches = true;
                }
            }
            if(
                right_march.succeeded &&
                right_march.terminal_predecessor_available)
            {
                reverse_cross =
                    source_path_marcher->advance_with_secant_predictor(
                        x_path_right_predecessor,
                        x_path_right,
                        path.samples[left_guard_position],
                        correction,
                        x_path_cross);
                if(reverse_cross.succeeded && left_march.succeeded)
                {
                    stab->align_transition_state(
                        x_path_left,
                        x_path_cross,
                        x_path_cross);
                    vec_ops->assign_mul(
                        T(1),
                        x_path_left,
                        T(-1),
                        x_path_cross,
                        x_path_difference);
                    const T scale = T(1) + std::max(
                        vec_ops->norm(x_path_left),
                        vec_ops->norm(x_path_cross));
                    reverse_cross_relative_distance =
                        vec_ops->norm(x_path_difference)/scale;
                    reverse_matches =
                        reverse_cross_relative_distance <=
                            matching_tolerance;
                }
                else if(reverse_cross.succeeded)
                {
                    vec_ops->assign(x_path_cross, x_path_left);
                    reverse_cross_relative_distance = T{};
                    reverse_matches = true;
                }
            }
            if(!left_march.succeeded && !reverse_matches)
            {
                throw std::runtime_error(
                    "failed to reconstruct the left source-path turning "
                    "guard: one-sided={" + left_march.diagnostic +
                    "}; reverse secant={" + reverse_cross.diagnostic +
                    "}");
            }
            if(!right_march.succeeded && !forward_matches)
            {
                throw std::runtime_error(
                    "failed to reconstruct the right source-path turning "
                    "guard: one-sided={" + right_march.diagnostic +
                    "}; forward secant={" + forward_cross.diagnostic +
                    "}");
            }
            if(left_march.succeeded && right_march.succeeded)
            {
                stab->align_transition_state(
                    x_path_left,
                    x_path_right,
                    x_path_right);
                vec_ops->assign_mul(
                    T(1),
                    x_path_left,
                    T(-1),
                    x_path_right,
                    x_path_difference);
                const T scale = T(1) + std::max(
                    vec_ops->norm(x_path_left),
                    vec_ops->norm(x_path_right));
                turning_guard_relative_distance =
                    vec_ops->norm(x_path_difference)/scale;
                log->info_f(
                    "stability source-path one-sided turning guards have "
                    "relative state distance %.6e; forward crossing "
                    "distance = %.6e (match = %i), reverse crossing "
                    "distance = %.6e (match = %i)",
                    double(turning_guard_relative_distance),
                    double(forward_cross_relative_distance),
                    forward_matches ? 1 : 0,
                    double(reverse_cross_relative_distance),
                    reverse_matches ? 1 : 0);

                const auto join_decision =
                    stability::analysis::decide_source_path_turning_join(
                        path.samples[left_guard_position].source_index,
                        path.samples[right_guard_position].source_index,
                        turning_guard_relative_distance,
                        forward_cross_relative_distance,
                        reverse_cross_relative_distance,
                        matching_tolerance,
                        parameters->stability_continuation.
                            allow_source_path_topology_splits);
                forward_matches = join_decision.forward_matches;
                reverse_matches = join_decision.reverse_matches;
                if(
                    join_decision.status ==
                    stability::analysis::
                        source_path_turning_join_status::topology_split)
                {
                    turning_topology_split = true;
                    turning_topology_diagnostic =
                        join_decision.diagnostic;
                    log->warning(
                        "stability source path preserves a non-joining "
                        "one-turn bracket as a topology split: " +
                        turning_topology_diagnostic);
                }
                else if(!join_decision.accepted())
                {
                    throw std::runtime_error(join_decision.diagnostic);
                }
            }
        }

        log->info_f(
            "stability source path split non-monotone bracket "
            "[%llu,%llu] across guard source points [%llu,%llu], "
            "guard lambdas = [%.16le,%.16le], estimated turning "
            "lambda = %.16le; left march = %zu step(s), right march "
            "= %zu step(s)",
            static_cast<unsigned long long>(lower_source_point),
            static_cast<unsigned long long>(upper_source_point),
            static_cast<unsigned long long>(
                path.samples[left_guard_position].source_index),
            static_cast<unsigned long long>(
                path.samples[right_guard_position].source_index),
            double(left_guard_parameter),
            double(right_guard_parameter),
            double(turning.parameter),
            left_march.completed_steps,
            right_march.completed_steps);

        log->info_f(
            "stability source path: confirming left turning guard "
            "source %llu at %.16le",
            static_cast<unsigned long long>(
                path.samples[left_guard_position].source_index),
            double(left_guard_parameter));
        const auto left_guard_stability =
            stab->analyze_confirmed_independent(
                x_path_left,
                left_guard_parameter);
        if(!left_guard_stability.succeeded())
        {
            throw std::runtime_error(
                "stability classification failed at reconstructed "
                "left turning guard: " +
                left_guard_stability.diagnostic);
        }
        log->info_f(
            "stability source path: confirming right turning guard "
            "source %llu at %.16le",
            static_cast<unsigned long long>(
                path.samples[right_guard_position].source_index),
            double(right_guard_parameter));
        const auto right_guard_stability =
            stab->analyze_confirmed_independent(
                x_path_right,
                right_guard_parameter);
        if(!right_guard_stability.succeeded())
        {
            throw std::runtime_error(
                "stability classification failed at reconstructed "
                "right turning guard: " +
                right_guard_stability.diagnostic);
        }

        const int lower_dimension =
            lower_stability.unstable.real_subspace_dimension();
        const int left_guard_dimension =
            left_guard_stability.unstable.real_subspace_dimension();
        const int right_guard_dimension =
            right_guard_stability.unstable.real_subspace_dimension();
        const int upper_dimension =
            upper_stability.unstable.real_subspace_dimension();
        std::size_t event_count = 0;
        if(lower_dimension != left_guard_dimension)
        {
            event_count += stab->refine_transition_sequence_confirmed(
                lower_state,
                lower_parameter,
                lower_stability,
                x_path_left,
                left_guard_parameter,
                left_guard_stability,
                on_event);
        }
        if(turning_topology_split)
        {
            source_path_topology_split_result split;
            split.left_source_point =
                path.samples[left_guard_position].source_index;
            split.right_source_point =
                path.samples[right_guard_position].source_index;
            split.left_parameter = left_guard_parameter;
            split.right_parameter = right_guard_parameter;
            split.state_parameter = turning.parameter;
            split.aligned_relative_distance =
                turning_guard_relative_distance;
            split.left_stability = left_guard_stability;
            split.right_stability = right_guard_stability;
            split.diagnostic = turning_topology_diagnostic;
            on_topology_split(split, x_path_left, x_path_right);
            log->warning_f(
                "stability source path split curve %i at a non-joining "
                "one-turn bracket across sources [%llu,%llu], estimated "
                "lambda = %.16le, aligned guard distance = %.6e, "
                "dim(U): left = (%i,%i), right = (%i,%i)",
                curve_number,
                static_cast<unsigned long long>(
                    split.left_source_point),
                static_cast<unsigned long long>(
                    split.right_source_point),
                double(split.state_parameter),
                double(split.aligned_relative_distance),
                split.left_stability.unstable.real,
                split.left_stability.unstable.complex_pairs,
                split.right_stability.unstable.real,
                split.right_stability.unstable.complex_pairs);
        }
        else if(left_guard_dimension != right_guard_dimension)
        {
            stability_transition_result_type fold_transition;
            fold_transition.status =
                stability::analysis::stability_transition_status::success;
            fold_transition.parameter = turning.parameter;
            fold_transition.stability = right_guard_stability;
            fold_transition.before_stability = left_guard_stability;
            fold_transition.after_stability = right_guard_stability;
            fold_transition.diagnostic =
                "stability transition localized at a source-path "
                "turning point without parameter interpolation";
            vector_workspace_t event_state_workspace(vec_ops);
            T_vec& event_state = event_state_workspace.get();
            vec_ops->assign_mul(
                T(1) - turning.right_fraction,
                x_path_left,
                turning.right_fraction,
                x_path_right,
                event_state);
            on_event(fold_transition, event_state);
            ++event_count;
            log->info_f(
                "stability source-path turning event on curve %i at "
                "lambda = %.16le, dim(U): before = (%i,%i), after = "
                "(%i,%i)",
                curve_number,
                double(turning.parameter),
                left_guard_stability.unstable.real,
                left_guard_stability.unstable.complex_pairs,
                right_guard_stability.unstable.real,
                right_guard_stability.unstable.complex_pairs);
        }
        if(right_guard_dimension != upper_dimension)
        {
            event_count += stab->refine_transition_sequence_confirmed(
                x_path_right,
                right_guard_parameter,
                right_guard_stability,
                upper_state,
                upper_parameter,
                upper_stability,
                on_event);
        }
        if(event_count == 0 && lower_dimension != upper_dimension)
        {
            throw std::runtime_error(
                "stability source path subdivision lost the endpoint "
                "dimension change");
        }
        return event_count;
    }

    void resolve_uncertainty(
        int curve_number,
        uint64_t lower_source_point,
        uint64_t upper_source_point,
        uncertainty_stage_t stage)
    {
        if(!uncertainty_registry->resolve(
            static_cast<std::size_t>(curve_number),
            lower_source_point,
            upper_source_point,
            stage))
        {
            throw std::runtime_error(
                "failed to persist classification uncertainty "
                "resolution");
        }
    }

    void persist_pending_topology_uncertainties()
    {
        for(auto& pending : pending_topology_uncertainties)
        {
            if(!uncertainty_registry->record_failure(
                pending.curve_number,
                pending.lower_source_point,
                pending.lower_parameter,
                pending.upper_source_point,
                pending.upper_parameter,
                pending.failed_parameter,
                uncertainty_stage_t::source_path_topology,
                std::move(pending.observations),
                std::move(pending.diagnostic)))
            {
                log->warning(
                    "stability traversal: the committed topology split "
                    "could not be added to the classification uncertainty "
                    "registry");
            }
        }
        pending_topology_uncertainties.clear();
    }

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
    source_path_marcher_t* source_path_marcher = nullptr;
    bif_diag_t* bif_diag = nullptr;
    queue_pointer_t* queue_pointer = nullptr;
    queue_lambda_t* queue_lambda = nullptr;
    queue_dims_t* queue_dims = nullptr;
    queue_source_indices_t* queue_source_indices = nullptr;
    stability_diagram_t* stability_diagram = nullptr;
    uncertainty_registry_t* uncertainty_registry = nullptr;
    std::vector<pending_topology_uncertainty>
        pending_topology_uncertainties;

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
