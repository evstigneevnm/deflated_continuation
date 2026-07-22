#ifndef __DEFLATION_CONTINUATION_HPP__
#define __DEFLATION_CONTINUATION_HPP__

/**
*   The main class of the whole Deflation-Continuaiton Process (DCP). 
*
*   It uses nonlinear operator and other set options to configure the whole project.
*   After vector and file operations, the nonlinear operator, log and monitor are configured,
*   this class is initialized and configured to perform the whole DCP.
*   data serialization is done using boost archive
*/
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <filesystem>
#include <sstream>
#include <type_traits>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <containers/knots.hpp>
#include <containers/knot_registry.h>
#include <containers/branch_intersection.h>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <containers/bifurcation_diagram/diagram_archive.h>
#include <containers/bifurcation_diagram/symmetry_event_registry_sync.h>
#include <containers/symmetry_event_registry.h>

#include <continuation/continuation.hpp>
#include <continuation/continuation_analytical.hpp> // inherited from continuation to put a nontrivial analytical solution on the curve, if needed.

#include <deflation/solution_storage.h>
#include <deflation/deflation.hpp>

#include <main/deflation_continuation/solver_bundle.h>
#include <main/deflation_continuation/parameter_application.h>
#include <main/deflation_continuation/rejected_candidate_cache.h>
#include <main/deflation_continuation/exact_solution_registry.h>
#include <main/deflation_continuation/knot_relocation_controller.h>
#include <main/deflation_continuation/analytical_branch_executor.h>
#include <main/deflation_continuation/knot_executor.h>



namespace main_classes{

namespace detail
{

template<class SolutionStorage, class Vector>
auto stabilize_solution_if_available(SolutionStorage* storage, Vector& x) -> decltype(storage->stabilize_in_place(x), void())
{
    storage->stabilize_in_place(x);
}

inline void stabilize_solution_if_available(...)
{
}

template<class Continuation, class SolutionStorage, class Vector>
auto set_solution_postprocessor_if_available(Continuation* continuation, SolutionStorage* storage, Vector*) -> decltype(storage->stabilize_in_place(std::declval<Vector&>()), void())
{
    continuation->set_solution_postprocessor([storage](Vector& x)
    {
        storage->stabilize_in_place(x);
    });
}

inline void set_solution_postprocessor_if_available(...)
{
}

template<class Continuation, class Resolver>
auto set_knot_resolver_if_available(Continuation* continuation, Resolver&& resolver)
    -> decltype(continuation->set_knot_resolver(std::forward<Resolver>(resolver)), void())
{
    continuation->set_knot_resolver(std::forward<Resolver>(resolver));
}

inline void set_knot_resolver_if_available(...)
{
}

template<class Continuation, class Relocator>
auto set_knot_relocator_if_available(Continuation* continuation, Relocator&& relocator)
    -> decltype(continuation->set_knot_relocator(std::forward<Relocator>(relocator)), void())
{
    continuation->set_knot_relocator(std::forward<Relocator>(relocator));
}

inline void set_knot_relocator_if_available(...)
{
}

template<class SolutionStorage, class Vector, class Scalar>
auto nearest_stabilized_distance_if_available(SolutionStorage* storage, Vector& x, Scalar& distance)
    -> decltype(storage->nearest_stabilized_distance(x), bool())
{
    distance = static_cast<Scalar>(storage->nearest_stabilized_distance(x));
    return true;
}

inline bool nearest_stabilized_distance_if_available(...)
{
    return false;
}

} // namespace detail


template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator, class Parameters, class SolutionStorage = deflation::solution_storage<VectorOperations, Log>, template<class, class, class, class, class> class ContinuationSystemOperator = continuation::system_operator_continuation>
class deflation_continuation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;

    //general linear solver used in continuation and deflation
    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        LinearOperator,
        Preconditioner,
        VectorOperations,
        monitor_t,
        Log,
        LinearSolver
        > sherman_morrison_linear_system_solve_t;

    //general system operator for the newton's method
    //TODO: move to nonlinear_operators?
    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;

    //convergence strategy for the newton's method for F(x) = 0
    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;
    //newton's method for F(x) = 0
    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;
    
    typedef container::knots<T> knots_t;
    typedef container::knot_registry<T> knot_registry_t;

    typedef container::curve_helper_container<VectorOperations> container_helper_t;

    typedef SolutionStorage sol_storage_def_t;


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

    typedef continuation::continuation<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t,
        ContinuationSystemOperator
        > continuate_t;

    typedef continuation::continuation_analytical<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t,
        ContinuationSystemOperator
        > continuate_analytical_t;

    typedef deflation::deflation<
        VectorOperations,
        VectorFileOperations,
        Log,
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t
        > deflate_t;

    struct solver_bundle_types
    {
        using vector_operations_type = VectorOperations;
        using vector_file_operations_type = VectorFileOperations;
        using log_type = Log;
        using nonlinear_operations_type = NonlinearOperations;
        using linear_operator_type = LinearOperator;
        using preconditioner_type = Preconditioner;
        using linear_system_type = sherman_morrison_linear_system_solve_t;
        using convergence_type = convergence_newton_t;
        using system_operator_type = system_operator_t;
        using newton_type = newton_t;
        using knots_type = knots_t;
        using solution_storage_type = sol_storage_def_t;
        using continuation_type = continuate_t;
        using analytical_continuation_type = continuate_analytical_t;
        using diagram_type = bif_diag_t;
        using deflation_type = deflate_t;
    };
    typedef deflation_continuation_detail::solver_bundle<
        solver_bundle_types> solver_bundle_t;

    typedef container::intersection_status intersection_status_t;
    typedef container::symmetry_event_registry<T> symmetry_event_registry_t;
    typedef container::symmetry_event_registry_sync<
        VectorOperations,
        VectorFileOperations,
        Log> symmetry_event_registry_sync_t;
    typedef deflation_continuation_detail::exact_solution_registry<
        NonlinearOperations,
        T,
        T_vec> exact_solution_registry_t;
    typedef deflation_continuation_detail::analytical_branch_executor<
        VectorOperations,
        Log,
        exact_solution_registry_t,
        continuate_analytical_t,
        bif_diag_t,
        bif_diag_curve_t> analytical_branch_executor_t;
    typedef std::decay_t<decltype(
        std::declval<Parameters&>()
            .deflation_continuation
            .restart_policy
            .knot_relocation)> knot_relocation_settings_t;
    typedef deflation_continuation_detail::knot_relocation_controller<
        T,
        intersection_status_t,
        knot_relocation_settings_t,
        Log> knot_relocation_controller_t;
    typedef deflation_continuation_detail::knot_executor_callbacks<
        T,
        T_vec,
        intersection_status_t> knot_executor_callbacks_t;
    typedef deflation_continuation_detail::knot_executor<
        VectorOperations,
        knots_t,
        Log,
        intersection_status_t> knot_executor_t;

    class branch_distance_workspace
    {
    public:
        branch_distance_workspace(VectorOperations* vec_ops_, sol_storage_def_t* storage_):
            vec_ops(vec_ops_),
            storage(storage_)
        {
            vec_ops->init_vector(a_work);
            vec_ops->start_use_vector(a_work);
            vec_ops->init_vector(b_work);
            vec_ops->start_use_vector(b_work);
            vec_ops->init_vector(diff);
            vec_ops->start_use_vector(diff);
        }

        ~branch_distance_workspace()
        {
            vec_ops->stop_use_vector(diff);
            vec_ops->free_vector(diff);
            vec_ops->stop_use_vector(b_work);
            vec_ops->free_vector(b_work);
            vec_ops->stop_use_vector(a_work);
            vec_ops->free_vector(a_work);
        }

        T distance(const T_vec& a, const T_vec& b)
        {
            vec_ops->assign(a, a_work);
            vec_ops->assign(b, b_work);
            detail::stabilize_solution_if_available(storage, a_work);
            detail::stabilize_solution_if_available(storage, b_work);
            vec_ops->assign_mul(T(1), a_work, T(-1), b_work, diff);
            return vec_ops->norm_l2(diff);
        }

    private:
        VectorOperations* vec_ops;
        sol_storage_def_t* storage;
        T_vec a_work;
        T_vec b_work;
        T_vec diff;
    };

public:
    deflation_continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, Parameters* parameters_, sol_storage_def_t* sol_storage_external_ = nullptr):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    log_linsolver(log_linsolver_),
    parameters(parameters_)
    {
        
        //add '/' to the end of the project dir, if needed
        
        project_dir = parameters->path_to_project;
        skip_files = parameters->deflation_continuation.skip_files;

        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';


        solver_bundle = std::make_unique<solver_bundle_t>(
            vec_ops,
            file_ops,
            log,
            log_linsolver,
            nonlin_op,
            project_dir,
            skip_files,
            sol_storage_external_);
        lin_op = solver_bundle->linear_operator();
        precond = solver_bundle->preconditioner();
        SM = solver_bundle->linear_system();
        conv_newton = solver_bundle->convergence();
        system_operator = solver_bundle->system_operator();
        newton = solver_bundle->newton();
        knots = solver_bundle->knots();
        sol_storage_def = solver_bundle->solution_storage();
        continuate = solver_bundle->continuation();
        continuate_analytical = solver_bundle->analytical_continuation();
        bif_diag = solver_bundle->diagram();
        deflate = solver_bundle->deflation();
        exact_solutions = std::make_unique<exact_solution_registry_t>(nonlin_op);
        analytical_branches = std::make_unique<analytical_branch_executor_t>(
            vec_ops,
            log,
            exact_solutions.get(),
            continuate_analytical,
            bif_diag);
        symmetry_event_registry = std::make_unique<symmetry_event_registry_t>(
            std::filesystem::path(project_dir)/"symmetry_event_registry.json");
        symmetry_event_registry_sync =
            std::make_unique<symmetry_event_registry_sync_t>(
                vec_ops,
                file_ops,
                log);
        detail::set_solution_postprocessor_if_available(continuate, sol_storage_def, static_cast<T_vec*>(nullptr));
        detail::set_solution_postprocessor_if_available(continuate_analytical, sol_storage_def, static_cast<T_vec*>(nullptr));
    }
    ~deflation_continuation() = default;



//  called to set all parameters from the parameter structure
    void set_parameters()
    {
        set_linsolver();
        set_extended_linsolver();
        set_newton();
        set_newton_continuation();
        set_newton_deflation();
        set_steps();
        set_deflation_knots();
        set_branch_intersection_policy();
        set_self_intersection_policy();
        set_isotropy_transition_policy();
    }

    void set_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true*/
    {
        mon_orig = deflation_continuation_detail::configure_linear_solver<T>(
            SM->get_linsolver_handle_original(),
            parameters->nonlinear_operator.linear_solver);
    }
    void set_extended_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, bool is_small_alpha = false, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true
*/    
    {
        const auto& linear_solver_parameters =
            parameters->deflation_continuation.linear_solver_extended;
        mon = deflation_continuation_detail::configure_linear_solver<T>(
            SM->get_linsolver_handle(),
            linear_solver_parameters);
        SM->is_small_alpha(linear_solver_parameters.is_small_alpha);
    }
    void set_newton()
    /*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/
    {
        deflation_continuation_detail::configure_newton_convergence(
            conv_newton,
            parameters->nonlinear_operator.newton);
    }

    void set_newton_continuation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.8), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        deflation_continuation_detail::configure_continuation_newton(
            continuate,
            parameters->deflation_continuation.newton_extended_continuation,
            parameters->deflation_continuation.restart_policy.allow_knot_interpolation_failure);
    }

    void set_newton_deflation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        deflation_continuation_detail::configure_deflation_newton(
            deflate,
            parameters->deflation_continuation.newton_extended_deflation);
    }

    void set_steps()
/*unsigned int max_S_, T ds_0_, unsigned int deflation_attempts_ = 5, unsigned int attempts_0_ = 4, int initial_direciton_ = -1, T step_ds_m_ = 0.2, T step_ds_p_ = 0.01*/    
    {
        
        unsigned int max_S_ = parameters->deflation_continuation.continuation_steps;
        T ds_0_ = parameters->deflation_continuation.step_size;
        T ds_max_ = parameters->deflation_continuation.max_step_size;
        int initial_direciton_ = parameters->deflation_continuation.initial_direciton;
        unsigned int deflation_attempts_ = parameters->deflation_continuation.deflation_attempts;


        deflate->set_max_retries(deflation_attempts_);
        const auto retry_policy =
            deflation_continuation_detail::make_corrector_retry_policy<T>(
                parameters->deflation_continuation.corrector_retry_policy);
        continuate->set_steps(max_S_, ds_0_, ds_max_, initial_direciton_, retry_policy);
        continuate_analytical->set_steps(max_S_, ds_0_, ds_max_, initial_direciton_, retry_policy);

        const auto progress_policy =
            deflation_continuation_detail::make_progress_monitor_policy<T>(
                parameters->deflation_continuation.progress_monitor_policy);
        continuate->set_progress_monitor_policy(progress_policy);
        const auto chart_policy =
            deflation_continuation_detail::make_predictor_chart_policy<T>(
                parameters->deflation_continuation.predictor_chart_policy);
        continuate->set_predictor_chart_policy(chart_policy);
        continuate_analytical->set_predictor_chart_policy(chart_policy);
    }

    void set_deflation_knots()
/*std::vector<T> knots_*/    
    {
        knots->add_element(parameters->deflation_continuation.deflation_knots);
    }

    void set_branch_intersection_policy()
    {
        const auto policy =
            deflation_continuation_detail::make_branch_intersection_policy<T>(
                parameters->deflation_continuation.branch_intersection_policy);

        continuate->set_branch_intersection_refinement(
            policy.forward_refinement_step_factor,
            policy.maximum_forward_refinements,
            policy.minimum_forward_refinements_for_verification,
            policy.maximum_verified_forward_steps_ahead,
            policy.maximum_verified_forward_distance_step_ratio);
        continuate_analytical->set_branch_intersection_refinement(
            policy.forward_refinement_step_factor,
            policy.maximum_forward_refinements,
            policy.minimum_forward_refinements_for_verification,
            policy.maximum_verified_forward_steps_ahead,
            policy.maximum_verified_forward_distance_step_ratio);

        if(!policy.enabled)
        {
            continuate->set_branch_intersection_checker({});
            continuate_analytical->set_branch_intersection_checker({});
            return;
        }

        auto workspace = std::make_shared<branch_distance_workspace>(vec_ops, sol_storage_def);
        auto checker =
            [this, policy, workspace](
                const T& lambda_left,
                const T_vec& x_left,
                const T& lambda_right,
                const T_vec& x_right,
                T& hit_lambda,
                T_vec& hit_x,
                int& hit_curve_number,
                uint64_t& hit_segment_id,
                T& hit_forward_steps_ahead,
                T& hit_endpoint_distance_step_ratio,
                std::string& reason) -> container::branch_intersection_detection
            {
                container::branch_intersection_result<T> result;
                auto distance =
                    [workspace](const T_vec& a, const T_vec& b) -> T
                    {
                        return workspace->distance(a, b);
                    };
                const bool found = bif_diag->find_branch_intersection(
                    lambda_left,
                    x_left,
                    lambda_right,
                    x_right,
                    policy,
                    hit_x,
                    result,
                    distance);
                if(!found)
                {
                    return container::branch_intersection_detection::none;
                }

                hit_lambda = result.lambda;
                hit_curve_number = result.curve_number;
                hit_segment_id = result.segment_id;
                hit_forward_steps_ahead = result.forward_steps_ahead;
                hit_endpoint_distance_step_ratio =
                    result.endpoint_distance_step_ratio;
                std::ostringstream stream;
                if(result.forward_approach)
                {
                    stream << "predicted known branch encounter with curve "
                           << result.curve_number
                           << ", segment " << result.segment_id
                           << ", current state distance = " << result.state_distance
                           << ", distance/step = "
                           << result.endpoint_distance_step_ratio
                           << ", estimated steps ahead = "
                           << result.forward_steps_ahead;
                }
                else
                {
                    stream << "known branch intersection with curve " << result.curve_number
                           << ", segment " << result.segment_id
                           << ", state distance = " << result.state_distance
                           << ", tolerance = " << result.state_tolerance;
                }
                reason = stream.str();
                if(policy.verbose)
                {
                    if(result.forward_approach)
                    {
                        log->info_f(
                            "MAIN:deflation_continuation: forward branch encounter predicted at lambda = %le: curve = %i, segment = %llu, current state distance = %le, distance/step = %le, estimated steps ahead = %le.",
                            double(result.lambda),
                            result.curve_number,
                            static_cast<unsigned long long>(result.segment_id),
                            double(result.state_distance),
                            double(result.endpoint_distance_step_ratio),
                            double(result.forward_steps_ahead));
                    }
                    else
                    {
                        log->info_f(
                            "MAIN:deflation_continuation: branch intersection detected at lambda = %le: curve = %i, segment = %llu, state distance = %le, tolerance = %le.",
                            double(result.lambda),
                            result.curve_number,
                            static_cast<unsigned long long>(result.segment_id),
                            double(result.state_distance),
                            double(result.state_tolerance));
                    }
                }
                return result.forward_approach
                    ? container::branch_intersection_detection::forward_approach
                    : container::branch_intersection_detection::verified;
            };

        continuate->set_branch_intersection_checker(checker);
        continuate_analytical->set_branch_intersection_checker(checker);
    }

    void set_self_intersection_policy()
    {
        const auto policy =
            deflation_continuation_detail::make_self_intersection_policy<T>(
                parameters->deflation_continuation.self_intersection_policy);

        if(!policy.enabled)
        {
            continuate->set_self_intersection_checker({});
            continuate_analytical->set_self_intersection_checker({});
            return;
        }

        auto workspace = std::make_shared<branch_distance_workspace>(vec_ops, sol_storage_def);
        auto checker =
            [this, policy, workspace](
                bif_diag_curve_t* curve,
                const T& lambda_left,
                const T_vec& x_left,
                const T& lambda_right,
                const T_vec& x_right,
                T& hit_lambda,
                T_vec& hit_x,
                std::string& reason) -> bool
            {
                if(curve == nullptr)
                {
                    return false;
                }

                std::vector<T> step_norms0;
                std::vector<T> step_norms1;
                nonlin_op->norm_bifurcation_diagram(x_left, step_norms0);
                nonlin_op->norm_bifurcation_diagram(x_right, step_norms1);

                container::branch_intersection_result<T> result;
                auto distance =
                    [workspace](const T_vec& a, const T_vec& b) -> T
                    {
                        return workspace->distance(a, b);
                    };
                const bool found = curve->find_self_intersection(
                    lambda_left,
                    x_left,
                    lambda_right,
                    x_right,
                    step_norms0,
                    step_norms1,
                    policy,
                    hit_x,
                    result,
                    distance);
                if(!found)
                {
                    return false;
                }

                hit_lambda = result.lambda;
                std::ostringstream stream;
                stream << "curve-local self intersection with segment " << result.segment_id
                       << ", state distance = " << result.state_distance
                       << ", tolerance = " << result.state_tolerance;
                reason = stream.str();
                if(policy.verbose)
                {
                    log->info_f(
                        "MAIN:deflation_continuation: self intersection detected at lambda = %le: curve = %i, segment = %llu, state distance = %le, tolerance = %le.",
                        double(result.lambda),
                        result.curve_number,
                        static_cast<unsigned long long>(result.segment_id),
                        double(result.state_distance),
                        double(result.state_tolerance));
                }
                return true;
            };

        continuate->set_self_intersection_checker(checker);
        continuate_analytical->set_self_intersection_checker(checker);
    }

    void set_isotropy_transition_policy()
    {
        const auto policy =
            deflation_continuation_detail::make_isotropy_transition_policy<T>(
                parameters->deflation_continuation.isotropy_transition_policy);
        continuate->set_isotropy_transition_policy(policy);
    }

    void use_analytical_solution(bool analytical_solution_ = false)
    {
        analytical_solution = analytical_solution_;
    }

    void add_solution_curve(const T_vec& x0_, const T& lambda0_)
    {
        bif_diag_curve_t* bdf;
        T_vec x0_stabilized;
        vec_ops->init_vector(x0_stabilized);
        vec_ops->start_use_vector(x0_stabilized);
        vec_ops->assign(x0_, x0_stabilized);
        detail::stabilize_solution_if_available(sol_storage_def, x0_stabilized);
        bif_diag->init_new_curve();
        bif_diag->get_current_ref(bdf);
        continuate->continuate_curve(bdf, x0_stabilized, lambda0_);
        bif_diag->close_curve();
        if(!bif_diag->commit_current_curve_symmetry_events())
        {
            throw std::runtime_error(
                "MAIN:deflation_continuation: failed to commit symmetry events for an accepted seed curve");
        }
        synchronize_symmetry_event_registry();
        vec_ops->stop_use_vector(x0_stabilized);
        vec_ops->free_vector(x0_stabilized);
    }

    bool build_analytical_solution_curve_if_available(
        const std::string& file_name,
        const bool file_exists)
    {
        return analytical_branches->build_if_available(
            file_exists,
            analytical_solution_policy_enabled(),
            parameters->deflation_continuation.analytical_solution_branches,
            knots->get_value(),
            parameters->deflation_continuation.restart_policy.allow_failed_continuation_curve_save,
            [this](T_vec& value)
            {
                detail::stabilize_solution_if_available(sol_storage_def, value);
            },
            [this, &file_name]()
            {
                save_data(file_name);
            });
    }

    bool analytical_solution_policy_enabled() const
    {
        return analytical_solution || parameters->deflation_continuation.add_analytical_solution_to_diagram;
    }

    bool load_data(const std::string& file_name_ = {})
    {
        if(file_name_.empty())
        {
            return false;
        }

        const std::string path = project_dir + file_name_;
        log->info_f(
            "MAIN:deflation_continuation: reading data for the bifurcaiton diagram from %s ...",
            path.c_str());
        const auto result = container::load_diagram_archive(path, *bif_diag);
        if(result.succeeded())
        {
            bif_diag->reset_curve_output_directories();
            synchronize_symmetry_event_registry();
            log->info_f(
                "MAIN:deflation_continuation: read data for the bifurcaiton diagram from %s",
                path.c_str());
            return true;
        }

        if(result.status == container::diagram_archive_status::missing)
        {
            log->warning_f(
                "MAIN:deflation_continuation: failed to load saved data for the bifurcaiton diagram %s",
                path.c_str());
        }
        else
        {
            log->error_f(
                "MAIN:deflation_continuation: reading bifurcation diagram archive %s failed: %s",
                path.c_str(),
                result.message.c_str());
        }
        return false;
    }

    void save_data(const std::string& file_name_ = {})
    {
        if(file_name_.empty())
        {
            return;
        }

        synchronize_symmetry_event_registry();
        const std::string path = project_dir + file_name_;
        log->info_f(
            "MAIN:deflation_continuation: saving data for the bifurcaiton diagram in %s ...",
            path.c_str());
        const auto result = container::save_diagram_archive(path, *bif_diag);
        if(!result.succeeded())
        {
            log->error_f(
                "MAIN:deflation_continuation: writing bifurcation diagram archive %s failed: %s",
                path.c_str(),
                result.message.c_str());
            return;
        }
        log->info_f(
            "MAIN:deflation_continuation: saved data for the bifurcaiton diagram in %s",
            path.c_str());
    }

    void synchronize_symmetry_event_registry()
    {
        if(symmetry_event_registry == nullptr ||
           symmetry_event_registry_sync == nullptr ||
           bif_diag == nullptr)
        {
            return;
        }

        const auto records = bif_diag->symmetry_event_records();
        const auto& policy =
            parameters->deflation_continuation.isotropy_transition_policy;
        const auto result = symmetry_event_registry_sync->synchronize(
            *symmetry_event_registry,
            records,
            std::filesystem::path(project_dir),
            policy.registry_lambda_tolerance,
            policy.registry_state_tolerance,
            [this](T_vec& value)
            {
                detail::stabilize_solution_if_available(sol_storage_def, value);
            });

        if(result.attempted && !result.saved)
        {
            log->warning(
                "MAIN:deflation_continuation: failed to save symmetry_event_registry.json");
            return;
        }
        if(result.registry_loaded && result.previous_event_count != result.event_count)
        {
            log->warning_f(
                "MAIN:deflation_continuation: rebuilt symmetry event registry changed node count from %lu to %lu.",
                static_cast<unsigned long>(result.previous_event_count),
                static_cast<unsigned long>(result.event_count));
        }
    }

    void edit()
    {
        std::string file_name = parameters->bifurcaiton_diagram_file_name;
        bool file_exists = load_data( file_name );
        if(file_exists)
        {
            std::cout << "entering interactive edit mode" << std::endl;
            std::cout << "enter 'd' to pop_back() the curve or 'q' to quit." << std::endl;
            char c = 'c';
            while(c != 'q')
            {
                std::cout << "file " << file_name << " contains:" << std::endl;
                bif_diag->print_curves_status();
                c = std::cin.get();
                if(c=='d')
                {
                    bif_diag->pop_back_curve();
                }
            }
            c = std::cin.get();
            std::cout << "save file(y/n)>>>";
            c = std::cin.get();
            if(c == 'y')
                save_data(file_name);

        }
        else
        {
            log->warning_f("MAIN:deflation_continuation: file %s doesn't exist; called edit with no file provided!", file_name.c_str());
        }

    }

    void reset_known_solutions_at_lambda(const T& lambda)
    {
        (void)lambda;
        sol_storage_def->clear();
    }

    intersection_status_t rebuild_intersections_at_lambda(const T& lambda)
    {
        reset_known_solutions_at_lambda(lambda);
        return bif_diag->find_intersection(lambda, sol_storage_def);
    }

    void execute()
    {
        const std::string file_name =
            parameters->bifurcaiton_diagram_file_name;
        knot_relocation_controller_t knot_relocation(
            parameters->deflation_continuation.restart_policy.knot_relocation,
            parameters->deflation_continuation.deflation_knots,
            project_dir,
            log);
        knot_registry_t knot_registry(knot_relocation.registry_file_name());

        auto knot_resolver =
            [&knot_registry](
                const T& requested_parameter,
                T& effective_parameter) -> bool
            {
                typename knot_registry_t::entry entry;
                if(knot_registry.find(requested_parameter, entry))
                {
                    effective_parameter = entry.effective;
                    return true;
                }
                effective_parameter = requested_parameter;
                return false;
            };
        auto active_knot_relocator =
            [this, &knot_registry, &knot_relocation](
                const T& requested_parameter,
                const T& parameter_left,
                const T_vec& value_left,
                const T& parameter_right,
                const T_vec& value_right,
                T& effective_parameter,
                T_vec& effective_value) -> bool
            {
                return knot_relocation.relocate_active_intersection(
                    requested_parameter,
                    parameter_left,
                    value_left,
                    parameter_right,
                    value_right,
                    knot_registry,
                    effective_parameter,
                    effective_value,
                    [this](
                        const T& candidate,
                        const T& interpolation_left,
                        const T_vec& left,
                        const T& interpolation_right,
                        const T_vec& right,
                        T_vec& output) -> bool
                    {
                        const T weight =
                            (candidate - interpolation_left)/
                            (interpolation_right - interpolation_left);
                        vec_ops->assign_mul(
                            T(1) - weight,
                            left,
                            weight,
                            right,
                            output);
                        detail::stabilize_solution_if_available(
                            sol_storage_def,
                            output);
                        if(!newton->solve(nonlin_op, output, candidate))
                        {
                            return false;
                        }
                        detail::stabilize_solution_if_available(
                            sol_storage_def,
                            output);
                        return true;
                    });
            };
        detail::set_knot_resolver_if_available(continuate, knot_resolver);
        detail::set_knot_relocator_if_available(
            continuate,
            active_knot_relocator);
        detail::set_knot_resolver_if_available(
            continuate_analytical,
            knot_resolver);
        detail::set_knot_relocator_if_available(
            continuate_analytical,
            active_knot_relocator);

        bif_diag_curve_t* active_curve = nullptr;
        knot_executor_callbacks_t callbacks;
        callbacks.load_archive = [this, &file_name]()
        {
            return load_data(file_name);
        };
        callbacks.build_analytical_branches =
            [this, &file_name](const bool archive_exists)
            {
                build_analytical_solution_curve_if_available(
                    file_name,
                    archive_exists);
            };
        callbacks.restore_output_settings = [this]()
        {
            bif_diag->set_skip_output(skip_files);
        };
        callbacks.current_curve_count = [this]()
        {
            return bif_diag->current_curve();
        };
        callbacks.resolve_parameter = knot_resolver;
        callbacks.relocation_registry_file = [&knot_relocation]()
        {
            return knot_relocation.registry_file_name();
        };
        callbacks.rebuild_intersections = [this](const T& parameter)
        {
            return rebuild_intersections_at_lambda(parameter);
        };
        callbacks.relocate_intersection =
            [this, &knot_registry, &knot_relocation](
                const T& requested_parameter,
                const intersection_status_t& failed_status,
                T& effective_parameter,
                intersection_status_t& effective_status)
            {
                return knot_relocation.relocate_restart_intersection(
                    requested_parameter,
                    failed_status,
                    knot_registry,
                    effective_parameter,
                    effective_status,
                    [this](const T& candidate)
                    {
                        return rebuild_intersections_at_lambda(candidate);
                    });
            };
        callbacks.find_deflated_solution = [this](const T& parameter)
        {
            return deflate->find_solution(parameter);
        };
        callbacks.get_deflated_solution = [this](T_vec& value)
        {
            deflate->get_solution_ref(value);
        };
        callbacks.stabilize = [this](T_vec& value)
        {
            detail::stabilize_solution_if_available(sol_storage_def, value);
        };
        callbacks.nearest_known_distance =
            [this](T_vec& value, T& distance)
            {
                return detail::nearest_stabilized_distance_if_available(
                    sol_storage_def,
                    value,
                    distance);
            };
        callbacks.continue_candidate =
            [this, &active_curve](T_vec& value, const T& parameter)
            {
                bif_diag->init_new_curve();
                bif_diag->get_current_ref(active_curve);
                const bool success = continuate->continuate_curve(
                    active_curve,
                    value,
                    parameter);
                bif_diag->close_curve();
                return success;
            };
        callbacks.accept_candidate =
            [this, &file_name, &active_curve](const T& parameter)
            {
                if(!bif_diag->commit_current_curve_symmetry_events())
                {
                    throw std::runtime_error(
                        "MAIN:deflation_continuation: failed to commit symmetry events for an accepted curve");
                }
                save_data(file_name);
                active_curve->find_intersection(parameter, sol_storage_def);
            };
        callbacks.discard_candidate = [this]()
        {
            bif_diag->discard_current_curve();
        };
        callbacks.save_archive = [this, &file_name]()
        {
            save_data(file_name);
        };

        const auto& restart =
            parameters->deflation_continuation.restart_policy;
        deflation_continuation_detail::knot_execution_policy<T> policy;
        policy.relocation_enabled = knot_relocation.enabled();
        policy.allow_incomplete_restart_intersections =
            restart.allow_incomplete_restart_intersections;
        policy.allow_failed_continuation_curve_save =
            restart.allow_failed_continuation_curve_save;
        policy.max_failed_continuations_per_knot =
            restart.max_failed_continuations_per_knot;
        policy.failed_continuation_rejection_tolerance =
            restart.failed_continuation_rejection_tolerance;
        policy.check_duplicate_after_deflation =
            restart.check_duplicate_after_deflation;
        policy.duplicate_after_deflation_tolerance =
            restart.duplicate_after_deflation_tolerance;
        policy.duplicate_after_deflation_retries =
            restart.duplicate_after_deflation_retries;

        knot_executor_t executor(
            vec_ops,
            knots,
            log,
            policy,
            std::move(callbacks));
        executor.execute();
    }

private:
//  references to the external classes:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    Parameters* parameters;

//created locally:
    LinearOperator* lin_op;
    Preconditioner* precond;
    monitor_t* mon;
    monitor_t* mon_orig; 
    sherman_morrison_linear_system_solve_t* SM;
    convergence_newton_t* conv_newton;
    system_operator_t* system_operator;
    newton_t* newton;
    knots_t* knots;
    bif_diag_t* bif_diag = nullptr;
    continuate_t* continuate;
    continuate_analytical_t* continuate_analytical;
    deflate_t* deflate;
    sol_storage_def_t* sol_storage_def = nullptr;
    std::unique_ptr<solver_bundle_t> solver_bundle;
    std::unique_ptr<exact_solution_registry_t> exact_solutions;
    std::unique_ptr<analytical_branch_executor_t> analytical_branches;
    std::unique_ptr<symmetry_event_registry_t> symmetry_event_registry;
    std::unique_ptr<symmetry_event_registry_sync_t> symmetry_event_registry_sync;
    std::string project_dir;
    bool analytical_solution = false;
    unsigned int skip_files;
};


}

#endif // __DEFLATION_CONTINUATION_HPP__
