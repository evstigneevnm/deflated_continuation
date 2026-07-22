#ifndef __MAIN_DEFLATION_CONTINUATION_PARAMETER_APPLICATION_H__
#define __MAIN_DEFLATION_CONTINUATION_PARAMETER_APPLICATION_H__

#include <containers/branch_intersection.h>
#include <continuation/corrector_retry_policy.h>
#include <continuation/predictor_chart_validation.h>
#include <continuation/progress_monitor.h>
#include <symmetry/continuation/isotropy_transition.h>

namespace main_classes
{
namespace deflation_continuation_detail
{

namespace parameter_application_detail
{

template<class Solver>
auto set_use_preconditioned_residual(Solver* solver, int value)
    -> decltype(solver->set_use_precond_resid(value), void())
{
    solver->set_use_precond_resid(value);
}

inline void set_use_preconditioned_residual(...)
{
}

template<class Solver>
auto set_residual_recalculation_frequency(Solver* solver, int value)
    -> decltype(solver->set_resid_recalc_freq(value), void())
{
    solver->set_resid_recalc_freq(value);
}

inline void set_residual_recalculation_frequency(...)
{
}

template<class Solver>
auto set_basis_size(Solver* solver, int value)
    -> decltype(solver->set_basis_size(value), void())
{
    solver->set_basis_size(value);
}

inline void set_basis_size(...)
{
}

template<class Continuation>
auto set_allow_knot_interpolation_failure(Continuation* continuation, bool value)
    -> decltype(continuation->set_allow_knot_interpolation_failure(value), void())
{
    continuation->set_allow_knot_interpolation_failure(value);
}

inline void set_allow_knot_interpolation_failure(...)
{
}

} // namespace parameter_application_detail

template<class Scalar, class Solver, class Parameters>
auto configure_linear_solver(Solver* solver, const Parameters& parameters)
    -> decltype(&solver->monitor())
{
    auto* monitor = &solver->monitor();
    monitor->init(
        static_cast<Scalar>(parameters.lin_solver_tol),
        Scalar(0),
        parameters.lin_solver_max_it);
    monitor->set_save_convergence_history(parameters.save_convergence_history);
    monitor->set_divide_out_norms_by_rel_base(
        parameters.divide_out_norms_by_rel_base);
    monitor->set_verbose(parameters.verbose);
    monitor->out_min_resid_norm();

    if(parameters.use_precond_resid >= 0)
    {
        parameter_application_detail::set_use_preconditioned_residual(
            solver,
            parameters.use_precond_resid);
    }
    if(parameters.resid_recalc_freq >= 0)
    {
        parameter_application_detail::set_residual_recalculation_frequency(
            solver,
            parameters.resid_recalc_freq);
    }
    if(parameters.basis_size > 0)
    {
        parameter_application_detail::set_basis_size(
            solver,
            parameters.basis_size);
    }
    return monitor;
}

template<class Convergence, class Parameters>
void configure_newton_convergence(
    Convergence* convergence,
    const Parameters& parameters)
{
    convergence->set_convergence_constants(
        parameters.tolerance,
        parameters.newton_max_it,
        parameters.newton_wight,
        parameters.store_norms_history,
        parameters.verbose);
}

template<class Continuation, class Parameters>
void configure_continuation_newton(
    Continuation* continuation,
    const Parameters& parameters,
    const bool allow_knot_interpolation_failure)
{
    continuation->set_newton(
        parameters.tolerance,
        parameters.newton_max_it,
        parameters.relax_tolerance_factor,
        parameters.relax_tolerance_steps,
        parameters.newton_wight,
        parameters.store_norms_history,
        parameters.verbose,
        parameters.stagnation_max,
        parameters.maximum_norm_increase,
        parameters.newton_wight_threshold);
    parameter_application_detail::set_allow_knot_interpolation_failure(
        continuation,
        allow_knot_interpolation_failure);
}

template<class Deflation, class Parameters>
void configure_deflation_newton(
    Deflation* deflation,
    const Parameters& parameters)
{
    deflation->set_newton(
        parameters.tolerance,
        parameters.newton_max_it,
        parameters.newton_wight,
        parameters.store_norms_history,
        parameters.verbose);
}

template<class Scalar, class Parameters>
continuation::corrector_retry_policy<Scalar> make_corrector_retry_policy(
    const Parameters& parameters)
{
    continuation::corrector_retry_policy<Scalar> policy;
    policy.maximum_retries = parameters.maximum_retries;
    policy.failure_reduction_factor = parameters.failure_reduction_factor;
    policy.minimum_step_size = parameters.minimum_step_size;
    policy.minimum_step_ratio = parameters.minimum_step_ratio;
    policy.successes_before_growth = parameters.successes_before_growth;
    policy.success_growth_factor = parameters.success_growth_factor;
    policy.validate();
    return policy;
}

template<class Scalar, class Parameters>
continuation::progress_monitor_policy<Scalar> make_progress_monitor_policy(
    const Parameters& parameters)
{
    continuation::progress_monitor_policy<Scalar> policy;
    policy.enabled = parameters.enabled;
    policy.window_size = parameters.window_size;
    policy.minimum_window_progress_ratio = parameters.minimum_window_progress_ratio;
    policy.validate();
    return policy;
}

template<class Scalar, class Parameters>
continuation::predictor_chart_policy<Scalar> make_predictor_chart_policy(
    const Parameters& parameters)
{
    continuation::predictor_chart_policy<Scalar> policy;
    policy.enabled = parameters.enabled;
    policy.enforce = parameters.enforce;
    policy.weak_progress_warning_ratio = parameters.weak_progress_warning_ratio;
    policy.minimum_progress_ratio = parameters.minimum_progress_ratio;
    policy.progress_jump_warning_ratio = parameters.progress_jump_warning_ratio;
    policy.maximum_progress_ratio = parameters.maximum_progress_ratio;
    policy.displacement_warning_ratio = parameters.displacement_warning_ratio;
    policy.maximum_displacement_ratio = parameters.maximum_displacement_ratio;
    policy.maximum_retries = parameters.maximum_retries;
    policy.step_reduction_factor = parameters.step_reduction_factor;
    continuation::validate_predictor_chart_policy(policy);
    return policy;
}

template<class Scalar, class Parameters>
container::branch_intersection_policy<Scalar> make_branch_intersection_policy(
    const Parameters& parameters)
{
    container::branch_intersection_policy<Scalar> policy;
    policy.enabled = parameters.enabled;
    policy.signature_norm_index = parameters.signature_norm_index;
    policy.signature_tolerance = parameters.signature_tolerance;
    policy.state_tolerance = parameters.state_tolerance;
    policy.minimum_step_fraction_from_start =
        parameters.minimum_step_fraction_from_start;
    policy.detect_forward_approach = parameters.detect_forward_approach;
    policy.maximum_forward_lookahead_steps =
        parameters.maximum_forward_lookahead_steps;
    policy.maximum_forward_distance_step_ratio =
        parameters.maximum_forward_distance_step_ratio;
    policy.minimum_forward_distance_reduction_ratio =
        parameters.minimum_forward_distance_reduction_ratio;
    policy.forward_distance_extrapolation_power =
        parameters.forward_distance_extrapolation_power;
    policy.forward_refinement_step_factor =
        parameters.forward_refinement_step_factor;
    policy.maximum_forward_refinements = parameters.maximum_forward_refinements;
    policy.minimum_forward_refinements_for_verification =
        parameters.minimum_forward_refinements_for_verification;
    policy.maximum_verified_forward_steps_ahead =
        parameters.maximum_verified_forward_steps_ahead;
    policy.maximum_verified_forward_distance_step_ratio =
        parameters.maximum_verified_forward_distance_step_ratio;
    policy.verbose = parameters.verbose;
    container::validate_branch_intersection_policy(policy);
    return policy;
}

template<class Scalar, class Parameters>
container::self_intersection_policy<Scalar> make_self_intersection_policy(
    const Parameters& parameters)
{
    container::self_intersection_policy<Scalar> policy;
    policy.enabled = parameters.enabled;
    policy.signature_norm_index = parameters.signature_norm_index;
    policy.signature_tolerance = parameters.signature_tolerance;
    policy.state_tolerance = parameters.state_tolerance;
    policy.minimum_step_fraction_from_start =
        parameters.minimum_step_fraction_from_start;
    policy.minimum_index_gap = parameters.minimum_index_gap;
    policy.verbose = parameters.verbose;
    return policy;
}

template<class Scalar, class Parameters>
symmetry::continuation::isotropy_transition_policy<Scalar>
make_isotropy_transition_policy(const Parameters& parameters)
{
    symmetry::continuation::isotropy_transition_policy<Scalar> policy;
    policy.enabled = parameters.enabled;
    policy.relative_mode_tolerance = parameters.relative_mode_tolerance;
    policy.maximum_order = parameters.maximum_order;
    policy.maximum_refinements = parameters.maximum_refinements;
    policy.refinement_step_factor = parameters.refinement_step_factor;
    policy.registry_lambda_tolerance = parameters.registry_lambda_tolerance;
    policy.registry_state_tolerance = parameters.registry_state_tolerance;
    policy.verbose = parameters.verbose;
    policy.validate();
    return policy;
}

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_PARAMETER_APPLICATION_H__
