#ifndef __MAIN_PARAMETERS_JSON_H__
#define __MAIN_PARAMETERS_JSON_H__

#include <vector>

#include <contrib/json/nlohmann/json.hpp>
#include <main/parameters/parameter_types.h>

namespace main_classes
{
namespace parameters_json_detail
{

template<class T, class Params>
void parse_extended_linear_solver(const nlohmann::json& json, Params& params)
{
    params.lin_solver_max_it = json.at("maximum_iterations").template get<unsigned int>();
    params.use_precond_resid = json.at("use_preconditioned_residual").template get<unsigned int>();
    params.resid_recalc_freq = json.at("residual_recalculate_frequency").template get<unsigned int>();
    params.basis_size = json.at("basis_size").template get<unsigned int>();
    params.lin_solver_tol = json.at("tolerance").template get<T>();
    params.is_small_alpha = json.at("use_small_alpha_approximation").template get<bool>();
    params.save_convergence_history = json.at("save_convergence_history").template get<bool>();
    params.divide_out_norms_by_rel_base = json.at("divide_norms_by_relative_base").template get<bool>();
    params.verbose = json.value("verbose", true);
}

template<class T, class Params>
void parse_linear_solver(const nlohmann::json& json, Params& params)
{
    params.lin_solver_max_it = json.at("maximum_iterations").template get<unsigned int>();
    params.use_precond_resid = json.at("use_preconditioned_residual").template get<unsigned int>();
    params.resid_recalc_freq = json.at("residual_recalculate_frequency").template get<unsigned int>();
    params.basis_size = json.at("basis_size").template get<unsigned int>();
    params.lin_solver_tol = json.at("tolerance").template get<T>();
    params.save_convergence_history = json.at("save_convergence_history").template get<bool>();
    params.divide_out_norms_by_rel_base = json.at("divide_norms_by_relative_base").template get<bool>();
    params.verbose = json.value("verbose", true);
}

template<class T, class Params>
void parse_newton(const nlohmann::json& json, Params& params)
{
    params.newton_max_it = json.at("maximum_iterations").template get<unsigned int>();
    params.newton_wight = json.at("update_wight_maximum").template get<T>();
    params.store_norms_history = json.at("save_norms_history").template get<bool>();
    params.verbose = json.at("verbose").template get<bool>();
    params.tolerance = json.at("tolerance").template get<T>();
}

template<class T, class Params>
void parse_continuation_newton(const nlohmann::json& json, Params& params)
{
    parse_newton<T>(json, params);
    params.relax_tolerance_factor = json.at("relax_tolerance_factor").template get<T>();
    params.relax_tolerance_steps = json.at("relax_tolerance_steps").template get<int>();
    params.stagnation_max = json.at("stagnation_max").template get<unsigned int>();
    params.maximum_norm_increase = json.at("maximum_norm_increase").template get<T>();
    params.newton_wight_threshold = json.at("newton_wight_threshold").template get<T>();
}

template<class T, class Params>
void parse_knot_relocation(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.registry_file = json.value("registry_file", params.registry_file);
    params.min_shift_abs = json.value("min_shift_abs", params.min_shift_abs);
    params.max_shift_abs = json.value("max_shift_abs", params.max_shift_abs);
    params.candidate_count = json.value("candidate_count", params.candidate_count);
    params.prefer_positive_shift = json.value("prefer_positive_shift", params.prefer_positive_shift);
    params.require_all_intersections = json.value("require_all_intersections", params.require_all_intersections);
    params.save_registry = json.value("save_registry", params.save_registry);
}

template<class T, class Params>
void parse_restart_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.allow_incomplete_restart_intersections = json.value(
        "allow_incomplete_restart_intersections", params.allow_incomplete_restart_intersections);
    params.allow_knot_interpolation_failure = json.value(
        "allow_knot_interpolation_failure", params.allow_knot_interpolation_failure);
    params.allow_failed_continuation_curve_save = json.value(
        "allow_failed_continuation_curve_save", params.allow_failed_continuation_curve_save);
    params.check_duplicate_after_deflation = json.value(
        "check_duplicate_after_deflation", params.check_duplicate_after_deflation);
    params.duplicate_after_deflation_retries = json.value(
        "duplicate_after_deflation_retries", params.duplicate_after_deflation_retries);
    params.duplicate_after_deflation_tolerance = json.value(
        "duplicate_after_deflation_tolerance", params.duplicate_after_deflation_tolerance);
    params.max_failed_continuations_per_knot = json.value(
        "max_failed_continuations_per_knot", params.max_failed_continuations_per_knot);
    params.failed_continuation_rejection_tolerance = json.value(
        "failed_continuation_rejection_tolerance", params.failed_continuation_rejection_tolerance);
    parse_knot_relocation<T>(json.value("knot_relocation", nlohmann::json::object()), params.knot_relocation);
}

template<class T, class Params>
void parse_branch_intersection_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.signature_norm_index = json.value("signature_norm_index", params.signature_norm_index);
    params.signature_tolerance = json.value("signature_tolerance", params.signature_tolerance);
    params.state_tolerance = json.value("state_tolerance", params.state_tolerance);
    params.minimum_step_fraction_from_start = json.value(
        "minimum_step_fraction_from_start", params.minimum_step_fraction_from_start);
    params.detect_forward_approach = json.value("detect_forward_approach", params.detect_forward_approach);
    params.maximum_forward_lookahead_steps = json.value(
        "maximum_forward_lookahead_steps", params.maximum_forward_lookahead_steps);
    params.maximum_forward_distance_step_ratio = json.value(
        "maximum_forward_distance_step_ratio", params.maximum_forward_distance_step_ratio);
    params.minimum_forward_distance_reduction_ratio = json.value(
        "minimum_forward_distance_reduction_ratio", params.minimum_forward_distance_reduction_ratio);
    params.forward_distance_extrapolation_power = json.value(
        "forward_distance_extrapolation_power", params.forward_distance_extrapolation_power);
    params.forward_refinement_step_factor = json.value(
        "forward_refinement_step_factor", params.forward_refinement_step_factor);
    params.maximum_forward_refinements = json.value(
        "maximum_forward_refinements", params.maximum_forward_refinements);
    params.minimum_forward_refinements_for_verification = json.value(
        "minimum_forward_refinements_for_verification", params.minimum_forward_refinements_for_verification);
    params.maximum_verified_forward_steps_ahead = json.value(
        "maximum_verified_forward_steps_ahead", params.maximum_verified_forward_steps_ahead);
    params.maximum_verified_forward_distance_step_ratio = json.value(
        "maximum_verified_forward_distance_step_ratio", params.maximum_verified_forward_distance_step_ratio);
    params.verbose = json.value("verbose", params.verbose);
}

template<class T, class Params>
void parse_self_intersection_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.signature_norm_index = json.value("signature_norm_index", params.signature_norm_index);
    params.signature_tolerance = json.value("signature_tolerance", params.signature_tolerance);
    params.state_tolerance = json.value("state_tolerance", params.state_tolerance);
    params.minimum_step_fraction_from_start = json.value(
        "minimum_step_fraction_from_start", params.minimum_step_fraction_from_start);
    params.minimum_index_gap = json.value("minimum_index_gap", params.minimum_index_gap);
    params.verbose = json.value("verbose", params.verbose);
}

template<class T, class Params>
void parse_isotropy_transition_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.relative_mode_tolerance = json.value("relative_mode_tolerance", params.relative_mode_tolerance);
    params.maximum_order = json.value("maximum_order", params.maximum_order);
    params.maximum_refinements = json.value("maximum_refinements", params.maximum_refinements);
    params.refinement_step_factor = json.value("refinement_step_factor", params.refinement_step_factor);
    params.registry_lambda_tolerance = json.value(
        "registry_lambda_tolerance", params.registry_lambda_tolerance);
    params.registry_state_tolerance = json.value(
        "registry_state_tolerance", params.registry_state_tolerance);
    params.verbose = json.value("verbose", params.verbose);
}

template<class T, class Params>
void parse_predictor_chart_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.enforce = json.value("enforce", params.enforce);
    params.weak_progress_warning_ratio = json.value(
        "weak_progress_warning_ratio", params.weak_progress_warning_ratio);
    params.minimum_progress_ratio = json.value("minimum_progress_ratio", params.minimum_progress_ratio);
    params.progress_jump_warning_ratio = json.value(
        "progress_jump_warning_ratio", params.progress_jump_warning_ratio);
    params.maximum_progress_ratio = json.value("maximum_progress_ratio", params.maximum_progress_ratio);
    params.displacement_warning_ratio = json.value(
        "displacement_warning_ratio", params.displacement_warning_ratio);
    params.maximum_displacement_ratio = json.value(
        "maximum_displacement_ratio", params.maximum_displacement_ratio);
    params.maximum_retries = json.value("maximum_retries", params.maximum_retries);
    params.step_reduction_factor = json.value("step_reduction_factor", params.step_reduction_factor);
}

template<class T, class Params>
void parse_corrector_retry_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.maximum_retries = json.value("maximum_retries", params.maximum_retries);
    params.failure_reduction_factor = json.value(
        "failure_reduction_factor", params.failure_reduction_factor);
    params.minimum_step_size = json.value("minimum_step_size", params.minimum_step_size);
    params.minimum_step_ratio = json.value("minimum_step_ratio", params.minimum_step_ratio);
    params.successes_before_growth = json.value(
        "successes_before_growth", params.successes_before_growth);
    params.success_growth_factor = json.value(
        "success_growth_factor", params.success_growth_factor);
}

template<class T, class Params>
void parse_progress_monitor_policy(const nlohmann::json& json, Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.window_size = json.value("window_size", params.window_size);
    params.minimum_window_progress_ratio = json.value(
        "minimum_window_progress_ratio", params.minimum_window_progress_ratio);
}

template<class T>
void parse_deflation_continuation(
    const nlohmann::json& json,
    typename parameters<T>::deflation_continuation_s& params)
{
    params.continuation_steps = json.at("maximum_continuation_steps").template get<unsigned int>();
    params.step_size = json.at("step_size").template get<T>();
    params.max_step_size = json.at("max_step_size").template get<T>();
    params.deflation_attempts = json.at("deflation_attempts").template get<unsigned int>();
    params.continuation_fail_attempts = json.at("continuation_fail_attempts").template get<unsigned int>();
    params.initial_direciton = json.at("initial_direciton").template get<int>();
    params.step_ds_m = json.at("minimum_step_multiplier").template get<T>();
    params.step_ds_p = json.at("maximum_step_multiplier").template get<T>();
    params.skip_files = json.at("skip_file_output").template get<unsigned int>();
    params.deflation_knots = json.at("deflation_knots").template get<std::vector<T>>();
    params.add_analytical_solution_to_diagram = json.value("add_analytical_solution_to_diagram", false);
    params.analytical_solution_branches = json.value(
        "analytical_solution_branches", std::vector<unsigned int>());

    parse_restart_policy<T>(json.value("restart_policy", nlohmann::json::object()), params.restart_policy);
    parse_branch_intersection_policy<T>(
        json.value("branch_intersection_policy", nlohmann::json::object()),
        params.branch_intersection_policy);
    parse_self_intersection_policy<T>(
        json.value("self_intersection_policy", nlohmann::json::object()),
        params.self_intersection_policy);
    parse_isotropy_transition_policy<T>(
        json.value("isotropy_transition_policy", nlohmann::json::object()),
        params.isotropy_transition_policy);
    parse_predictor_chart_policy<T>(
        json.value("predictor_chart_policy", nlohmann::json::object()),
        params.predictor_chart_policy);
    parse_corrector_retry_policy<T>(
        json.value("corrector_retry_policy", nlohmann::json::object()),
        params.corrector_retry_policy);
    parse_progress_monitor_policy<T>(
        json.value("progress_monitor_policy", nlohmann::json::object()),
        params.progress_monitor_policy);

    parse_extended_linear_solver<T>(json.at("linear_solver_extended"), params.linear_solver_extended);
    parse_continuation_newton<T>(json.at("newton_continuation"), params.newton_extended_continuation);
    parse_newton<T>(json.at("newton_deflation"), params.newton_extended_deflation);

    if(!json.contains("corrector_retry_policy"))
    {
        params.corrector_retry_policy.maximum_retries = params.continuation_fail_attempts;
        params.corrector_retry_policy.failure_reduction_factor = T(1)-params.step_ds_m;
        params.corrector_retry_policy.successes_before_growth = 6;
        params.corrector_retry_policy.success_growth_factor = T(1.25);
    }
}

template<class T>
void parse_stability_continuation(
    const nlohmann::json& json,
    typename parameters<T>::stability_continuation_s& params)
{
    params.linear_operator_stable_eigenvalues_left_halfplane =
        json.at("left_halfplane_stable_eigenvalues").template get<bool>();
    params.Krylov_subspace = json.at("Krylov_subspace_dimension").template get<unsigned int>();
    params.desired_spectrum = json.at("desired_spectrum").template get<unsigned int>();
    params.Cayley_transform_sigma_mu =
        json.at("Cayley_transform_sigma_mu").template get<std::vector<T>>();
    parse_linear_solver<T>(json.at("linear_solver"), params.linear_solver);
    parse_newton<T>(json.at("newton"), params.newton);
}

template<class T>
void parse_nonlinear_operator(
    const nlohmann::json& json,
    typename parameters<T>::nonlinear_operator_s& params)
{
    params.N_size = json.at("discrete_problem_dimensions").template get<std::vector<std::size_t>>();
    params.problem_real_parameters_vector =
        json.at("problem_real_parameters_vector").template get<std::vector<T>>();
    params.problem_int_parameters_vector =
        json.at("problem_int_parameters_vector").template get<std::vector<int>>();
    parse_linear_solver<T>(json.at("linear_solver"), params.linear_solver);
    parse_newton<T>(json.at("newton"), params.newton);
}

template<class Params>
void parse_plot_solutions(const nlohmann::json& json, Params& params)
{
    params.plot_solution_frequency = json.at("plot_solution_frequency").template get<int>();
}

} // namespace parameters_json_detail

template<class T>
void from_json(const nlohmann::json& json, parameters<T>& params)
{
    params.nvidia_pci_id = json.at("gpu_pci_id").template get<int>();
    params.use_high_precision_reduction = json.at("use_high_precision_reduction").template get<bool>();
    params.path_to_project = json.at("path_to_project").template get<std::string>();
    params.bifurcaiton_diagram_file_name =
        json.at("bifurcaiton_diagram_file_name").template get<std::string>();
    params.stability_diagram_file_name =
        json.at("stability_diagram_file_name").template get<std::string>();

    parameters_json_detail::parse_deflation_continuation<T>(
        json.at("deflation_continuation"), params.deflation_continuation);
    parameters_json_detail::parse_stability_continuation<T>(
        json.at("stability_continuation"), params.stability_continuation);
    parameters_json_detail::parse_nonlinear_operator<T>(
        json.at("nonlinear_operator"), params.nonlinear_operator);
    parameters_json_detail::parse_plot_solutions(
        json.at("plot_solutions"), params.plot_solutions);
}

} // namespace main_classes

#endif // __MAIN_PARAMETERS_JSON_H__
