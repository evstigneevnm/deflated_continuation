#ifndef __MAIN_PARAMETERS_JSON_H__
#define __MAIN_PARAMETERS_JSON_H__

#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>
#include <main/parameters/parameter_types.h>

namespace main_classes
{
namespace parameters_json_detail
{

template<class T>
std::complex<T> parse_complex_value(
    const nlohmann::json& json)
{
    if(json.is_number())
        return {json.template get<T>(), T{}};
    if(json.is_array() && json.size() == 2)
    {
        return {
            json.at(0).template get<T>(),
            json.at(1).template get<T>()};
    }
    if(json.is_object())
    {
        return {
            json.at("real").template get<T>(),
            json.at("imaginary").template get<T>()};
    }
    throw std::invalid_argument(
        "complex value must be a number, [real, imaginary], "
        "or an object with real and imaginary fields");
}

template<class T, class Config>
void parse_matrix_free_stability_config(
    const nlohmann::json& json,
    Config& config)
{
    config = {};
    config.enabled = json.value("enabled", config.enabled);
    config.linearization_scale = json.value(
        "linearization_scale",
        config.linearization_scale);

    const auto transformation =
        json.value("transformation", nlohmann::json::object());
    config.transformation.type =
        stability::analysis::
            parse_matrix_free_spectral_transformation(
                transformation.value(
                    "type",
                    std::string(
                        stability::analysis::
                            matrix_free_spectral_transformation_name(
                                config.transformation.type))));
    config.transformation.step = transformation.value(
        "step",
        config.transformation.step);
    config.transformation.repetitions = transformation.value(
        "repetitions",
        config.transformation.repetitions);
    if(transformation.contains("shifts"))
    {
        config.transformation.shifts.clear();
        for(const auto& shift : transformation.at("shifts"))
        {
            config.transformation.shifts.push_back(
                parse_complex_value<T>(shift));
        }
    }

    const auto outer =
        json.value("outer", nlohmann::json::object());
    config.outer.desired_eigenvalues = outer.value(
        "desired_eigenvalues",
        config.outer.desired_eigenvalues);
    config.outer.krylov_dimension = outer.value(
        "krylov_dimension",
        config.outer.krylov_dimension);
    config.outer.restart_dimension = outer.value(
        "restart_dimension",
        config.outer.restart_dimension);
    config.outer.maximum_restarts = outer.value(
        "maximum_restarts",
        config.outer.maximum_restarts);
    config.outer.absolute_tolerance = outer.value(
        "absolute_tolerance",
        config.outer.absolute_tolerance);
    config.outer.relative_tolerance = outer.value(
        "relative_tolerance",
        config.outer.relative_tolerance);
    config.outer.preserve_conjugate_pairs = outer.value(
        "preserve_conjugate_pairs",
        config.outer.preserve_conjugate_pairs);
    config.outer.orthogonalization = outer.value(
        "orthogonalization",
        config.outer.orthogonalization);
    config.outer.reorthogonalization = outer.value(
        "reorthogonalization",
        config.outer.reorthogonalization);
    config.outer.dgks_eta = outer.value(
        "dgks_eta",
        config.outer.dgks_eta);
    config.outer.breakdown_absolute_tolerance = outer.value(
        "breakdown_absolute_tolerance",
        config.outer.breakdown_absolute_tolerance);
    config.outer.breakdown_relative_tolerance = outer.value(
        "breakdown_relative_tolerance",
        config.outer.breakdown_relative_tolerance);
    config.outer.maximum_orthogonalization_passes = outer.value(
        "maximum_orthogonalization_passes",
        config.outer.maximum_orthogonalization_passes);

    const auto recovery =
        json.value("recovery", nlohmann::json::object());
    config.recovery.relative_basis_tolerance = recovery.value(
        "relative_basis_tolerance",
        config.recovery.relative_basis_tolerance);
    config.recovery.orthogonalization_passes = recovery.value(
        "orthogonalization_passes",
        config.recovery.orthogonalization_passes);
    config.recovery.absolute_residual_tolerance = recovery.value(
        "absolute_residual_tolerance",
        config.recovery.absolute_residual_tolerance);
    config.recovery.relative_residual_tolerance = recovery.value(
        "relative_residual_tolerance",
        config.recovery.relative_residual_tolerance);
    config.recovery.minimum_converged_eigenpairs = recovery.value(
        "minimum_converged_eigenpairs",
        config.recovery.minimum_converged_eigenpairs);

    const auto inner =
        json.value("inner_solver", nlohmann::json::object());
    config.inner_solver.basis_size = inner.value(
        "basis_size",
        config.inner_solver.basis_size);
    config.inner_solver.batch_size = inner.value(
        "batch_size",
        config.inner_solver.batch_size);
    const std::string preconditioner_side = inner.value(
        "preconditioner_side",
        std::string(1, config.inner_solver.preconditioner_side));
    if(preconditioner_side.size() != 1)
    {
        throw std::invalid_argument(
            "matrix-free inner preconditioner_side must be one "
            "character");
    }
    config.inner_solver.preconditioner_side =
        preconditioner_side.front();
    config.inner_solver.orthogonalization = inner.value(
        "orthogonalization",
        config.inner_solver.orthogonalization);
    config.inner_solver.reorthogonalization = inner.value(
        "reorthogonalization",
        config.inner_solver.reorthogonalization);
    config.inner_solver.dgks_eta = inner.value(
        "dgks_eta",
        config.inner_solver.dgks_eta);
    config.inner_solver.breakdown_relative_tolerance = inner.value(
        "breakdown_relative_tolerance",
        config.inner_solver.breakdown_relative_tolerance);
    config.inner_solver.maximum_orthogonalization_passes = inner.value(
        "maximum_orthogonalization_passes",
        config.inner_solver.maximum_orthogonalization_passes);
    config.inner_solver.restart_on_false_ritz_convergence = inner.value(
        "restart_on_false_ritz_convergence",
        config.inner_solver.restart_on_false_ritz_convergence);
    config.inner_solver.basis_retry_sizes = inner.value(
        "basis_retry_sizes",
        config.inner_solver.basis_retry_sizes);
    config.inner_solver.relative_tolerance = inner.value(
        "relative_tolerance",
        config.inner_solver.relative_tolerance);
    config.inner_solver.absolute_tolerance = inner.value(
        "absolute_tolerance",
        config.inner_solver.absolute_tolerance);
    config.inner_solver.maximum_iterations = inner.value(
        "maximum_iterations",
        config.inner_solver.maximum_iterations);
    config.inner_solver.minimum_iterations = inner.value(
        "minimum_iterations",
        config.inner_solver.minimum_iterations);
    config.inner_solver.save_convergence_history = inner.value(
        "save_convergence_history",
        config.inner_solver.save_convergence_history);
    config.inner_solver.divide_norms_by_relative_base = inner.value(
        "divide_norms_by_relative_base",
        config.inner_solver.divide_norms_by_relative_base);
    config.inner_solver.output_minimum_residual = inner.value(
        "output_minimum_residual",
        config.inner_solver.output_minimum_residual);
    config.inner_solver.verbose = inner.value(
        "verbose",
        config.inner_solver.verbose);

    const auto retry =
        json.value("retry", nlohmann::json::object());
    config.retry.enabled = retry.value(
        "enabled",
        config.retry.enabled);
    config.retry.maximum_shift_retries = retry.value(
        "maximum_shift_retries",
        config.retry.maximum_shift_retries);
    config.retry.initial_shift_perturbation = retry.value(
        "initial_shift_perturbation",
        config.retry.initial_shift_perturbation);
    config.retry.perturbation_growth = retry.value(
        "perturbation_growth",
        config.retry.perturbation_growth);
    config.retry.preconditioner_pole_absolute_tolerance =
        retry.value(
            "preconditioner_pole_absolute_tolerance",
            config.retry.
                preconditioner_pole_absolute_tolerance);
    config.retry.preconditioner_pole_relative_tolerance =
        retry.value(
            "preconditioner_pole_relative_tolerance",
            config.retry.
                preconditioner_pole_relative_tolerance);

    const auto aggregation =
        json.value("aggregation", nlohmann::json::object());
    config.aggregation.absolute_tolerance = aggregation.value(
        "absolute_tolerance",
        config.aggregation.absolute_tolerance);
    config.aggregation.relative_tolerance = aggregation.value(
        "relative_tolerance",
        config.aggregation.relative_tolerance);
    config.aggregation.minimum_successful_scans = aggregation.value(
        "minimum_successful_scans",
        config.aggregation.minimum_successful_scans);
    config.aggregation.minimum_eigenpairs = aggregation.value(
        "minimum_eigenpairs",
        config.aggregation.minimum_eigenpairs);
    config.aggregation.require_all_scans = aggregation.value(
        "require_all_scans",
        config.aggregation.require_all_scans);
    config.aggregation.probe_count = aggregation.value(
        "probe_count",
        config.aggregation.probe_count);
    config.aggregation.minimum_successful_probes =
        aggregation.value(
            "minimum_successful_probes",
            config.aggregation.minimum_successful_probes);
    config.aggregation.require_all_probes = aggregation.value(
        "require_all_probes",
        config.aggregation.require_all_probes);
    config.aggregation.eigenvector_independence_tolerance =
        aggregation.value(
            "eigenvector_independence_tolerance",
            config.aggregation.
                eigenvector_independence_tolerance);
    config.aggregation.eigenvector_orthogonalization_passes =
        aggregation.value(
            "eigenvector_orthogonalization_passes",
            config.aggregation.
                eigenvector_orthogonalization_passes);

    const auto recycling =
        json.value("recycling", nlohmann::json::object());
    config.recycling.enabled = recycling.value(
        "enabled",
        config.recycling.enabled);
    config.recycling.maximum_vectors = recycling.value(
        "maximum_vectors",
        config.recycling.maximum_vectors);
    config.recycling.innovation_weight = recycling.value(
        "innovation_weight",
        config.recycling.innovation_weight);
    config.recycling.absolute_residual_tolerance = recycling.value(
        "absolute_residual_tolerance",
        config.recycling.absolute_residual_tolerance);
    config.recycling.relative_residual_tolerance = recycling.value(
        "relative_residual_tolerance",
        config.recycling.relative_residual_tolerance);

    const auto tracking = json.value(
        "invariant_subspace_tracking",
        nlohmann::json::object());
    config.invariant_subspace_tracking.enabled = tracking.value(
        "enabled",
        config.invariant_subspace_tracking.enabled);
    config.invariant_subspace_tracking.maximum_dimension = tracking.value(
        "maximum_dimension",
        config.invariant_subspace_tracking.maximum_dimension);
    config.invariant_subspace_tracking.maximum_seed_vectors = tracking.value(
        "maximum_seed_vectors",
        config.invariant_subspace_tracking.maximum_seed_vectors);
    config.invariant_subspace_tracking.
        coverage_recovery_maximum_seed_vectors = tracking.value(
            "coverage_recovery_maximum_seed_vectors",
            config.invariant_subspace_tracking.
                coverage_recovery_maximum_seed_vectors);
    config.invariant_subspace_tracking.orthogonalization_passes =
        tracking.value(
            "orthogonalization_passes",
            config.invariant_subspace_tracking.orthogonalization_passes);
    config.invariant_subspace_tracking.seed_innovation_weight =
        tracking.value(
            "seed_innovation_weight",
            config.invariant_subspace_tracking.seed_innovation_weight);
    config.invariant_subspace_tracking.dependence_tolerance = tracking.value(
        "dependence_tolerance",
        config.invariant_subspace_tracking.dependence_tolerance);
    config.invariant_subspace_tracking.minimum_retained_residual_ratio =
        tracking.value(
            "minimum_retained_residual_ratio",
            config.invariant_subspace_tracking.
                minimum_retained_residual_ratio);
    config.invariant_subspace_tracking.absolute_invariance_tolerance =
        tracking.value(
            "absolute_invariance_tolerance",
            config.invariant_subspace_tracking.
                absolute_invariance_tolerance);
    config.invariant_subspace_tracking.relative_invariance_tolerance =
        tracking.value(
            "relative_invariance_tolerance",
            config.invariant_subspace_tracking.
                relative_invariance_tolerance);
    config.invariant_subspace_tracking.real_eigenvalue_tolerance =
        tracking.value(
            "real_eigenvalue_tolerance",
            config.invariant_subspace_tracking.
                real_eigenvalue_tolerance);
    config.invariant_subspace_tracking.eigenvalue_group_tolerance =
        tracking.value(
            "eigenvalue_group_tolerance",
            config.invariant_subspace_tracking.
                eigenvalue_group_tolerance);
    config.invariant_subspace_tracking.principal_angle_rank_tolerance =
        tracking.value(
            "principal_angle_rank_tolerance",
            config.invariant_subspace_tracking.
                principal_angle_rank_tolerance);

    const auto small_system =
        json.value("small_system", nlohmann::json::object());
    config.small_system.enabled = small_system.value(
        "enabled",
        config.small_system.enabled);
    config.small_system.maximum_dimension = small_system.value(
        "maximum_dimension",
        config.small_system.maximum_dimension);
    config.small_system.prefer = small_system.value(
        "prefer",
        config.small_system.prefer);
    config.small_system.absolute_residual_tolerance =
        small_system.value(
            "absolute_residual_tolerance",
            config.small_system.absolute_residual_tolerance);
    config.small_system.relative_residual_tolerance =
        small_system.value(
            "relative_residual_tolerance",
            config.small_system.relative_residual_tolerance);
}

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
    const auto overrides =
        json.value("manual_overrides", nlohmann::json::array());
    for(const auto& item: overrides)
    {
        typename Params::manual_override_s parsed;
        parsed.requested =
            item.at("requested").template get<T>();
        parsed.effective =
            item.at("effective").template get<T>();
        parsed.reason = item.value(
            "reason",
            std::string("manual_non_singular_override"));
        params.manual_overrides.push_back(std::move(parsed));
    }
}

template<class Params>
void parse_seed_schedule(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.registry_file =
        json.value("registry_file", params.registry_file);
    params.save_registry =
        json.value("save_registry", params.save_registry);
}

template<class Params>
void parse_failed_candidate_registry(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.registry_file = json.value(
        "registry_file",
        params.registry_file);
    params.state_directory = json.value(
        "state_directory",
        params.state_directory);
    params.policy_generation = json.value(
        "policy_generation",
        params.policy_generation);
}

template<class Params>
void parse_recovery_registry(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.registry_file = json.value(
        "registry_file",
        params.registry_file);
    params.checkpoint_directory = json.value(
        "checkpoint_directory",
        params.checkpoint_directory);
    params.policy_generation = json.value(
        "policy_generation",
        params.policy_generation);
    params.process_before_deflation = json.value(
        "process_before_deflation",
        params.process_before_deflation);
}

template<class T, class Params>
void parse_topology_registry(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    params.enabled = json.value("enabled", params.enabled);
    params.registry_file = json.value(
        "registry_file",
        params.registry_file);
    params.endpoint_directory = json.value(
        "endpoint_directory",
        params.endpoint_directory);
    params.policy_generation = json.value(
        "policy_generation",
        params.policy_generation);
    params.absolute_parameter_tolerance = json.value(
        "absolute_parameter_tolerance",
        params.absolute_parameter_tolerance);
    params.relative_parameter_tolerance = json.value(
        "relative_parameter_tolerance",
        params.relative_parameter_tolerance);
    params.state_tolerance = json.value(
        "state_tolerance",
        params.state_tolerance);
    params.minimum_tangent_line_similarity = json.value(
        "minimum_tangent_line_similarity",
        params.minimum_tangent_line_similarity);
    params.record_transverse_junctions = json.value(
        "record_transverse_junctions",
        params.record_transverse_junctions);
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
    params.preserve_partial_curves = json.value(
        "preserve_partial_curves", params.preserve_partial_curves);
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
    parse_seed_schedule(
        json.value("seed_schedule", nlohmann::json::object()),
        params.seed_schedule);
    parse_failed_candidate_registry(
        json.value(
            "failed_candidate_registry",
            nlohmann::json::object()),
        params.failed_candidate_registry);
    parse_recovery_registry(
        json.value("recovery_registry", nlohmann::json::object()),
        params.recovery_registry);
    parse_topology_registry<T>(
        json.value("topology_registry", nlohmann::json::object()),
        params.topology_registry);
}

template<class T, class Params>
void parse_continuation_parameter_bounds(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    if(json.empty())
    {
        return;
    }
    params.enabled = json.value("enabled", true);
    params.minimum =
        json.at("minimum").template get<T>();
    params.maximum =
        json.at("maximum").template get<T>();
    params.resolve_with_knot_registry = json.value(
        "resolve_with_knot_registry",
        params.resolve_with_knot_registry);
    if(params.enabled && !(params.minimum < params.maximum))
    {
        throw std::invalid_argument(
            "continuation_parameter_bounds requires minimum < maximum");
    }
}

template<class Params>
void parse_boundary_refinement_policy(
    const nlohmann::json& json,
    Params& params)
{
    params.set_default();
    params.preserve_last_converged_point = json.value(
        "preserve_last_converged_point",
        params.preserve_last_converged_point);
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
    params.localize_analytical_targets = json.value(
        "localize_analytical_targets", params.localize_analytical_targets);
    params.analytical_target_parameter_tolerance = json.value(
        "analytical_target_parameter_tolerance",
        params.analytical_target_parameter_tolerance);
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

    parse_continuation_parameter_bounds<T>(
        json.value(
            "continuation_parameter_bounds",
            nlohmann::json::object()),
        params.continuation_parameter_bounds);
    parse_boundary_refinement_policy(
        json.value(
            "boundary_refinement_policy",
            nlohmann::json::object()),
        params.boundary_refinement_policy);
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
    params.set_default();
    params.linear_operator_stable_eigenvalues_left_halfplane =
        json.at("left_halfplane_stable_eigenvalues").template get<bool>();
    params.Krylov_subspace = json.at("Krylov_subspace_dimension").template get<unsigned int>();
    params.desired_spectrum = json.at("desired_spectrum").template get<unsigned int>();
    params.Cayley_transform_sigma_mu =
        json.at("Cayley_transform_sigma_mu").template get<std::vector<T>>();
    params.stability_boundary_tolerance = json.value(
        "stability_boundary_tolerance",
        params.stability_boundary_tolerance);
    params.real_eigenvalue_tolerance = json.value(
        "real_eigenvalue_tolerance",
        params.real_eigenvalue_tolerance);
    params.conjugate_pair_tolerance = json.value(
        "conjugate_pair_tolerance",
        params.conjugate_pair_tolerance);
    params.require_converged_eigenpairs = json.value(
        "require_converged_eigenpairs",
        params.require_converged_eigenpairs);
    params.require_nonempty_spectrum = json.value(
        "require_nonempty_spectrum",
        params.require_nonempty_spectrum);
    params.require_complete_scan_coverage = json.value(
        "require_complete_scan_coverage",
        params.require_complete_scan_coverage);
    params.spectrum_classification_retries = json.value(
        "spectrum_classification_retries",
        params.spectrum_classification_retries);
    params.transition_classification_confirmations = json.value(
        "transition_classification_confirmations",
        params.transition_classification_confirmations);
    params.confirm_regular_stability_points = json.value(
        "confirm_regular_stability_points",
        params.confirm_regular_stability_points);
    params.correct_stability_transitions_with_newton = json.value(
        "correct_stability_transitions_with_newton",
        params.correct_stability_transitions_with_newton);
    params.recover_failed_transition_classification_with_newton =
        json.value(
            "recover_failed_transition_classification_with_newton",
            params.
                recover_failed_transition_classification_with_newton);
    params.recover_failed_transition_newton_with_parameter_homotopy =
        json.value(
            "recover_failed_transition_newton_with_parameter_homotopy",
            params.
                recover_failed_transition_newton_with_parameter_homotopy);
    params.transition_newton_homotopy_maximum_subdivisions =
        json.value(
            "transition_newton_homotopy_maximum_subdivisions",
            params.
                transition_newton_homotopy_maximum_subdivisions);
    if(params.transition_newton_homotopy_maximum_subdivisions < 2)
    {
        throw std::invalid_argument(
            "transition Newton parameter homotopy requires at least "
            "two subdivisions");
    }
    params.transition_refinement_maximum_iterations = json.value(
        "transition_refinement_maximum_iterations",
        params.transition_refinement_maximum_iterations);
    params.transition_refinement_maximum_subdivisions = json.value(
        "transition_refinement_maximum_subdivisions",
        params.transition_refinement_maximum_subdivisions);
    params.transition_refinement_parameter_tolerance = json.value(
        "transition_refinement_parameter_tolerance",
        params.transition_refinement_parameter_tolerance);
    params.turning_point_guard_source_points = json.value(
        "turning_point_guard_source_points",
        params.turning_point_guard_source_points);
    params.allow_source_path_topology_splits = json.value(
        "allow_source_path_topology_splits",
        params.allow_source_path_topology_splits);
    params.symmetry_endpoint_guard_source_points = json.value(
        "symmetry_endpoint_guard_source_points",
        params.symmetry_endpoint_guard_source_points);
    parse_matrix_free_stability_config<T>(
        json.value(
            "matrix_free_eigensolver",
            nlohmann::json::object()),
        params.matrix_free_eigensolver);
    const auto uncertainty_registry = json.value(
        "classification_uncertainty_registry",
        nlohmann::json::object());
    params.classification_uncertainty_registry.enabled =
        uncertainty_registry.value(
            "enabled",
            params.classification_uncertainty_registry.enabled);
    params.classification_uncertainty_registry.file_name =
        uncertainty_registry.value(
            "file_name",
            params.classification_uncertainty_registry.file_name);
    params.classification_uncertainty_registry.
        maximum_diagnostic_length = uncertainty_registry.value(
            "maximum_diagnostic_length",
            params.classification_uncertainty_registry.
                maximum_diagnostic_length);
    params.classification_uncertainty_registry.retain_resolved =
        uncertainty_registry.value(
            "retain_resolved",
            params.classification_uncertainty_registry.
                retain_resolved);
    if(
        params.classification_uncertainty_registry.enabled &&
        (
            params.classification_uncertainty_registry.file_name.empty() ||
            params.classification_uncertainty_registry.
                maximum_diagnostic_length == 0))
    {
        throw std::invalid_argument(
            "invalid stability classification uncertainty registry "
            "configuration");
    }
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
