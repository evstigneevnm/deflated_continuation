#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <main/parameters.hpp>

namespace
{

template<class T>
bool close_value(const T& left, const T& right)
{
    const T scale = std::max<T>(T(1), std::max<T>(std::abs(left), std::abs(right)));
    return std::abs(left - right) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
}

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

template<class T>
void verify_full_configuration(const std::string& file_name)
{
    const auto parameters = main_classes::read_parameters_json<T>(file_name);
    require(parameters.nvidia_pci_id == -1, "gpu_pci_id");
    require(parameters.use_high_precision_reduction, "use_high_precision_reduction");
    require(parameters.path_to_project == "./build/KS1D_test_full/", "path_to_project");
    require(parameters.bifurcaiton_diagram_file_name == "bifurcation_diagram.dat", "diagram file");

    const auto& continuation = parameters.deflation_continuation;
    require(continuation.continuation_steps == 5000, "maximum_continuation_steps");
    require(close_value(continuation.step_size, T(0.02)), "step_size");
    require(close_value(continuation.max_step_size, T(0.25)), "max_step_size");
    require(continuation.initial_direciton == -1, "initial_direciton");
    require(continuation.deflation_knots.size() == 54, "deflation_knots");
    require(continuation.add_analytical_solution_to_diagram, "analytical branch flag");
    require(continuation.analytical_solution_branches.empty(), "analytical branch selection");

    require(continuation.restart_policy.knot_relocation.enabled, "knot relocation");
    require(continuation.restart_policy.knot_relocation.candidate_count == 12, "relocation candidates");
    const auto& manual_overrides =
        continuation.restart_policy.knot_relocation.manual_overrides;
    require(manual_overrides.size() == 2, "manual knot overrides");
    const auto require_override =
        [&manual_overrides](T requested, T effective, const std::string& message)
        {
            const auto override_it = std::find_if(
                manual_overrides.begin(),
                manual_overrides.end(),
                [requested](const auto& knot_override)
                {
                    return close_value(knot_override.requested, requested);
                });
            require(override_it != manual_overrides.end(), message + " requested knot");
            require(close_value(override_it->effective, effective), message + " effective knot");
        };
    require_override(T(4), T(4.01), "primary bifurcation override");
    require_override(T(100), T(101), "terminal pole override");
    require(
        continuation.restart_policy.seed_schedule.enabled,
        "deterministic seed schedule");
    require(
        continuation.continuation_parameter_bounds.enabled,
        "explicit continuation bounds");
    require(
        close_value(
            continuation.continuation_parameter_bounds.minimum,
            T(1)),
        "continuation minimum");
    require(
        close_value(
            continuation.continuation_parameter_bounds.maximum,
            T(110)),
        "continuation requested maximum");
    require(
        continuation.continuation_parameter_bounds
            .resolve_with_knot_registry,
        "continuation bound registry resolution");
    require(
        continuation.boundary_refinement_policy
            .preserve_last_converged_point,
        "transactional boundary refinement");
    require(continuation.branch_intersection_policy.detect_forward_approach, "forward branch detection");
    require(continuation.branch_intersection_policy.forward_distance_extrapolation_power == 2,
            "forward extrapolation power");
    require(continuation.self_intersection_policy.enabled, "self intersection");
    require(continuation.isotropy_transition_policy.maximum_order == 16, "isotropy order");
    require(continuation.predictor_chart_policy.enforce, "predictor chart enforcement");
    require(continuation.corrector_retry_policy.maximum_retries == 8, "corrector retries");
    require(continuation.progress_monitor_policy.enabled, "progress monitor");
    require(!continuation.linear_solver_extended.verbose, "extended solver verbosity");
    require(close_value(continuation.newton_extended_deflation.newton_wight, T(0.5)),
            "deflation Newton weight");

    require(parameters.nonlinear_operator.N_size.size() == 1, "nonlinear dimensions size");
    const std::size_t nonlinear_dimension =
        parameters.nonlinear_operator.N_size.front();
    require(
        nonlinear_dimension >= 32 &&
            (nonlinear_dimension & (nonlinear_dimension - 1)) == 0,
        "nonlinear dimension");
    require(parameters.nonlinear_operator.problem_real_parameters_vector.size() == 2,
            "problem real parameters");
    require(!parameters.stability_continuation.linear_solver.verbose,
            "stability solver verbosity");
    require(
        close_value(
            parameters.stability_continuation.
                stability_boundary_tolerance,
            T(1.0e-7)),
        "stability boundary tolerance default");
    require(
        close_value(
            parameters.stability_continuation.
                real_eigenvalue_tolerance,
            T(1.0e-7)),
        "real eigenvalue tolerance default");
    require(
        close_value(
            parameters.stability_continuation.
                conjugate_pair_tolerance,
            T(1.0e-6)),
        "conjugate-pair tolerance default");
    require(
        parameters.stability_continuation.
            require_converged_eigenpairs,
        "require converged eigenpairs default");
    require(
        parameters.stability_continuation.
            require_complete_scan_coverage,
        "require complete scan coverage default");
    require(
        parameters.stability_continuation.
            spectrum_classification_retries == 2,
        "spectrum classification retry default");
    require(
        parameters.stability_continuation.
            transition_classification_confirmations == 2,
        "transition classification confirmation count");
    require(
        parameters.stability_continuation.
            transition_refinement_maximum_iterations == 20,
        "transition refinement maximum iterations");
    require(
        parameters.stability_continuation.
            transition_refinement_maximum_subdivisions == 8,
        "transition refinement maximum subdivisions");
    require(
        close_value(
            parameters.stability_continuation.
                transition_refinement_parameter_tolerance,
            T(1.0e-8)),
        "transition refinement parameter tolerance");
    require(
        parameters.stability_continuation.
            symmetry_endpoint_guard_source_points == 50,
        "symmetry endpoint source-point guard");
}

template<class T>
void verify_optional_policy_defaults(const std::string& file_name)
{
    auto json = main_classes::read_json(file_name);
    auto& continuation_json = json.at("deflation_continuation");
    continuation_json.erase("restart_policy");
    continuation_json.erase("branch_intersection_policy");
    continuation_json.erase("self_intersection_policy");
    continuation_json.erase("isotropy_transition_policy");
    continuation_json.erase("predictor_chart_policy");
    continuation_json.erase("corrector_retry_policy");
    continuation_json.erase("progress_monitor_policy");
    continuation_json.erase("continuation_parameter_bounds");
    continuation_json.erase("boundary_refinement_policy");
    auto& stability_json = json.at("stability_continuation");
    stability_json.erase("correct_stability_transitions_with_newton");
    stability_json.erase("transition_refinement_maximum_iterations");
    stability_json.erase("transition_refinement_parameter_tolerance");
    stability_json.erase("transition_classification_confirmations");
    stability_json.erase("symmetry_endpoint_guard_source_points");

    const auto parameters = json.get<main_classes::parameters<T>>();
    const auto& continuation = parameters.deflation_continuation;
    require(!continuation.restart_policy.allow_incomplete_restart_intersections,
            "restart default");
    require(!continuation.restart_policy.knot_relocation.enabled,
            "knot relocation default");
    require(!continuation.restart_policy.seed_schedule.enabled,
            "seed schedule default");
    require(!continuation.continuation_parameter_bounds.enabled,
            "continuation bounds default");
    require(
        continuation.boundary_refinement_policy
            .preserve_last_converged_point,
        "boundary transaction default");
    require(!continuation.branch_intersection_policy.enabled,
            "branch intersection default");
    require(!continuation.self_intersection_policy.enabled,
            "self intersection default");
    require(!continuation.isotropy_transition_policy.enabled,
            "isotropy transition default");
    require(continuation.predictor_chart_policy.enabled,
            "predictor chart default");
    require(continuation.predictor_chart_policy.enforce,
            "predictor chart enforcement default");
    require(continuation.corrector_retry_policy.maximum_retries ==
                continuation.continuation_fail_attempts,
            "legacy corrector retry fallback");
    require(continuation.corrector_retry_policy.successes_before_growth == 6,
            "legacy corrector growth fallback");
    require(!continuation.progress_monitor_policy.enabled,
            "progress monitor default");
    const auto& stability = parameters.stability_continuation;
    require(
        !stability.correct_stability_transitions_with_newton,
        "secant stability transition refinement default");
    require(
        stability.transition_refinement_maximum_iterations == 20,
        "transition refinement maximum iterations default");
    require(
        stability.transition_refinement_maximum_subdivisions == 8,
        "transition refinement maximum subdivisions default");
    require(
        close_value(
            stability.transition_refinement_parameter_tolerance,
            T(0)),
        "transition refinement parameter tolerance default");
    require(
        stability.symmetry_endpoint_guard_source_points == 0,
        "symmetry endpoint source-point guard default");
    require(
        stability.spectrum_classification_retries == 2,
        "spectrum classification retry default");
    require(
        stability.transition_classification_confirmations == 2,
        "transition classification confirmation default");
}

template<class T>
void verify_stability_classifier_overrides(const std::string& file_name)
{
    auto json = main_classes::read_json(file_name);
    auto& stability_json = json.at("stability_continuation");
    stability_json["stability_boundary_tolerance"] = 2.5e-5;
    stability_json["real_eigenvalue_tolerance"] = 3.5e-6;
    stability_json["conjugate_pair_tolerance"] = 4.5e-4;
    stability_json["require_converged_eigenpairs"] = false;
    stability_json["require_nonempty_spectrum"] = false;
    stability_json["require_complete_scan_coverage"] = false;
    stability_json["spectrum_classification_retries"] = 4;
    stability_json["transition_classification_confirmations"] = 3;
    stability_json["correct_stability_transitions_with_newton"] =
        false;
    stability_json["transition_refinement_maximum_iterations"] = 37;
    stability_json["transition_refinement_maximum_subdivisions"] = 6;
    stability_json["transition_refinement_parameter_tolerance"] =
        7.5e-9;
    stability_json["symmetry_endpoint_guard_source_points"] = 31;

    const auto parameters =
        json.get<main_classes::parameters<T>>();
    const auto& stability = parameters.stability_continuation;
    require(
        close_value(
            stability.stability_boundary_tolerance,
            T(2.5e-5)),
        "stability boundary tolerance override");
    require(
        close_value(
            stability.real_eigenvalue_tolerance,
            T(3.5e-6)),
        "real eigenvalue tolerance override");
    require(
        close_value(
            stability.conjugate_pair_tolerance,
            T(4.5e-4)),
        "conjugate-pair tolerance override");
    require(
        !stability.require_converged_eigenpairs,
        "require converged eigenpairs override");
    require(
        !stability.require_nonempty_spectrum,
        "require nonempty spectrum override");
    require(
        !stability.require_complete_scan_coverage,
        "require complete scan coverage override");
    require(
        stability.spectrum_classification_retries == 4,
        "spectrum classification retry override");
    require(
        stability.transition_classification_confirmations == 3,
        "transition classification confirmation override");
    require(
        !stability.correct_stability_transitions_with_newton,
        "stability transition refinement override");
    require(
        stability.transition_refinement_maximum_iterations == 37,
        "stability transition maximum iterations override");
    require(
        stability.transition_refinement_maximum_subdivisions == 6,
        "stability transition maximum subdivisions override");
    require(
        close_value(
            stability.transition_refinement_parameter_tolerance,
            T(7.5e-9)),
        "stability transition parameter tolerance override");
    require(
        stability.symmetry_endpoint_guard_source_points == 31,
        "symmetry endpoint source-point guard override");
}

template<class T>
void verify_matrix_free_stability_configuration(
    const std::string& file_name)
{
    auto json = main_classes::read_json(file_name);
    auto& stability_json = json.at("stability_continuation");
    stability_json["matrix_free_eigensolver"] = {
        {"enabled", true},
        {"linearization_scale", -1.0},
        {"transformation", {
            {"type", "explicit_euler"},
            {"step", 0.125},
            {"repetitions", 4},
            {"shifts", {
                {0.25, 1.5},
                {
                    {"real", -0.5},
                    {"imaginary", 3.0}
                }
            }}
        }},
        {"outer", {
            {"desired_eigenvalues", 8},
            {"krylov_dimension", 30},
            {"restart_dimension", 14},
            {"maximum_restarts", 25},
            {"relative_tolerance", 2.0e-9}
        }},
        {"recovery", {
            {"relative_basis_tolerance", 3.0e-10},
            {"orthogonalization_passes", 3},
            {"relative_residual_tolerance", 4.0e-9},
            {"minimum_converged_eigenpairs", 5}
        }},
        {"inner_solver", {
            {"basis_size", 18},
            {"batch_size", 6},
            {"preconditioner_side", "R"},
            {"basis_retry_sizes", {36, 72}},
            {"relative_tolerance", 5.0e-11},
            {"maximum_iterations", 120},
            {"verbose", true}
        }},
        {"retry", {
            {"enabled", true},
            {"maximum_shift_retries", 4},
            {"initial_shift_perturbation", 0.075},
            {"perturbation_growth", 1.5},
            {"preconditioner_pole_absolute_tolerance", 9.0e-12},
            {"preconditioner_pole_relative_tolerance", 2.0e-8}
        }},
        {"aggregation", {
            {"absolute_tolerance", 6.0e-8},
            {"minimum_successful_scans", 2},
            {"minimum_eigenpairs", 7},
            {"require_all_scans", false},
            {"probe_count", 3},
            {"require_all_probes", false},
            {"eigenvector_independence_tolerance", 8.0e-7},
            {"eigenvector_orthogonalization_passes", 3}
        }},
        {"small_system", {
            {"enabled", true},
            {"maximum_dimension", 192},
            {"prefer", true},
            {"absolute_residual_tolerance", 7.0e-10},
            {"relative_residual_tolerance", 9.0e-9}
        }}
    };

    const auto parameters =
        json.get<main_classes::parameters<T>>();
    const auto& config =
        parameters.stability_continuation.
            matrix_free_eigensolver;
    require(config.enabled, "matrix-free eigensolver enabled");
    require(
        close_value(config.linearization_scale, T(-1)),
        "matrix-free linearization scale");
    require(
        config.transformation.type ==
            stability::analysis::
                matrix_free_spectral_transformation::
                    explicit_euler,
        "matrix-free transformation");
    require(
        close_value(config.transformation.step, T(0.125)) &&
            config.transformation.repetitions == 4,
        "matrix-free transformation parameters");
    require(
        config.transformation.shifts.size() == 2 &&
            close_value(
                config.transformation.shifts[0].real(),
                T(0.25)) &&
            close_value(
                config.transformation.shifts[0].imag(),
                T(1.5)) &&
            close_value(
                config.transformation.shifts[1].real(),
                T(-0.5)) &&
            close_value(
                config.transformation.shifts[1].imag(),
                T(3.0)),
        "matrix-free complex shifts");
    require(
        config.outer.desired_eigenvalues == 8 &&
            config.outer.krylov_dimension == 30 &&
            config.outer.restart_dimension == 14 &&
            config.outer.maximum_restarts == 25 &&
            close_value(
                config.outer.relative_tolerance,
                T(2.0e-9)),
        "matrix-free outer solver");
    require(
        close_value(
            config.recovery.relative_basis_tolerance,
            T(3.0e-10)) &&
            config.recovery.orthogonalization_passes == 3 &&
            close_value(
                config.recovery.relative_residual_tolerance,
                T(4.0e-9)) &&
            config.recovery.minimum_converged_eigenpairs == 5,
        "matrix-free recovery");
    require(
        config.inner_solver.basis_size == 18 &&
            config.inner_solver.batch_size == 6 &&
            config.inner_solver.preconditioner_side == 'R' &&
            config.inner_solver.basis_retry_sizes ==
                std::vector<unsigned int>({36, 72}) &&
            close_value(
                config.inner_solver.relative_tolerance,
                T(5.0e-11)) &&
            config.inner_solver.maximum_iterations == 120 &&
            config.inner_solver.verbose,
        "matrix-free inner solver");
    require(
        config.retry.enabled &&
            config.retry.maximum_shift_retries == 4 &&
            close_value(
                config.retry.initial_shift_perturbation,
                T(0.075)) &&
            close_value(
                config.retry.perturbation_growth,
                T(1.5)) &&
            close_value(
                config.retry.
                    preconditioner_pole_absolute_tolerance,
                T(9.0e-12)) &&
            close_value(
                config.retry.
                    preconditioner_pole_relative_tolerance,
                T(2.0e-8)),
        "matrix-free retry policy");
    require(
        config.small_system.enabled &&
            config.small_system.maximum_dimension == 192 &&
            config.small_system.prefer &&
            close_value(
                config.small_system.absolute_residual_tolerance,
                T(7.0e-10)) &&
            close_value(
                config.small_system.relative_residual_tolerance,
                T(9.0e-9)),
        "small-system eigensolver policy");
    require(
        close_value(
            config.aggregation.absolute_tolerance,
            T(6.0e-8)) &&
            config.aggregation.minimum_successful_scans == 2 &&
            config.aggregation.minimum_eigenpairs == 7 &&
            !config.aggregation.require_all_scans &&
            config.aggregation.probe_count == 3 &&
            !config.aggregation.require_all_probes &&
            close_value(
                config.aggregation.
                    eigenvector_independence_tolerance,
                T(8.0e-7)) &&
            config.aggregation.
                eigenvector_orthogonalization_passes == 3,
        "matrix-free aggregation");
}

}

int main(int argc, char** argv)
{
    const std::string file_name = argc > 1
        ? argv[1]
        : "json_project_files/KS1D_test_full.json";
    try
    {
        verify_full_configuration<double>(file_name);
        verify_full_configuration<float>(file_name);
        verify_optional_policy_defaults<double>(file_name);
        verify_optional_policy_defaults<float>(file_name);
        verify_stability_classifier_overrides<double>(file_name);
        verify_stability_classifier_overrides<float>(file_name);
        verify_matrix_free_stability_configuration<double>(file_name);
        verify_matrix_free_stability_configuration<float>(file_name);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
