#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

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
    require(continuation.deflation_knots.size() == 39, "deflation_knots");
    require(continuation.add_analytical_solution_to_diagram, "analytical branch flag");
    require(continuation.analytical_solution_branches.empty(), "analytical branch selection");

    require(continuation.restart_policy.knot_relocation.enabled, "knot relocation");
    require(continuation.restart_policy.knot_relocation.candidate_count == 12, "relocation candidates");
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
    require(parameters.nonlinear_operator.N_size.front() == 64, "nonlinear dimension");
    require(parameters.nonlinear_operator.problem_real_parameters_vector.size() == 2,
            "problem real parameters");
    require(!parameters.stability_continuation.linear_solver.verbose,
            "stability solver verbosity");
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

    const auto parameters = json.get<main_classes::parameters<T>>();
    const auto& continuation = parameters.deflation_continuation;
    require(!continuation.restart_policy.allow_incomplete_restart_intersections,
            "restart default");
    require(!continuation.restart_policy.knot_relocation.enabled,
            "knot relocation default");
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
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
