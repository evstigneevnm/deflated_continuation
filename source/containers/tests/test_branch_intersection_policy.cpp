#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <containers/branch_intersection.h>

namespace
{

bool close(const double left, const double right, const double tolerance = 1.0e-14)
{
    return std::abs(left - right) <= tolerance;
}

void require(const bool condition, const char* message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    using policy_t = container::branch_intersection_policy<double>;

    policy_t policy;
    auto result = container::evaluate_forward_branch_approach(2.0, 1.0, 1.0, policy);
    require(!result.found, "forward approach must be disabled by default");

    policy.detect_forward_approach = true;
    container::validate_branch_intersection_policy(policy);
    result = container::evaluate_forward_branch_approach(2.0, 1.0, 1.0, policy);
    require(result.found, "one-step linear approach was not detected");
    require(close(result.steps_ahead, 1.0), "linear lookahead estimate is incorrect");
    require(
        close(result.endpoint_distance_step_ratio, 1.0),
        "endpoint distance/step ratio is incorrect");

    policy.forward_distance_extrapolation_power = 2;
    result = container::evaluate_forward_branch_approach(2.0, 1.0, 1.0, policy);
    require(result.found, "quadratic-distance approach was not detected");
    require(close(result.steps_ahead, 1.0/3.0), "quadratic lookahead estimate is incorrect");

    result = container::evaluate_forward_branch_approach(1.0, 1.1, 0.1, policy);
    require(!result.found, "a receding branch was accepted as an approach");

    result = container::evaluate_forward_branch_approach(1.0, 0.99, 0.01, policy);
    require(!result.found, "a slow distant approach exceeded the lookahead budget");

    result = container::evaluate_forward_branch_approach(2.0, 0.5, 0.1, policy);
    require(!result.found, "an endpoint too far from the branch was accepted");

    const double ks_previous_distance = 0.01445716409237621;
    const double ks_endpoint_distance = 0.006519656143624547;
    result = container::evaluate_forward_branch_approach(
        ks_previous_distance,
        ks_endpoint_distance,
        ks_previous_distance - ks_endpoint_distance,
        policy);
    require(result.found, "the recorded KS1D pitchfork approach was not detected");
    const double ks_lambda_previous = 16.0000042182963;
    const double ks_lambda_endpoint = 16.00000085787678;
    const double ks_lambda_intersection =
        ks_lambda_endpoint +
        result.steps_ahead*(ks_lambda_endpoint - ks_lambda_previous);
    require(
        close(ks_lambda_intersection, 16.0, 1.0e-8),
        "the KS1D pitchfork parameter extrapolation is inaccurate");

    require(
        !container::forward_branch_event_is_localized(2u, 0.05, 0.2, policy),
        "an event was localized before the minimum refinement count");
    require(
        container::forward_branch_event_is_localized(3u, 0.05, 0.2, policy),
        "a refined local branch event was not accepted");
    require(
        !container::forward_branch_event_is_localized(3u, 0.2, 0.2, policy),
        "an event beyond the verification lookahead was accepted");
    require(
        !container::forward_branch_event_is_localized(3u, 0.05, 0.4, policy),
        "an event beyond the verification distance/step ratio was accepted");

    policy.minimum_forward_distance_reduction_ratio = 0.5;
    result = container::evaluate_forward_branch_approach(1.0, 0.6, 0.5, policy);
    require(!result.found, "insufficient relative distance reduction was accepted");

    bool invalid_power_rejected = false;
    try
    {
        policy.forward_distance_extrapolation_power = 3;
        container::validate_branch_intersection_policy(policy);
    }
    catch(const std::invalid_argument&)
    {
        invalid_power_rejected = true;
    }
    require(invalid_power_rejected, "invalid extrapolation power was not rejected");

    bool invalid_refinement_factor_rejected = false;
    try
    {
        policy.forward_distance_extrapolation_power = 2;
        policy.forward_refinement_step_factor = 1.0;
        container::validate_branch_intersection_policy(policy);
    }
    catch(const std::invalid_argument&)
    {
        invalid_refinement_factor_rejected = true;
    }
    require(
        invalid_refinement_factor_rejected,
        "invalid forward-refinement step factor was not rejected");

    bool zero_refinement_budget_rejected = false;
    try
    {
        policy.forward_refinement_step_factor = 0.25;
        policy.maximum_forward_refinements = 0;
        container::validate_branch_intersection_policy(policy);
    }
    catch(const std::invalid_argument&)
    {
        zero_refinement_budget_rejected = true;
    }
    require(zero_refinement_budget_rejected, "zero forward-refinement budget was not rejected");

    bool invalid_analytical_tolerance_rejected = false;
    try
    {
        policy.maximum_forward_refinements = 8;
        policy.analytical_target_parameter_tolerance = 0.0;
        container::validate_branch_intersection_policy(policy);
    }
    catch(const std::invalid_argument&)
    {
        invalid_analytical_tolerance_rejected = true;
    }
    require(
        invalid_analytical_tolerance_rejected,
        "zero analytical-target parameter tolerance was not rejected");

    std::cout << "branch intersection policy tests passed\n";
    return EXIT_SUCCESS;
}
