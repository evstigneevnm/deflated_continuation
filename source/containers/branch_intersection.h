#ifndef __CONTAINER_BRANCH_INTERSECTION_H__
#define __CONTAINER_BRANCH_INTERSECTION_H__

#include <cstdint>
#include <stdexcept>
#include <string>

namespace container
{

template<class T>
struct branch_intersection_policy
{
    bool enabled = false;
    unsigned int signature_norm_index = 0;
    T signature_tolerance = T(1.0e-6);
    T state_tolerance = T(1.0e-8);
    T minimum_step_fraction_from_start = T(1.0e-3);
    bool detect_forward_approach = false;
    T maximum_forward_lookahead_steps = T(1);
    T maximum_forward_distance_step_ratio = T(1);
    T minimum_forward_distance_reduction_ratio = T(0.05);
    unsigned int forward_distance_extrapolation_power = 1;
    T forward_refinement_step_factor = T(0.25);
    unsigned int maximum_forward_refinements = 8;
    unsigned int minimum_forward_refinements_for_verification = 3;
    T maximum_verified_forward_steps_ahead = T(0.1);
    T maximum_verified_forward_distance_step_ratio = T(0.3);
    bool verbose = false;
};

enum class branch_intersection_detection
{
    none,
    verified,
    forward_approach
};

template<class T>
struct forward_branch_approach_result
{
    bool found = false;
    T steps_ahead = T(0);
    T endpoint_distance_step_ratio = T(0);
};

template<class T>
void validate_branch_intersection_policy(const branch_intersection_policy<T>& policy)
{
    if(policy.signature_tolerance < T(0) || policy.state_tolerance < T(0))
    {
        throw std::invalid_argument("branch intersection tolerances must be nonnegative");
    }
    if(policy.minimum_step_fraction_from_start < T(0))
    {
        throw std::invalid_argument("branch intersection minimum step fraction must be nonnegative");
    }
    if(policy.maximum_forward_lookahead_steps <= T(0))
    {
        throw std::invalid_argument("branch intersection forward lookahead must be positive");
    }
    if(policy.maximum_forward_distance_step_ratio <= T(0))
    {
        throw std::invalid_argument("branch intersection forward distance/step ratio must be positive");
    }
    if(policy.minimum_forward_distance_reduction_ratio < T(0) ||
       policy.minimum_forward_distance_reduction_ratio >= T(1))
    {
        throw std::invalid_argument(
            "branch intersection minimum forward distance reduction ratio must be in [0,1)");
    }
    if(policy.forward_distance_extrapolation_power != 1 &&
       policy.forward_distance_extrapolation_power != 2)
    {
        throw std::invalid_argument(
            "branch intersection forward distance extrapolation power must be 1 or 2");
    }
    if(policy.forward_refinement_step_factor <= T(0) ||
       policy.forward_refinement_step_factor >= T(1))
    {
        throw std::invalid_argument(
            "branch intersection forward refinement step factor must be in (0,1)");
    }
    if(policy.maximum_forward_refinements == 0)
    {
        throw std::invalid_argument(
            "branch intersection maximum forward refinements must be positive");
    }
    if(policy.minimum_forward_refinements_for_verification == 0 ||
       policy.minimum_forward_refinements_for_verification >
           policy.maximum_forward_refinements)
    {
        throw std::invalid_argument(
            "branch intersection minimum verification refinements must be positive and not exceed the refinement limit");
    }
    if(policy.maximum_verified_forward_steps_ahead <= T(0) ||
       policy.maximum_verified_forward_steps_ahead >
           policy.maximum_forward_lookahead_steps)
    {
        throw std::invalid_argument(
            "branch intersection verified forward lookahead must be positive and not exceed the detection lookahead");
    }
    if(policy.maximum_verified_forward_distance_step_ratio <= T(0) ||
       policy.maximum_verified_forward_distance_step_ratio >
           policy.maximum_forward_distance_step_ratio)
    {
        throw std::invalid_argument(
            "branch intersection verified distance/step ratio must be positive and not exceed the detection ratio");
    }
}

template<class T>
forward_branch_approach_result<T> evaluate_forward_branch_approach(
    const T& previous_distance,
    const T& endpoint_distance,
    const T& accepted_step_distance,
    const branch_intersection_policy<T>& policy)
{
    forward_branch_approach_result<T> result;
    if(!policy.detect_forward_approach ||
       previous_distance <= T(0) ||
       endpoint_distance < T(0) ||
       accepted_step_distance <= T(0) ||
       endpoint_distance >= previous_distance)
    {
        return result;
    }

    const T distance_reduction = previous_distance - endpoint_distance;
    if(distance_reduction <
       policy.minimum_forward_distance_reduction_ratio*previous_distance)
    {
        return result;
    }

    const T previous_metric =
        policy.forward_distance_extrapolation_power == 2
            ? previous_distance*previous_distance
            : previous_distance;
    const T endpoint_metric =
        policy.forward_distance_extrapolation_power == 2
            ? endpoint_distance*endpoint_distance
            : endpoint_distance;
    const T metric_reduction = previous_metric - endpoint_metric;
    if(metric_reduction <= T(0))
    {
        return result;
    }

    result.steps_ahead = endpoint_metric/metric_reduction;
    result.endpoint_distance_step_ratio = endpoint_distance/accepted_step_distance;
    result.found =
        result.steps_ahead <= policy.maximum_forward_lookahead_steps &&
        result.endpoint_distance_step_ratio <=
            policy.maximum_forward_distance_step_ratio;
    return result;
}

template<class T>
bool forward_branch_event_is_localized(
    const unsigned int refinement_count,
    const T& steps_ahead,
    const T& endpoint_distance_step_ratio,
    const unsigned int minimum_refinements,
    const T& maximum_steps_ahead,
    const T& maximum_distance_step_ratio)
{
    return
        refinement_count >= minimum_refinements &&
        steps_ahead >= T(0) &&
        steps_ahead <= maximum_steps_ahead &&
        endpoint_distance_step_ratio >= T(0) &&
        endpoint_distance_step_ratio <= maximum_distance_step_ratio;
}

template<class T>
bool forward_branch_event_is_localized(
    const unsigned int refinement_count,
    const T& steps_ahead,
    const T& endpoint_distance_step_ratio,
    const branch_intersection_policy<T>& policy)
{
    return forward_branch_event_is_localized(
        refinement_count,
        steps_ahead,
        endpoint_distance_step_ratio,
        policy.minimum_forward_refinements_for_verification,
        policy.maximum_verified_forward_steps_ahead,
        policy.maximum_verified_forward_distance_step_ratio);
}

template<class T>
struct self_intersection_policy
{
    bool enabled = false;
    unsigned int signature_norm_index = 0;
    T signature_tolerance = T(1.0e-6);
    T state_tolerance = T(1.0e-8);
    T minimum_step_fraction_from_start = T(1.0e-3);
    uint64_t minimum_index_gap = 50;
    bool verbose = false;
};

template<class T>
struct branch_intersection_result
{
    bool found = false;
    bool forward_approach = false;
    branch_intersection_detection detection = branch_intersection_detection::none;
    T lambda = T(0);
    T signature_distance = T(0);
    T state_distance = T(0);
    T state_tolerance = T(0);
    T forward_steps_ahead = T(0);
    T endpoint_distance_step_ratio = T(0);
    int curve_number = -1;
    uint64_t segment_id = 0;
    uint64_t semicurve_id = 0;
    uint64_t lower_point_index = 0;
    uint64_t upper_point_index = 0;
    std::string reason;
};

} // namespace container

#endif // __CONTAINER_BRANCH_INTERSECTION_H__
