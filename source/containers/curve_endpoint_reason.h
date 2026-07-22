#ifndef __CURVE_ENDPOINT_REASON_H__
#define __CURVE_ENDPOINT_REASON_H__

#include <string>

namespace container
{

enum class curve_endpoint_reason
{
    none,
    boundary_min,
    boundary_max,
    known_branch,
    analytical_branch,
    symmetry_intersection,
    closed_return,
    self_intersection,
    max_steps,
    no_progress,
    unresolved_branch_intersection,
    hard_failure,
    knot_interpolation_failure
};

inline const char* to_string(curve_endpoint_reason reason)
{
    switch(reason)
    {
    case curve_endpoint_reason::none:
        return "none";
    case curve_endpoint_reason::boundary_min:
        return "boundary_min";
    case curve_endpoint_reason::boundary_max:
        return "boundary_max";
    case curve_endpoint_reason::known_branch:
        return "known_branch";
    case curve_endpoint_reason::analytical_branch:
        return "analytical_branch";
    case curve_endpoint_reason::symmetry_intersection:
        return "symmetry_intersection";
    case curve_endpoint_reason::closed_return:
        return "closed_return";
    case curve_endpoint_reason::self_intersection:
        return "self_intersection";
    case curve_endpoint_reason::max_steps:
        return "max_steps";
    case curve_endpoint_reason::no_progress:
        return "no_progress";
    case curve_endpoint_reason::unresolved_branch_intersection:
        return "unresolved_branch_intersection";
    case curve_endpoint_reason::hard_failure:
        return "hard_failure";
    case curve_endpoint_reason::knot_interpolation_failure:
        return "knot_interpolation_failure";
    }
    return "none";
}

inline curve_endpoint_reason curve_endpoint_reason_from_string(const std::string& value)
{
    if(value == "boundary_min")
        return curve_endpoint_reason::boundary_min;
    if(value == "boundary_max")
        return curve_endpoint_reason::boundary_max;
    if(value == "known_branch")
        return curve_endpoint_reason::known_branch;
    if(value == "analytical_branch")
        return curve_endpoint_reason::analytical_branch;
    if(value == "symmetry_intersection")
        return curve_endpoint_reason::symmetry_intersection;
    if(value == "closed_return")
        return curve_endpoint_reason::closed_return;
    if(value == "self_intersection")
        return curve_endpoint_reason::self_intersection;
    if(value == "max_steps")
        return curve_endpoint_reason::max_steps;
    if(value == "no_progress")
        return curve_endpoint_reason::no_progress;
    if(value == "unresolved_branch_intersection")
        return curve_endpoint_reason::unresolved_branch_intersection;
    if(value == "hard_failure")
        return curve_endpoint_reason::hard_failure;
    if(value == "knot_interpolation_failure")
        return curve_endpoint_reason::knot_interpolation_failure;
    return curve_endpoint_reason::none;
}

inline bool is_incomplete_endpoint(curve_endpoint_reason reason)
{
    return reason == curve_endpoint_reason::max_steps ||
           reason == curve_endpoint_reason::no_progress ||
           reason == curve_endpoint_reason::unresolved_branch_intersection ||
           reason == curve_endpoint_reason::hard_failure ||
           reason == curve_endpoint_reason::knot_interpolation_failure;
}

inline bool is_terminal_endpoint(curve_endpoint_reason reason)
{
    return reason != curve_endpoint_reason::none;
}

} // namespace container

#endif // __CURVE_ENDPOINT_REASON_H__
