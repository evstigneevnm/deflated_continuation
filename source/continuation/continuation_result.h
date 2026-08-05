#ifndef __CONTINUATION_CONTINUATION_RESULT_H__
#define __CONTINUATION_CONTINUATION_RESULT_H__

#include <array>
#include <cstdint>
#include <string>

#include <containers/curve_endpoint_reason.h>

namespace continuation
{

enum class continuation_failure_kind
{
    none,
    initial_tangent,
    predictor_chart,
    corrector_retry_limit,
    minimum_step,
    invalid_number,
    linear_solver,
    tangent,
    no_progress,
    maximum_steps,
    unresolved_intersection,
    unknown
};

inline const char* to_string(const continuation_failure_kind kind)
{
    switch(kind)
    {
    case continuation_failure_kind::none:
        return "none";
    case continuation_failure_kind::initial_tangent:
        return "initial_tangent";
    case continuation_failure_kind::predictor_chart:
        return "predictor_chart";
    case continuation_failure_kind::corrector_retry_limit:
        return "corrector_retry_limit";
    case continuation_failure_kind::minimum_step:
        return "minimum_step";
    case continuation_failure_kind::invalid_number:
        return "invalid_number";
    case continuation_failure_kind::linear_solver:
        return "linear_solver";
    case continuation_failure_kind::tangent:
        return "tangent";
    case continuation_failure_kind::no_progress:
        return "no_progress";
    case continuation_failure_kind::maximum_steps:
        return "maximum_steps";
    case continuation_failure_kind::unresolved_intersection:
        return "unresolved_intersection";
    case continuation_failure_kind::unknown:
        return "unknown";
    }
    return "unknown";
}

enum class semicurve_status
{
    not_started,
    complete,
    open_recoverable
};

template<class Scalar>
struct semicurve_result
{
    semicurve_status status = semicurve_status::not_started;
    continuation_failure_kind failure = continuation_failure_kind::none;
    container::curve_endpoint_reason endpoint_reason =
        container::curve_endpoint_reason::none;
    int direction = 0;
    unsigned int accepted_points = 0;
    std::uint64_t segment_id = 0;
    std::uint64_t first_point_index = 0;
    std::uint64_t last_point_index = 0;
    Scalar start_parameter = Scalar(0);
    Scalar last_parameter = Scalar(0);
    Scalar attempted_step = Scalar(0);
    unsigned int retry_count = 0;
    std::string message;

    bool has_progress() const
    {
        // The initial seed is stored in every started semicurve. At least one
        // additional corrected state is required for useful branch progress.
        return accepted_points > 1;
    }

    bool complete() const
    {
        return status == semicurve_status::complete;
    }

    bool recoverable() const
    {
        return status == semicurve_status::open_recoverable && has_progress();
    }
};

template<class Scalar>
struct continuation_curve_result
{
    std::array<semicurve_result<Scalar>, 2> semicurves{};
    unsigned int semicurves_started = 0;
    bool branch_closed = false;

    bool complete() const
    {
        if(semicurves_started == 0)
        {
            return false;
        }
        for(unsigned int index = 0; index < semicurves_started; ++index)
        {
            if(!semicurves[index].complete())
            {
                return false;
            }
        }
        return branch_closed || semicurves_started == semicurves.size();
    }

    bool has_valid_progress() const
    {
        for(const auto& semicurve: semicurves)
        {
            if(semicurve.has_progress())
            {
                return true;
            }
        }
        return false;
    }

    bool has_recoverable_segment() const
    {
        for(const auto& semicurve: semicurves)
        {
            if(semicurve.recoverable())
            {
                return true;
            }
        }
        return false;
    }
};

} // namespace continuation

#endif // __CONTINUATION_CONTINUATION_RESULT_H__
