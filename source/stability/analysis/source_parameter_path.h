#ifndef __STABILITY_ANALYSIS_SOURCE_PARAMETER_PATH_H__
#define __STABILITY_ANALYSIS_SOURCE_PARAMETER_PATH_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace stability
{
namespace analysis
{

template<class Real>
struct source_parameter_sample
{
    std::uint64_t source_index = 0;
    Real parameter = Real{};
};

template<class Real>
struct source_parameter_monotone_span
{
    std::size_t begin_position = 0;
    std::size_t end_position = 0;
    int direction = 0;
};

template<class Real>
struct source_parameter_path_result
{
    bool valid = false;
    bool monotone = false;
    std::vector<std::size_t> turning_positions;
    std::vector<source_parameter_monotone_span<Real>> monotone_spans;
    std::string diagnostic;
};

template<class Real>
struct source_parameter_turning_estimate
{
    bool valid = false;
    std::size_t left_position = 0;
    std::size_t right_position = 0;
    Real right_fraction = Real{};
    Real parameter = Real{};
    std::string diagnostic;
};

template<class Real>
struct source_parameter_terminal_step_result
{
    bool valid = false;
    bool discontinuous = false;
    int preceding_direction = 0;
    int terminal_direction = 0;
    Real reference_step = Real{};
    Real terminal_step = Real{};
    Real step_ratio = Real{};
    std::string diagnostic;
};

enum class source_path_turning_join_status
{
    joined,
    topology_split,
    rejected
};

template<class Real>
struct source_path_turning_join_result
{
    source_path_turning_join_status status =
        source_path_turning_join_status::rejected;
    bool forward_matches = false;
    bool reverse_matches = false;
    std::string diagnostic;

    bool accepted() const
    {
        return status != source_path_turning_join_status::rejected;
    }
};

template<class Real>
source_path_turning_join_result<Real> decide_source_path_turning_join(
    std::uint64_t left_source_index,
    std::uint64_t right_source_index,
    Real guard_relative_distance,
    Real forward_cross_relative_distance,
    Real reverse_cross_relative_distance,
    Real matching_tolerance,
    bool allow_topology_split)
{
    source_path_turning_join_result<Real> result;
    result.forward_matches =
        forward_cross_relative_distance <= matching_tolerance;
    result.reverse_matches =
        reverse_cross_relative_distance <= matching_tolerance;

    std::ostringstream diagnostic;
    diagnostic
        << "one-sided source-path turning guards across sources ["
        << left_source_index << ',' << right_source_index << "] have "
        << "guard distance " << guard_relative_distance
        << ", forward crossing distance "
        << forward_cross_relative_distance
        << ", reverse crossing distance "
        << reverse_cross_relative_distance
        << ", tolerance " << matching_tolerance;

    if(result.forward_matches || result.reverse_matches)
    {
        result.status = source_path_turning_join_status::joined;
        diagnostic << "; at least one crossing joins the guards";
    }
    else if(allow_topology_split)
    {
        result.status = source_path_turning_join_status::topology_split;
        diagnostic << "; preserving the non-joining bracket as a "
                      "topology split";
    }
    else
    {
        result.status = source_path_turning_join_status::rejected;
        diagnostic << "; neither crossing joins the guards";
    }
    result.diagnostic = diagnostic.str();
    return result;
}

namespace detail
{

template<class Real>
Real source_parameter_abs(Real value)
{
    using std::abs;
    return abs(value);
}

template<class Real>
int source_parameter_direction(
    Real left,
    Real right,
    Real relative_tolerance)
{
    const Real scale = std::max<Real>(
        Real(1),
        std::max(
            source_parameter_abs(left),
            source_parameter_abs(right)));
    const Real difference = right - left;
    if(source_parameter_abs(difference) <= relative_tolerance*scale)
        return 0;
    return difference > Real(0) ? 1 : -1;
}

template<class Real>
bool source_parameter_is_finite(const Real& value)
{
    using std::isfinite;
    return isfinite(value);
}

} // namespace detail

template<class Real>
source_parameter_path_result<Real> analyze_source_parameter_path(
    const std::vector<source_parameter_sample<Real>>& samples,
    Real relative_tolerance =
        Real(128)*std::numeric_limits<Real>::epsilon())
{
    source_parameter_path_result<Real> result;
    if(samples.size() < 2)
    {
        result.diagnostic =
            "source-parameter path requires at least two samples";
        return result;
    }
    if(relative_tolerance < Real(0))
    {
        result.diagnostic =
            "source-parameter path tolerance must be nonnegative";
        return result;
    }
    for(std::size_t index = 1; index < samples.size(); ++index)
    {
        if(samples[index].source_index <= samples[index - 1].source_index)
        {
            result.diagnostic =
                "source-parameter path indices must increase strictly";
            return result;
        }
        if(
            !detail::source_parameter_is_finite(
                samples[index - 1].parameter) ||
            !detail::source_parameter_is_finite(samples[index].parameter))
        {
            result.diagnostic =
                "source-parameter path contains a non-finite parameter";
            return result;
        }
    }

    std::size_t span_begin = 0;
    int span_direction = 0;
    for(std::size_t index = 1; index < samples.size(); ++index)
    {
        const int direction = detail::source_parameter_direction(
            samples[index - 1].parameter,
            samples[index].parameter,
            relative_tolerance);
        if(direction == 0)
            continue;
        if(span_direction == 0)
        {
            span_direction = direction;
            continue;
        }
        if(direction == span_direction)
            continue;

        const std::size_t turning_position = index - 1;
        result.turning_positions.push_back(turning_position);
        result.monotone_spans.push_back(
            {span_begin, turning_position, span_direction});
        span_begin = turning_position;
        span_direction = direction;
    }

    result.monotone_spans.push_back(
        {span_begin, samples.size() - 1, span_direction});
    result.valid = true;
    result.monotone = result.turning_positions.empty();
    result.diagnostic = result.monotone
        ? "source-parameter path is monotone"
        : "source-parameter path contains " +
              std::to_string(result.turning_positions.size()) +
              " turning point(s)";
    return result;
}

template<class Real>
source_parameter_turning_estimate<Real>
estimate_source_parameter_turning_point(
    const std::vector<source_parameter_sample<Real>>& samples,
    std::size_t turning_position,
    Real relative_tolerance =
        Real(128)*std::numeric_limits<Real>::epsilon())
{
    source_parameter_turning_estimate<Real> result;
    if(relative_tolerance < Real(0))
    {
        result.diagnostic =
            "turning-point tolerance must be nonnegative";
        return result;
    }
    if(
        turning_position == 0 ||
        turning_position + 1 >= samples.size())
    {
        result.diagnostic =
            "turning-point estimate requires one sample on each side";
        return result;
    }

    const Real previous = samples[turning_position - 1].parameter;
    const Real current = samples[turning_position].parameter;
    const Real next = samples[turning_position + 1].parameter;
    if(
        !detail::source_parameter_is_finite(previous) ||
        !detail::source_parameter_is_finite(current) ||
        !detail::source_parameter_is_finite(next))
    {
        result.diagnostic =
            "turning-point estimate contains a non-finite parameter";
        return result;
    }

    const Real curvature = previous - Real(2)*current + next;
    const Real scale = std::max<Real>(
        Real(1),
        std::max(
            detail::source_parameter_abs(previous),
            std::max(
                detail::source_parameter_abs(current),
                detail::source_parameter_abs(next))));
    if(detail::source_parameter_abs(curvature) <= relative_tolerance*scale)
    {
        result.diagnostic =
            "turning-point estimate has unresolved local curvature";
        return result;
    }

    const Real offset =
        Real(0.5)*(previous - next)/curvature;
    if(!detail::source_parameter_is_finite(offset))
    {
        result.diagnostic =
            "turning-point estimate produced a non-finite offset";
        return result;
    }
    if(offset < Real(-1) || offset > Real(1))
    {
        result.diagnostic =
            "turning-point estimate lies outside the local source "
            "interval";
        return result;
    }

    if(offset < Real(0))
    {
        result.left_position = turning_position - 1;
        result.right_position = turning_position;
        result.right_fraction = offset + Real(1);
    }
    else
    {
        result.left_position = turning_position;
        result.right_position = turning_position + 1;
        result.right_fraction = offset;
    }
    if(
        result.right_fraction < Real(0) ||
        result.right_fraction > Real(1))
    {
        result.diagnostic =
            "turning-point interpolation fraction is outside [0,1]";
        return result;
    }

    const Real linear = Real(0.5)*(next - previous);
    const Real quadratic = Real(0.5)*curvature;
    result.parameter =
        current + linear*offset + quadratic*offset*offset;
    result.valid = detail::source_parameter_is_finite(result.parameter);
    result.diagnostic = result.valid
        ? "turning point estimated by a local quadratic in source order"
        : "turning-point parameter estimate is non-finite";
    return result;
}

template<class Real>
source_parameter_terminal_step_result<Real>
analyze_source_parameter_terminal_step(
    const std::vector<source_parameter_sample<Real>>& samples,
    Real discontinuity_step_ratio = Real(32),
    Real relative_tolerance =
        Real(128)*std::numeric_limits<Real>::epsilon())
{
    source_parameter_terminal_step_result<Real> result;
    if(samples.size() < 3)
    {
        result.diagnostic =
            "terminal source-parameter step requires at least three "
            "samples";
        return result;
    }
    if(
        discontinuity_step_ratio <= Real(1) ||
        relative_tolerance < Real(0))
    {
        result.diagnostic =
            "terminal source-parameter step requires a ratio greater "
            "than one and a nonnegative tolerance";
        return result;
    }

    Real maximum_preceding_step = Real{};
    int preceding_direction = 0;
    for(std::size_t index = 1; index + 1 < samples.size(); ++index)
    {
        const Real left = samples[index - 1].parameter;
        const Real right = samples[index].parameter;
        if(
            !detail::source_parameter_is_finite(left) ||
            !detail::source_parameter_is_finite(right))
        {
            result.diagnostic =
                "terminal source-parameter step contains a non-finite "
                "preceding parameter";
            return result;
        }
        const int direction = detail::source_parameter_direction(
            left,
            right,
            relative_tolerance);
        if(direction != 0)
            preceding_direction = direction;
        maximum_preceding_step = std::max(
            maximum_preceding_step,
            detail::source_parameter_abs(right - left));
    }

    const Real terminal_left =
        samples[samples.size() - 2].parameter;
    const Real terminal_right = samples.back().parameter;
    if(
        !detail::source_parameter_is_finite(terminal_left) ||
        !detail::source_parameter_is_finite(terminal_right))
    {
        result.diagnostic =
            "terminal source-parameter step contains a non-finite "
            "terminal parameter";
        return result;
    }

    result.preceding_direction = preceding_direction;
    result.terminal_direction = detail::source_parameter_direction(
        terminal_left,
        terminal_right,
        relative_tolerance);
    result.reference_step = maximum_preceding_step;
    result.terminal_step = detail::source_parameter_abs(
        terminal_right - terminal_left);
    if(maximum_preceding_step > Real(0))
    {
        result.step_ratio =
            result.terminal_step/maximum_preceding_step;
    }
    result.valid = true;
    result.discontinuous =
        preceding_direction != 0 &&
        result.terminal_direction != 0 &&
        preceding_direction != result.terminal_direction &&
        maximum_preceding_step > Real(0) &&
        result.step_ratio >= discontinuity_step_ratio;
    if(result.discontinuous)
    {
        result.diagnostic =
            "terminal source-parameter step reverses direction with "
            "an excessive step ratio";
    }
    else
    {
        result.diagnostic =
            "terminal source-parameter step is locally consistent";
    }
    return result;
}

} // namespace analysis
} // namespace stability

#endif // __STABILITY_ANALYSIS_SOURCE_PARAMETER_PATH_H__
