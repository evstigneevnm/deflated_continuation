#ifndef __BIFURCATION_DIAGRAM_CURVE_INTERSECTION_SEARCH_H__
#define __BIFURCATION_DIAGRAM_CURVE_INTERSECTION_SEARCH_H__

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <containers/branch_intersection.h>
#include <containers/curve_endpoint_reason.h>

namespace container
{

template<class VectorOperations, class Interpolator, class Point>
class curve_intersection_search
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using point_container_type = std::vector<Point>;

    void bind(
        VectorOperations* vector_operations,
        Interpolator* interpolator,
        point_container_type* points,
        const std::vector<uint64_t>* incomplete_segment_ids,
        vector_type* first_work,
        vector_type* second_work)
    {
        vector_operations_ = vector_operations;
        interpolator_ = interpolator;
        points_ = points;
        incomplete_segment_ids_ = incomplete_segment_ids;
        first_work_ = first_work;
        second_work_ = second_work;
    }

    template<class StateDistance>
    bool find_branch_intersection(
        const scalar_type& step_lambda0,
        const vector_type& step_x0,
        const scalar_type& step_lambda1,
        const vector_type& step_x1,
        const std::vector<scalar_type>& step_norms0,
        const std::vector<scalar_type>& step_norms1,
        const branch_intersection_policy<scalar_type>& policy,
        const bool segment_metadata_available,
        const std::string& directory,
        const int curve_number,
        vector_type& hit_x,
        branch_intersection_result<scalar_type>& result,
        StateDistance&& state_distance)
    {
        if(!policy.enabled ||
           step_norms0.size() <= policy.signature_norm_index ||
           step_norms1.size() <= policy.signature_norm_index)
        {
            return false;
        }

        const scalar_type step_signature0 = step_norms0[policy.signature_norm_index];
        const scalar_type step_signature1 = step_norms1[policy.signature_norm_index];
        const int point_count = static_cast<int>(points_->size());
        for(int index = 0; index < point_count - 1; ++index)
        {
            const auto& lower = (*points_)[static_cast<std::size_t>(index)];
            const auto& upper = (*points_)[static_cast<std::size_t>(index + 1)];
            if(!can_interpolate_between(lower, upper, segment_metadata_available))
            {
                continue;
            }

            scalar_type lambda_lower = scalar_type(0);
            scalar_type lambda_upper = scalar_type(0);
            if(!interval_overlap(
                   step_lambda0,
                   step_lambda1,
                   lower.lambda,
                   upper.lambda,
                   lambda_lower,
                   lambda_upper))
            {
                continue;
            }

            scalar_type old_signature0 = scalar_type(0);
            scalar_type old_signature1 = scalar_type(0);
            if(!get_signature_value(lower, policy.signature_norm_index, old_signature0) ||
               !get_signature_value(upper, policy.signature_norm_index, old_signature1))
            {
                continue;
            }

            const scalar_type signature_scale = std::max<scalar_type>(
                scalar_type(1),
                std::max<scalar_type>(
                    std::max<scalar_type>(abs_value(step_signature0), abs_value(step_signature1)),
                    std::max<scalar_type>(abs_value(old_signature0), abs_value(old_signature1))));
            const scalar_type signature_tolerance = policy.signature_tolerance*signature_scale;
            if(!signature_envelopes_overlap(
                   step_signature0,
                   step_signature1,
                   old_signature0,
                   old_signature1,
                   signature_tolerance))
            {
                continue;
            }

            scalar_type candidate_lambda = scalar_type(0);
            scalar_type signature_distance = scalar_type(0);
            if(!find_signature_candidate_lambda(
                   lambda_lower,
                   lambda_upper,
                   step_lambda0,
                   step_signature0,
                   step_lambda1,
                   step_signature1,
                   lower.lambda,
                   old_signature0,
                   upper.lambda,
                   old_signature1,
                   signature_tolerance,
                   candidate_lambda,
                   signature_distance))
            {
                continue;
            }
            if(!candidate_has_step_progress(
                   candidate_lambda,
                   step_lambda0,
                   step_lambda1,
                   policy.minimum_step_fraction_from_start))
            {
                candidate_lambda = lambda_upper;
                const scalar_type upper_difference =
                    interpolate_scalar(
                        candidate_lambda,
                        step_lambda0,
                        step_signature0,
                        step_lambda1,
                        step_signature1) -
                    interpolate_scalar(
                        candidate_lambda,
                        lower.lambda,
                        old_signature0,
                        upper.lambda,
                        old_signature1);
                signature_distance = abs_value(upper_difference);
                if(signature_distance > signature_tolerance ||
                   !candidate_has_step_progress(
                       candidate_lambda,
                       step_lambda0,
                       step_lambda1,
                       policy.minimum_step_fraction_from_start))
                {
                    candidate_lambda = scalar_type(0.5)*(lambda_lower + lambda_upper);
                    const scalar_type midpoint_difference =
                        interpolate_scalar(
                            candidate_lambda,
                            step_lambda0,
                            step_signature0,
                            step_lambda1,
                            step_signature1) -
                        interpolate_scalar(
                            candidate_lambda,
                            lower.lambda,
                            old_signature0,
                            upper.lambda,
                            old_signature1);
                    signature_distance = abs_value(midpoint_difference);
                    if(signature_distance > signature_tolerance ||
                       !candidate_has_step_progress(
                           candidate_lambda,
                           step_lambda0,
                           step_lambda1,
                           policy.minimum_step_fraction_from_start))
                    {
                        continue;
                    }
                }
            }

            if(!evaluate_segment(
                   index,
                   index + 1,
                   candidate_lambda,
                   segment_metadata_available,
                   directory,
                   *second_work_))
            {
                continue;
            }

            const scalar_type weight = interpolation_weight(
                candidate_lambda,
                step_lambda0,
                step_lambda1);
            vector_operations_->assign_mul(
                scalar_type(1) - weight,
                step_x0,
                weight,
                step_x1,
                *first_work_);
            const scalar_type distance = static_cast<scalar_type>(
                state_distance(*first_work_, *second_work_));
            const scalar_type state_scale = state_signature_scale(
                candidate_lambda,
                step_lambda0,
                step_signature0,
                step_lambda1,
                step_signature1,
                lower.lambda,
                old_signature0,
                upper.lambda,
                old_signature1);
            const scalar_type state_tolerance = policy.state_tolerance*state_scale;
            if(distance <= state_tolerance)
            {
                vector_operations_->assign(*second_work_, hit_x);
                set_result(
                    result,
                    branch_intersection_detection::verified,
                    candidate_lambda,
                    signature_distance,
                    distance,
                    state_tolerance,
                    curve_number,
                    lower,
                    upper,
                    "known_branch_intersection");
                return true;
            }
        }

        if(!policy.detect_forward_approach)
        {
            return false;
        }

        const scalar_type accepted_step_distance = static_cast<scalar_type>(
            state_distance(step_x0, step_x1));
        if(accepted_step_distance <= scalar_type(0))
        {
            return false;
        }

        for(int index = 0; index < point_count - 1; ++index)
        {
            const auto& lower = (*points_)[static_cast<std::size_t>(index)];
            const auto& upper = (*points_)[static_cast<std::size_t>(index + 1)];
            if(!can_interpolate_between(lower, upper, segment_metadata_available) ||
               !contains_lambda(lower, upper, step_lambda0) ||
               !contains_lambda(lower, upper, step_lambda1))
            {
                continue;
            }

            scalar_type old_signature_lower = scalar_type(0);
            scalar_type old_signature_upper = scalar_type(0);
            if(!get_signature_value(lower, policy.signature_norm_index, old_signature_lower) ||
               !get_signature_value(upper, policy.signature_norm_index, old_signature_upper))
            {
                continue;
            }
            const scalar_type old_signature0 = interpolate_scalar(
                step_lambda0,
                lower.lambda,
                old_signature_lower,
                upper.lambda,
                old_signature_upper);
            const scalar_type old_signature1 = interpolate_scalar(
                step_lambda1,
                lower.lambda,
                old_signature_lower,
                upper.lambda,
                old_signature_upper);

            const scalar_type signature_distance0 = abs_value(step_signature0 - old_signature0);
            const scalar_type signature_distance1 = abs_value(step_signature1 - old_signature1);
            const scalar_type signature_reduction = abs_value(signature_distance0 - signature_distance1);
            const auto signature_approach = evaluate_forward_branch_approach(
                signature_distance0,
                signature_distance1,
                signature_reduction,
                policy);
            const scalar_type signature_scale = std::max<scalar_type>(
                scalar_type(1),
                std::max<scalar_type>(abs_value(step_signature1), abs_value(old_signature1)));
            if(!signature_approach.found &&
               signature_distance1 > policy.signature_tolerance*signature_scale)
            {
                continue;
            }

            if(!evaluate_segment(
                   index,
                   index + 1,
                   step_lambda0,
                   segment_metadata_available,
                   directory,
                   *first_work_))
            {
                continue;
            }
            const scalar_type state_distance0 = static_cast<scalar_type>(
                state_distance(step_x0, *first_work_));
            if(!evaluate_segment(
                   index,
                   index + 1,
                   step_lambda1,
                   segment_metadata_available,
                   directory,
                   *second_work_))
            {
                continue;
            }
            const scalar_type state_distance1 = static_cast<scalar_type>(
                state_distance(step_x1, *second_work_));
            const auto state_approach = evaluate_forward_branch_approach(
                state_distance0,
                state_distance1,
                accepted_step_distance,
                policy);
            if(!state_approach.found)
            {
                continue;
            }

            const scalar_type candidate_lambda =
                step_lambda1 +
                state_approach.steps_ahead*(step_lambda1 - step_lambda0);
            if(!evaluate_at_lambda(
                   candidate_lambda,
                   segment_metadata_available,
                   directory,
                   hit_x))
            {
                continue;
            }
            set_result(
                result,
                branch_intersection_detection::forward_approach,
                candidate_lambda,
                signature_distance1,
                state_distance1,
                std::max<scalar_type>(
                    policy.state_tolerance*signature_scale,
                    policy.maximum_forward_distance_step_ratio*accepted_step_distance),
                curve_number,
                lower,
                upper,
                "known_branch_forward_approach");
            result.forward_approach = true;
            result.forward_steps_ahead = state_approach.steps_ahead;
            result.endpoint_distance_step_ratio =
                state_approach.endpoint_distance_step_ratio;
            return true;
        }
        return false;
    }

    template<class StateDistance>
    bool find_self_intersection(
        const scalar_type& step_lambda0,
        const vector_type& step_x0,
        const scalar_type& step_lambda1,
        const vector_type& step_x1,
        const std::vector<scalar_type>& step_norms0,
        const std::vector<scalar_type>& step_norms1,
        const self_intersection_policy<scalar_type>& policy,
        const bool segment_metadata_available,
        const std::string& directory,
        const int curve_number,
        vector_type& hit_x,
        branch_intersection_result<scalar_type>& result,
        StateDistance&& state_distance)
    {
        if(!policy.enabled ||
           step_norms0.size() <= policy.signature_norm_index ||
           step_norms1.size() <= policy.signature_norm_index ||
           points_->size() < 2)
        {
            return false;
        }

        const uint64_t latest_index = points_->back().point_index;
        const scalar_type step_signature0 = step_norms0[policy.signature_norm_index];
        const scalar_type step_signature1 = step_norms1[policy.signature_norm_index];
        const int point_count = static_cast<int>(points_->size());
        for(int index = 0; index < point_count - 1; ++index)
        {
            const auto& lower = (*points_)[static_cast<std::size_t>(index)];
            const auto& upper = (*points_)[static_cast<std::size_t>(index + 1)];
            const uint64_t candidate_latest_index = std::max(lower.point_index, upper.point_index);
            if(latest_index <= candidate_latest_index + policy.minimum_index_gap ||
               !can_interpolate_between(lower, upper, segment_metadata_available))
            {
                continue;
            }

            scalar_type lambda_lower = scalar_type(0);
            scalar_type lambda_upper = scalar_type(0);
            if(!interval_overlap(
                   step_lambda0,
                   step_lambda1,
                   lower.lambda,
                   upper.lambda,
                   lambda_lower,
                   lambda_upper))
            {
                continue;
            }

            scalar_type old_signature0 = scalar_type(0);
            scalar_type old_signature1 = scalar_type(0);
            if(!get_signature_value(lower, policy.signature_norm_index, old_signature0) ||
               !get_signature_value(upper, policy.signature_norm_index, old_signature1))
            {
                continue;
            }

            const scalar_type signature_scale = std::max<scalar_type>(
                scalar_type(1),
                std::max<scalar_type>(
                    std::max<scalar_type>(abs_value(step_signature0), abs_value(step_signature1)),
                    std::max<scalar_type>(abs_value(old_signature0), abs_value(old_signature1))));
            const scalar_type signature_tolerance = policy.signature_tolerance*signature_scale;
            if(!signature_envelopes_overlap(
                   step_signature0,
                   step_signature1,
                   old_signature0,
                   old_signature1,
                   signature_tolerance))
            {
                continue;
            }

            scalar_type candidate_lambda = scalar_type(0);
            scalar_type signature_distance = scalar_type(0);
            if(!find_signature_candidate_lambda(
                   lambda_lower,
                   lambda_upper,
                   step_lambda0,
                   step_signature0,
                   step_lambda1,
                   step_signature1,
                   lower.lambda,
                   old_signature0,
                   upper.lambda,
                   old_signature1,
                   signature_tolerance,
                   candidate_lambda,
                   signature_distance) ||
               !candidate_has_step_progress(
                   candidate_lambda,
                   step_lambda0,
                   step_lambda1,
                   policy.minimum_step_fraction_from_start))
            {
                continue;
            }

            if(!evaluate_segment(
                   index,
                   index + 1,
                   candidate_lambda,
                   segment_metadata_available,
                   directory,
                   *second_work_))
            {
                continue;
            }

            const scalar_type weight = interpolation_weight(
                candidate_lambda,
                step_lambda0,
                step_lambda1);
            vector_operations_->assign_mul(
                scalar_type(1) - weight,
                step_x0,
                weight,
                step_x1,
                *first_work_);
            const scalar_type distance = static_cast<scalar_type>(
                state_distance(*first_work_, *second_work_));
            const scalar_type state_tolerance = policy.state_tolerance*state_signature_scale(
                candidate_lambda,
                step_lambda0,
                step_signature0,
                step_lambda1,
                step_signature1,
                lower.lambda,
                old_signature0,
                upper.lambda,
                old_signature1);
            if(distance <= state_tolerance)
            {
                vector_operations_->assign(*second_work_, hit_x);
                set_result(
                    result,
                    branch_intersection_detection::verified,
                    candidate_lambda,
                    signature_distance,
                    distance,
                    state_tolerance,
                    curve_number,
                    lower,
                    upper,
                    "self_intersection");
                return true;
            }
        }
        return false;
    }

    static scalar_type abs_value(const scalar_type& value)
    {
        return value < scalar_type(0) ? -value : value;
    }

    static bool same_scalar(const scalar_type& left, const scalar_type& right)
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(abs_value(left), abs_value(right)));
        return abs_value(left - right) <=
            scalar_type(64)*std::numeric_limits<scalar_type>::epsilon()*scale;
    }

    static bool interval_overlap(
        const scalar_type& first0,
        const scalar_type& first1,
        const scalar_type& second0,
        const scalar_type& second1,
        scalar_type& lower,
        scalar_type& upper)
    {
        lower = std::max(std::min(first0, first1), std::min(second0, second1));
        upper = std::min(std::max(first0, first1), std::max(second0, second1));
        return lower <= upper || same_scalar(lower, upper);
    }

    static scalar_type interpolation_weight(
        const scalar_type& lambda,
        const scalar_type& lambda0,
        const scalar_type& lambda1)
    {
        if(same_scalar(lambda0, lambda1))
        {
            return scalar_type(0.5);
        }
        return (lambda - lambda0)/(lambda1 - lambda0);
    }

    static scalar_type interpolate_scalar(
        const scalar_type& lambda,
        const scalar_type& lambda0,
        const scalar_type& value0,
        const scalar_type& lambda1,
        const scalar_type& value1)
    {
        const scalar_type weight = interpolation_weight(lambda, lambda0, lambda1);
        return (scalar_type(1) - weight)*value0 + weight*value1;
    }

    static bool signature_envelopes_overlap(
        const scalar_type& new_signature0,
        const scalar_type& new_signature1,
        const scalar_type& old_signature0,
        const scalar_type& old_signature1,
        const scalar_type& tolerance)
    {
        const scalar_type new_min = std::min(new_signature0, new_signature1);
        const scalar_type new_max = std::max(new_signature0, new_signature1);
        const scalar_type old_min = std::min(old_signature0, old_signature1);
        const scalar_type old_max = std::max(old_signature0, old_signature1);
        return (new_min - tolerance) <= old_max &&
               (old_min - tolerance) <= new_max;
    }

    static bool find_signature_candidate_lambda(
        const scalar_type& lambda_lower,
        const scalar_type& lambda_upper,
        const scalar_type& step_lambda0,
        const scalar_type& step_signature0,
        const scalar_type& step_lambda1,
        const scalar_type& step_signature1,
        const scalar_type& old_lambda0,
        const scalar_type& old_signature0,
        const scalar_type& old_lambda1,
        const scalar_type& old_signature1,
        const scalar_type& tolerance,
        scalar_type& candidate_lambda,
        scalar_type& signature_distance)
    {
        const scalar_type lower_difference =
            interpolate_scalar(
                lambda_lower,
                step_lambda0,
                step_signature0,
                step_lambda1,
                step_signature1) -
            interpolate_scalar(
                lambda_lower,
                old_lambda0,
                old_signature0,
                old_lambda1,
                old_signature1);
        const scalar_type upper_difference =
            interpolate_scalar(
                lambda_upper,
                step_lambda0,
                step_signature0,
                step_lambda1,
                step_signature1) -
            interpolate_scalar(
                lambda_upper,
                old_lambda0,
                old_signature0,
                old_lambda1,
                old_signature1);

        const scalar_type lower_abs = abs_value(lower_difference);
        const scalar_type upper_abs = abs_value(upper_difference);
        if(lower_abs <= tolerance || same_scalar(lambda_lower, lambda_upper))
        {
            candidate_lambda = lambda_lower;
            signature_distance = lower_abs;
            return signature_distance <= tolerance;
        }
        if(upper_abs <= tolerance)
        {
            candidate_lambda = lambda_upper;
            signature_distance = upper_abs;
            return true;
        }
        if(lower_difference*upper_difference > scalar_type(0))
        {
            return false;
        }

        const scalar_type denominator = upper_difference - lower_difference;
        candidate_lambda = same_scalar(denominator, scalar_type(0))
            ? scalar_type(0.5)*(lambda_lower + lambda_upper)
            : lambda_lower - lower_difference*(lambda_upper - lambda_lower)/denominator;
        if(candidate_lambda < lambda_lower || candidate_lambda > lambda_upper)
        {
            return false;
        }
        const scalar_type candidate_difference =
            interpolate_scalar(
                candidate_lambda,
                step_lambda0,
                step_signature0,
                step_lambda1,
                step_signature1) -
            interpolate_scalar(
                candidate_lambda,
                old_lambda0,
                old_signature0,
                old_lambda1,
                old_signature1);
        signature_distance = abs_value(candidate_difference);
        return signature_distance <= tolerance;
    }

    static bool candidate_has_step_progress(
        const scalar_type& candidate_lambda,
        const scalar_type& step_lambda0,
        const scalar_type& step_lambda1,
        const scalar_type& minimum_step_fraction_from_start)
    {
        if(same_scalar(step_lambda0, step_lambda1))
        {
            return false;
        }
        const scalar_type fraction =
            (candidate_lambda - step_lambda0)/(step_lambda1 - step_lambda0);
        return fraction > minimum_step_fraction_from_start &&
               fraction <= scalar_type(1) + minimum_step_fraction_from_start;
    }

private:
    bool is_incomplete_segment(const uint64_t segment_id) const
    {
        return std::find(
                   incomplete_segment_ids_->begin(),
                   incomplete_segment_ids_->end(),
                   segment_id) != incomplete_segment_ids_->end();
    }

    bool can_interpolate_between(
        const Point& lower,
        const Point& upper,
        const bool segment_metadata_available) const
    {
        if(!segment_metadata_available)
        {
            return true;
        }
        // Accepted states remain a valid searchable branch segment even when
        // its terminal endpoint is recoverable. Segment boundaries, rather
        // than completion status, are interpolation barriers.
        return lower.segment_id == upper.segment_id;
    }

    static bool contains_lambda(
        const Point& lower,
        const Point& upper,
        const scalar_type& lambda)
    {
        return (lower.lambda - lambda)*(upper.lambda - lambda) <= scalar_type(0);
    }

    static bool get_signature_value(
        const Point& point,
        const unsigned int signature_index,
        scalar_type& value)
    {
        if(point.vector_norms.size() <= signature_index)
        {
            return false;
        }
        value = point.vector_norms[signature_index];
        return true;
    }

    static scalar_type state_signature_scale(
        const scalar_type& lambda,
        const scalar_type& step_lambda0,
        const scalar_type& step_signature0,
        const scalar_type& step_lambda1,
        const scalar_type& step_signature1,
        const scalar_type& old_lambda0,
        const scalar_type& old_signature0,
        const scalar_type& old_lambda1,
        const scalar_type& old_signature1)
    {
        return std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(
                abs_value(interpolate_scalar(
                    lambda,
                    step_lambda0,
                    step_signature0,
                    step_lambda1,
                    step_signature1)),
                abs_value(interpolate_scalar(
                    lambda,
                    old_lambda0,
                    old_signature0,
                    old_lambda1,
                    old_signature1))));
    }

    bool evaluate_segment(
        const int lower_index,
        const int upper_index,
        const scalar_type& lambda,
        const bool segment_metadata_available,
        const std::string& directory,
        vector_type& output)
    {
        return interpolator_->evaluate_segment_at_lambda(
            *points_,
            lower_index,
            upper_index,
            lambda,
            segment_metadata_available,
            directory,
            output);
    }

    bool evaluate_at_lambda(
        const scalar_type& lambda,
        const bool segment_metadata_available,
        const std::string& directory,
        vector_type& output)
    {
        const int point_count = static_cast<int>(points_->size());
        for(int index = 0; index < point_count - 1; ++index)
        {
            const auto& lower = (*points_)[static_cast<std::size_t>(index)];
            const auto& upper = (*points_)[static_cast<std::size_t>(index + 1)];
            if(!can_interpolate_between(lower, upper, segment_metadata_available))
            {
                continue;
            }
            if(contains_lambda(lower, upper, lambda))
            {
                return evaluate_segment(
                    index,
                    index + 1,
                    lambda,
                    segment_metadata_available,
                    directory,
                    output);
            }
        }
        return false;
    }

    static void set_result(
        branch_intersection_result<scalar_type>& result,
        const branch_intersection_detection detection,
        const scalar_type& lambda,
        const scalar_type& signature_distance,
        const scalar_type& state_distance,
        const scalar_type& state_tolerance,
        const int curve_number,
        const Point& lower,
        const Point& upper,
        const char* reason)
    {
        result.found = true;
        result.detection = detection;
        result.lambda = lambda;
        result.signature_distance = signature_distance;
        result.state_distance = state_distance;
        result.state_tolerance = state_tolerance;
        result.curve_number = curve_number;
        result.segment_id = lower.segment_id;
        result.semicurve_id = lower.semicurve_id;
        result.lower_point_index = lower.point_index;
        result.upper_point_index = upper.point_index;
        result.reason = reason;
    }

    VectorOperations* vector_operations_ = nullptr;
    Interpolator* interpolator_ = nullptr;
    point_container_type* points_ = nullptr;
    const std::vector<uint64_t>* incomplete_segment_ids_ = nullptr;
    vector_type* first_work_ = nullptr;
    vector_type* second_work_ = nullptr;
};

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_INTERSECTION_SEARCH_H__
