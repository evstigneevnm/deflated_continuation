#ifndef __CONTINUATION_PREDICTOR_CHART_VALIDATION_H__
#define __CONTINUATION_PREDICTOR_CHART_VALIDATION_H__

#include <stdexcept>

#include <common/scalar_math.h>

namespace continuation
{

enum class predictor_chart_decision
{
    accept,
    accept_with_warning,
    reject_tangent,
    reject_chart
};

inline const char* predictor_chart_decision_name(const predictor_chart_decision decision)
{
    switch(decision)
    {
        case predictor_chart_decision::accept:
            return "accept";
        case predictor_chart_decision::accept_with_warning:
            return "accept_with_warning";
        case predictor_chart_decision::reject_tangent:
            return "reject_tangent";
        case predictor_chart_decision::reject_chart:
            return "reject_chart";
    }
    return "unknown";
}

template<class T>
struct predictor_chart_policy
{
    bool enabled = true;
    bool enforce = true;
    T weak_progress_warning_ratio = T(0.2);
    T minimum_progress_ratio = T(0.02);
    T progress_jump_warning_ratio = T(5);
    T maximum_progress_ratio = T(50);
    T displacement_warning_ratio = T(20);
    T maximum_displacement_ratio = T(100);
    unsigned int maximum_retries = 4;
    T step_reduction_factor = T(0.2);
};

template<class T>
void validate_predictor_chart_policy(const predictor_chart_policy<T>& policy)
{
    const bool finite =
        common::scalar_math::isfinite(policy.weak_progress_warning_ratio) &&
        common::scalar_math::isfinite(policy.minimum_progress_ratio) &&
        common::scalar_math::isfinite(policy.progress_jump_warning_ratio) &&
        common::scalar_math::isfinite(policy.maximum_progress_ratio) &&
        common::scalar_math::isfinite(policy.displacement_warning_ratio) &&
        common::scalar_math::isfinite(policy.maximum_displacement_ratio) &&
        common::scalar_math::isfinite(policy.step_reduction_factor);
    if(!finite)
    {
        throw std::invalid_argument("predictor chart policy thresholds must be finite");
    }
    if(policy.minimum_progress_ratio < T(0) ||
       policy.weak_progress_warning_ratio < policy.minimum_progress_ratio)
    {
        throw std::invalid_argument(
            "predictor chart policy requires 0 <= minimum_progress_ratio <= weak_progress_warning_ratio");
    }
    if(policy.progress_jump_warning_ratio < T(0) ||
       policy.maximum_progress_ratio < policy.progress_jump_warning_ratio)
    {
        throw std::invalid_argument(
            "predictor chart policy requires 0 <= progress_jump_warning_ratio <= maximum_progress_ratio");
    }
    if(policy.displacement_warning_ratio < T(0) ||
       policy.maximum_displacement_ratio < policy.displacement_warning_ratio)
    {
        throw std::invalid_argument(
            "predictor chart policy requires 0 <= displacement_warning_ratio <= maximum_displacement_ratio");
    }
    if(policy.step_reduction_factor <= T(0) || policy.step_reduction_factor >= T(1))
    {
        throw std::invalid_argument("predictor chart policy step_reduction_factor must be in (0,1)");
    }
}

template<class T>
struct predictor_chart_validation_result
{
    T progress_scale = T(0);
    T progress_ratio = T(0);
    T displacement_ratio = T(0);
    bool raw_non_positive = false;
    bool charted_non_positive = false;
    bool charted_weak_warning = false;
    bool charted_weak_reject = false;
    bool progress_jump_warning = false;
    bool progress_jump_reject = false;
    bool displacement_warning = false;
    bool displacement_reject = false;
    bool non_finite = false;
    predictor_chart_decision decision = predictor_chart_decision::accept;

    bool needs_repair() const
    {
        return non_finite ||
               raw_non_positive ||
               charted_non_positive ||
               charted_weak_reject ||
               progress_jump_reject ||
               displacement_reject;
    }

    bool rejected() const
    {
        return decision == predictor_chart_decision::reject_tangent ||
               decision == predictor_chart_decision::reject_chart;
    }
};

template<class T>
predictor_chart_validation_result<T> validate_predictor_chart(
    const T& ds,
    const T& raw_tangent_progress,
    const T& charted_tangent_progress,
    const T& predictor_chart_displacement,
    const predictor_chart_policy<T>& policy = predictor_chart_policy<T>())
{
    predictor_chart_validation_result<T> result;
    result.non_finite =
        !common::scalar_math::isfinite(ds) ||
        !common::scalar_math::isfinite(raw_tangent_progress) ||
        !common::scalar_math::isfinite(charted_tangent_progress) ||
        !common::scalar_math::isfinite(predictor_chart_displacement);
    const T abs_raw_progress = common::scalar_math::abs(raw_tangent_progress);
    const T abs_charted_progress = common::scalar_math::abs(charted_tangent_progress);
    result.progress_scale = abs_raw_progress > T(0) ? abs_raw_progress : common::scalar_math::abs(ds);
    result.progress_ratio = result.progress_scale > T(0) ? charted_tangent_progress/result.progress_scale : T(0);
    result.displacement_ratio =
        common::scalar_math::abs(ds) > T(0) ? predictor_chart_displacement/common::scalar_math::abs(ds) : T(0);

    result.raw_non_positive = raw_tangent_progress <= T(0);
    result.charted_non_positive = charted_tangent_progress <= T(0);
    result.charted_weak_warning =
        !result.charted_non_positive &&
        abs_charted_progress < policy.weak_progress_warning_ratio*result.progress_scale;
    result.charted_weak_reject =
        !result.charted_non_positive &&
        abs_charted_progress < policy.minimum_progress_ratio*result.progress_scale;
    result.progress_jump_warning =
        common::scalar_math::abs(result.progress_ratio) > policy.progress_jump_warning_ratio;
    result.progress_jump_reject =
        common::scalar_math::abs(result.progress_ratio) > policy.maximum_progress_ratio;
    result.displacement_warning = result.displacement_ratio > policy.displacement_warning_ratio;
    result.displacement_reject = result.displacement_ratio > policy.maximum_displacement_ratio;

    if(!policy.enabled)
    {
        result.decision = predictor_chart_decision::accept;
    }
    else if(result.non_finite || result.raw_non_positive)
    {
        result.decision = policy.enforce ?
            predictor_chart_decision::reject_tangent :
            predictor_chart_decision::accept_with_warning;
    }
    else if(result.charted_non_positive ||
            result.charted_weak_reject ||
            result.progress_jump_reject ||
            result.displacement_reject)
    {
        result.decision = policy.enforce ?
            predictor_chart_decision::reject_chart :
            predictor_chart_decision::accept_with_warning;
    }
    else if(result.charted_weak_warning ||
            result.progress_jump_warning ||
            result.displacement_warning)
    {
        result.decision = predictor_chart_decision::accept_with_warning;
    }
    return result;
}

} // namespace continuation

#endif
