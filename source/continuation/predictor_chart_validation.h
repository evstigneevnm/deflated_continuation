#ifndef __CONTINUATION_PREDICTOR_CHART_VALIDATION_H__
#define __CONTINUATION_PREDICTOR_CHART_VALIDATION_H__

#include <common/scalar_math.h>

namespace continuation
{

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

    bool needs_repair() const
    {
        return raw_non_positive ||
               charted_non_positive ||
               charted_weak_reject ||
               progress_jump_reject ||
               displacement_reject;
    }
};

template<class T>
predictor_chart_validation_result<T> validate_predictor_chart(
    const T& ds,
    const T& raw_tangent_progress,
    const T& charted_tangent_progress,
    const T& predictor_chart_displacement)
{
    predictor_chart_validation_result<T> result;
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
        abs_charted_progress < T(0.2)*result.progress_scale;
    result.charted_weak_reject =
        !result.charted_non_positive &&
        abs_charted_progress < T(0.02)*result.progress_scale;
    result.progress_jump_warning = result.progress_ratio > T(5) || result.progress_ratio < T(-5);
    result.progress_jump_reject = result.progress_ratio > T(50) || result.progress_ratio < T(-50);
    result.displacement_warning = result.displacement_ratio > T(20);
    result.displacement_reject = result.displacement_ratio > T(100);
    return result;
}

} // namespace continuation

#endif
