#ifndef __CONTINUATION_PREDICTOR_CHART_DIAGNOSTICS_H__
#define __CONTINUATION_PREDICTOR_CHART_DIAGNOSTICS_H__

#include <continuation/predictor_chart_validation.h>

namespace continuation
{

template<class Log, class T>
void log_predictor_chart_validation_warnings(
    Log* log,
    const T& ds,
    const T& raw_progress,
    const T& charted_progress,
    const T& chart_displacement,
    const predictor_chart_validation_result<T>& validation)
{
    if(validation.non_finite)
    {
        log->warning_f(
            "continuation::predict: validation warning: predictor or chart diagnostics contain a nonfinite value: raw = %le, charted = %le, chart displacement = %le, ds = %le.",
            double(raw_progress), double(charted_progress), double(chart_displacement), double(ds));
    }
    if(validation.raw_non_positive)
    {
        log->warning_f(
            "continuation::predict: validation warning: raw predictor progress is non-positive: raw = %le, ds = %le.",
            double(raw_progress), double(ds));
    }
    if(validation.charted_non_positive)
    {
        log->warning_f(
            "continuation::predict: validation warning: charted predictor progress is non-positive: raw = %le, charted = %le, chart displacement = %le.",
            double(raw_progress), double(charted_progress), double(chart_displacement));
    }
    else if(validation.charted_weak_warning)
    {
        log->warning_f(
            "continuation::predict: validation warning: charted predictor progress is weak: raw = %le, charted = %le, charted/raw-scale = %le.",
            double(raw_progress), double(charted_progress), double(validation.progress_ratio));
    }
    if(validation.progress_jump_warning)
    {
        log->warning_f(
            "continuation::predict: validation warning: charted predictor progress changed too much: raw = %le, charted = %le, charted/raw-scale = %le.",
            double(raw_progress), double(charted_progress), double(validation.progress_ratio));
    }
    if(validation.displacement_warning)
    {
        const T effective_ds = validation.displacement_ratio > T(0)
            ? chart_displacement/validation.displacement_ratio
            : T(0);
        log->warning_f(
            "continuation::predict: validation warning: chart displacement is large relative to ds: ||x1 - x_p|| = %le, ds = %le, ratio = %le.",
            double(chart_displacement), double(effective_ds), double(validation.displacement_ratio));
    }
}

} // namespace continuation

#endif
