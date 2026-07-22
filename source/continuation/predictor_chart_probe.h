#ifndef __CONTINUATION_PREDICTOR_CHART_PROBE_H__
#define __CONTINUATION_PREDICTOR_CHART_PROBE_H__

#include <continuation/chart_helpers.h>
#include <continuation/predictor_chart_validation.h>

namespace continuation
{

template<class T>
struct predictor_chart_probe_result
{
    T raw_x_progress = T(0);
    T raw_lambda_progress = T(0);
    T raw_tangent_progress = T(0);
    T charted_x_progress = T(0);
    T charted_lambda_progress = T(0);
    T charted_tangent_progress = T(0);
    T chart_displacement = T(0);
    predictor_chart_validation_result<T> validation;
};

template<class VectorOperations, class Vector, class Scalar>
predictor_chart_probe_result<Scalar> evaluate_predictor_chart(
    VectorOperations* vec_ops,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    const Vector& raw_predictor,
    const Scalar& raw_lambda,
    const Vector& charted_predictor,
    const Scalar& charted_lambda,
    Vector& work,
    const Scalar& ds,
    const predictor_chart_policy<Scalar>& policy)
{
    predictor_chart_probe_result<Scalar> result;
    vec_ops->assign_mul(Scalar(1), raw_predictor, Scalar(-1), x0, work);
    result.raw_x_progress = vec_ops->scalar_prod(work, x0_s);
    result.raw_lambda_progress = (raw_lambda - lambda0)*lambda0_s;
    result.raw_tangent_progress = result.raw_x_progress + result.raw_lambda_progress;

    vec_ops->assign_mul(Scalar(1), charted_predictor, Scalar(-1), x0, work);
    result.charted_x_progress = vec_ops->scalar_prod(work, x0_s);
    result.charted_lambda_progress = (charted_lambda - lambda0)*lambda0_s;
    result.charted_tangent_progress = result.charted_x_progress + result.charted_lambda_progress;

    vec_ops->assign_mul(Scalar(1), charted_predictor, Scalar(-1), raw_predictor, work);
    result.chart_displacement = vec_ops->norm_l2(work);
    result.validation = validate_predictor_chart(
        ds,
        result.raw_tangent_progress,
        result.charted_tangent_progress,
        result.chart_displacement,
        policy);
    return result;
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
predictor_chart_probe_result<Scalar> probe_predictor_chart(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    const Scalar& ds,
    Vector& raw_predictor,
    Vector& charted_predictor,
    Vector& work,
    const predictor_chart_policy<Scalar>& policy)
{
    vec_ops->assign(x0, raw_predictor);
    vec_ops->add_mul(ds, x0_s, raw_predictor);
    const Scalar raw_lambda = lambda0 + ds*lambda0_s;
    Scalar charted_lambda = raw_lambda;

    chart::begin_continuation_chart(vec_ops, log, op, x0, lambda0, x0_s, lambda0_s);
    chart::stabilize_predictor_for_continuation(
        vec_ops,
        log,
        op,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        raw_predictor,
        raw_lambda,
        charted_predictor,
        charted_lambda);

    return evaluate_predictor_chart(
        vec_ops,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        raw_predictor,
        raw_lambda,
        charted_predictor,
        charted_lambda,
        work,
        ds,
        policy);
}

} // namespace continuation

#endif
