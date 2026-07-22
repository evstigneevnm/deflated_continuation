#ifndef __CONTINUATION_CHART_HELPERS_H__
#define __CONTINUATION_CHART_HELPERS_H__

#include <symmetry/continuation/geometry_traits.h>

namespace continuation
{
namespace chart
{

template<class NonlinearOperator, class Vector, class Scalar>
using has_isotropy_transition =
    symmetry::continuation::geometry::has_isotropy_transition<
        NonlinearOperator,
        Vector,
        Scalar>;

template<class VectorOperations, class Log, class NonlinearOperator, class Vector>
auto prepare_continuation_seed_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    const Vector& source,
    Vector& destination,
    int)
    -> decltype(op->prepare_continuation_seed(source, destination), void())
{
    op->prepare_continuation_seed(source, destination);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector>
void prepare_continuation_seed_impl(
    VectorOperations* vec_ops,
    Log*,
    NonlinearOperator*,
    const Vector& source,
    Vector& destination,
    long)
{
    if(&source != &destination)
    {
        vec_ops->assign(source, destination);
    }
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector>
void prepare_continuation_seed(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& source,
    Vector& destination)
{
    prepare_continuation_seed_impl(vec_ops, log, op, source, destination, 0);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
auto accept_continuation_step_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    Vector& state,
    const Scalar& lambda,
    Vector& tangent,
    Scalar& tangent_lambda,
    int)
    -> decltype(op->accept_continuation_step(state, lambda, tangent, tangent_lambda), void())
{
    op->accept_continuation_step(state, lambda, tangent, tangent_lambda);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void accept_continuation_step_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator*,
    Vector&,
    const Scalar&,
    Vector&,
    Scalar&,
    long)
{
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void accept_continuation_step(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    Vector& state,
    const Scalar& lambda,
    Vector& tangent,
    Scalar& tangent_lambda)
{
    accept_continuation_step_impl(
        vec_ops,
        log,
        op,
        state,
        lambda,
        tangent,
        tangent_lambda,
        0);
}

template<class NonlinearOperator, class Vector, class Scalar>
using has_continuation_chart =
    symmetry::continuation::geometry::has_continuation_chart<
        NonlinearOperator,
        Vector,
        Scalar>;

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
auto begin_continuation_chart_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    int)
    -> decltype(op->begin_continuation_chart(x0, lambda0, x0_s, lambda0_s), void())
{
    op->begin_continuation_chart(x0, lambda0, x0_s, lambda0_s);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void begin_continuation_chart_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator*,
    const Vector&,
    const Scalar&,
    const Vector&,
    const Scalar&,
    long)
{
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void begin_continuation_chart(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s)
{
    begin_continuation_chart_impl(vec_ops, log, op, x0, lambda0, x0_s, lambda0_s, 0);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
auto restore_continuation_chart_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    int)
    -> decltype(op->restore_continuation_chart(x0, lambda0, x0_s, lambda0_s), void())
{
    op->restore_continuation_chart(x0, lambda0, x0_s, lambda0_s);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void restore_continuation_chart_impl(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    long)
{
    begin_continuation_chart(vec_ops, log, op, x0, lambda0, x0_s, lambda0_s);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void restore_continuation_chart(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s)
{
    restore_continuation_chart_impl(vec_ops, log, op, x0, lambda0, x0_s, lambda0_s, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto project_relative_to_impl(
    VectorOperations*,
    NonlinearOperator* op,
    const Vector& reference,
    Vector& x,
    int)
    -> decltype(op->project_relative_to(reference, x), void())
{
    op->project_relative_to(reference, x);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto project_relative_to_impl(
    VectorOperations*,
    NonlinearOperator* op,
    const Vector&,
    Vector& x,
    long)
    -> decltype(op->project(x), void())
{
    op->project(x);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_relative_to_impl(
    VectorOperations*,
    NonlinearOperator*,
    const Vector&,
    Vector&,
    ...)
{
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_relative_to(
    VectorOperations* vec_ops,
    NonlinearOperator* op,
    const Vector& reference,
    Vector& x)
{
    project_relative_to_impl(vec_ops, op, reference, x, 0);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
auto stabilize_predictor_for_continuation_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    const Vector& x_predictor,
    const Scalar& lambda_predictor,
    Vector& x_trial,
    Scalar& lambda_trial,
    int)
    -> decltype(
        op->stabilize_predictor_for_continuation(
            x0,
            lambda0,
            x0_s,
            lambda0_s,
            x_predictor,
            lambda_predictor,
            x_trial,
            lambda_trial),
        void())
{
    op->stabilize_predictor_for_continuation(
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        x_predictor,
        lambda_predictor,
        x_trial,
        lambda_trial);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void stabilize_predictor_for_continuation_impl(
    VectorOperations* vec_ops,
    Log*,
    NonlinearOperator*,
    const Vector&,
    const Scalar&,
    const Vector&,
    const Scalar&,
    const Vector& x_predictor,
    const Scalar& lambda_predictor,
    Vector& x_trial,
    Scalar& lambda_trial,
    long)
{
    if(&x_predictor != &x_trial)
    {
        vec_ops->assign(x_predictor, x_trial);
    }
    lambda_trial = lambda_predictor;
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void stabilize_predictor_for_continuation(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& x0,
    const Scalar& lambda0,
    const Vector& x0_s,
    const Scalar& lambda0_s,
    const Vector& x_predictor,
    const Scalar& lambda_predictor,
    Vector& x_trial,
    Scalar& lambda_trial)
{
    stabilize_predictor_for_continuation_impl(
        vec_ops,
        log,
        op,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        x_predictor,
        lambda_predictor,
        x_trial,
        lambda_trial,
        0);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
auto stabilize_corrector_trial_impl(
    VectorOperations*,
    Log*,
    NonlinearOperator* op,
    const Vector& reference,
    const Scalar& reference_lambda,
    Vector& trial,
    Scalar& trial_lambda,
    int)
    -> decltype(op->stabilize_corrector_trial(reference, reference_lambda, trial, trial_lambda), void())
{
    op->stabilize_corrector_trial(reference, reference_lambda, trial, trial_lambda);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void stabilize_corrector_trial_impl(
    VectorOperations* vec_ops,
    Log*,
    NonlinearOperator* op,
    const Vector& reference,
    const Scalar&,
    Vector& trial,
    Scalar&,
    long)
{
    project_relative_to(vec_ops, op, reference, trial);
}

template<class VectorOperations, class Log, class NonlinearOperator, class Vector, class Scalar>
void stabilize_corrector_trial(
    VectorOperations* vec_ops,
    Log* log,
    NonlinearOperator* op,
    const Vector& reference,
    const Scalar& reference_lambda,
    Vector& trial,
    Scalar& trial_lambda)
{
    stabilize_corrector_trial_impl(vec_ops, log, op, reference, reference_lambda, trial, trial_lambda, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto stabilize_for_arclength_impl(
    VectorOperations*,
    NonlinearOperator* op,
    const Vector& reference,
    const Vector& source,
    Vector& destination,
    int)
    -> decltype(op->stabilize_for_arclength(reference, source, destination), void())
{
    op->stabilize_for_arclength(reference, source, destination);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_for_arclength_impl(
    VectorOperations* vec_ops,
    NonlinearOperator*,
    const Vector&,
    const Vector& source,
    Vector& destination,
    long)
{
    if(&source != &destination)
    {
        vec_ops->assign(source, destination);
    }
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_for_arclength(
    VectorOperations* vec_ops,
    NonlinearOperator* op,
    const Vector& reference,
    const Vector& source,
    Vector& destination)
{
    stabilize_for_arclength_impl(vec_ops, op, reference, source, destination, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto stabilize_tangent_for_arclength_impl(
    VectorOperations*,
    NonlinearOperator* op,
    const Vector& reference,
    const Vector& tangent,
    Vector& destination,
    int)
    -> decltype(op->stabilize_tangent_for_arclength(reference, tangent, destination), void())
{
    op->stabilize_tangent_for_arclength(reference, tangent, destination);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_tangent_for_arclength_impl(
    VectorOperations* vec_ops,
    NonlinearOperator*,
    const Vector&,
    const Vector& tangent,
    Vector& destination,
    long)
{
    if(&tangent != &destination)
    {
        vec_ops->assign(tangent, destination);
    }
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_tangent_for_arclength(
    VectorOperations* vec_ops,
    NonlinearOperator* op,
    const Vector& reference,
    const Vector& tangent,
    Vector& destination)
{
    stabilize_tangent_for_arclength_impl(vec_ops, op, reference, tangent, destination, 0);
}

template<class NonlinearOperator, class Vector, class Scalar>
auto detect_isotropy_transition_impl(
    NonlinearOperator* op,
    const Vector& previous,
    const Vector& candidate,
    const symmetry::continuation::isotropy_transition_policy<Scalar>& policy,
    int)
    -> decltype(op->detect_continuation_isotropy_transition(previous, candidate, policy))
{
    return op->detect_continuation_isotropy_transition(previous, candidate, policy);
}

template<class NonlinearOperator, class Vector, class Scalar>
symmetry::continuation::isotropy_transition_result<Scalar>
detect_isotropy_transition_impl(
    NonlinearOperator*,
    const Vector&,
    const Vector&,
    const symmetry::continuation::isotropy_transition_policy<Scalar>&,
    long)
{
    return {};
}

template<class NonlinearOperator, class Vector, class Scalar>
symmetry::continuation::isotropy_transition_result<Scalar>
detect_isotropy_transition(
    NonlinearOperator* op,
    const Vector& previous,
    const Vector& candidate,
    const symmetry::continuation::isotropy_transition_policy<Scalar>& policy)
{
    return detect_isotropy_transition_impl(op, previous, candidate, policy, 0);
}

template<class Log, class NonlinearOperator>
auto log_continuation_chart_impl(
    Log* log,
    NonlinearOperator* op,
    const char* context,
    int)
    -> decltype(op->log_continuation_chart(log, context), void())
{
    op->log_continuation_chart(log, context);
}

template<class Log, class NonlinearOperator>
auto log_continuation_chart_impl(
    Log* log,
    NonlinearOperator* op,
    const char* context,
    long)
    -> decltype(op->log_projection_diagnostics(log, context), void())
{
    op->log_projection_diagnostics(log, context);
}

template<class Log, class NonlinearOperator>
void log_continuation_chart_impl(
    Log*,
    NonlinearOperator*,
    const char*,
    ...)
{
}

template<class Log, class NonlinearOperator>
void log_continuation_chart(Log* log, NonlinearOperator* op, const char* context)
{
    log_continuation_chart_impl(log, op, context, 0);
}

} // namespace chart
} // namespace continuation

#endif
