#ifndef __NONLINEAR_OPERATORS_PROJECTED_OPERATOR_HELPERS_H__
#define __NONLINEAR_OPERATORS_PROJECTED_OPERATOR_HELPERS_H__

namespace nonlinear_operators
{
namespace detail
{

template<class NonlinearOperator, class Vector, class Scalar>
auto set_linearization_point(NonlinearOperator* op, const Vector& x, const Scalar& lambda, int)
    -> decltype(op->set_projected_linearization_point(x, lambda), void())
{
    op->set_projected_linearization_point(x, lambda);
}

template<class NonlinearOperator, class Vector, class Scalar>
void set_linearization_point(NonlinearOperator* op, const Vector& x, const Scalar& lambda, long)
{
    op->set_linearization_point(x, lambda);
}

template<class NonlinearOperator, class Vector, class Scalar>
void set_linearization_point(NonlinearOperator* op, const Vector& x, const Scalar& lambda)
{
    set_linearization_point(op, x, lambda, 0);
}

template<class NonlinearOperator, class Vector, class Scalar>
auto residual(NonlinearOperator* op, const Vector& x, const Scalar& lambda, Vector& out, int)
    -> decltype(op->projected_F(x, lambda, out), void())
{
    op->projected_F(x, lambda, out);
}

template<class NonlinearOperator, class Vector, class Scalar>
void residual(NonlinearOperator* op, const Vector& x, const Scalar& lambda, Vector& out, long)
{
    op->F(x, lambda, out);
}

template<class NonlinearOperator, class Vector, class Scalar>
void residual(NonlinearOperator* op, const Vector& x, const Scalar& lambda, Vector& out)
{
    residual(op, x, lambda, out, 0);
}

template<class NonlinearOperator, class Vector, class Scalar>
auto residual_at_linearization(NonlinearOperator* op, const Vector&, const Scalar&, Vector& out, int)
    -> decltype(op->projected_F_at_linearization(out), void())
{
    op->projected_F_at_linearization(out);
}

template<class NonlinearOperator, class Vector, class Scalar>
void residual_at_linearization(NonlinearOperator* op, const Vector& x, const Scalar& lambda, Vector& out, long)
{
    op->F(x, lambda, out);
}

template<class NonlinearOperator, class Vector, class Scalar>
void residual_at_linearization(NonlinearOperator* op, const Vector& x, const Scalar& lambda, Vector& out)
{
    residual_at_linearization(op, x, lambda, out, 0);
}

template<class NonlinearOperator, class Vector>
auto jacobian_alpha(NonlinearOperator* op, Vector& out, int)
    -> decltype(op->projected_jacobian_alpha(out), void())
{
    op->projected_jacobian_alpha(out);
}

template<class NonlinearOperator, class Vector>
void jacobian_alpha(NonlinearOperator* op, Vector& out, long)
{
    op->jacobian_alpha(out);
}

template<class NonlinearOperator, class Vector>
void jacobian_alpha(NonlinearOperator* op, Vector& out)
{
    jacobian_alpha(op, out, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto project_current_tangent(VectorOperations*, NonlinearOperator* op, const Vector& source, Vector& destination, int)
    -> decltype(op->project_current_tangent(source, destination), void())
{
    op->project_current_tangent(source, destination);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_current_tangent(VectorOperations* vec_ops, NonlinearOperator*, const Vector& source, Vector& destination, long)
{
    if(&source != &destination)
    {
        vec_ops->assign(source, destination);
    }
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_current_tangent(VectorOperations* vec_ops, NonlinearOperator* op, const Vector& source, Vector& destination)
{
    project_current_tangent(vec_ops, op, source, destination, 0);
}

template<class NonlinearOperator, class Vector>
auto project_state(NonlinearOperator* op, Vector& x, int) -> decltype(op->project(x), void())
{
    op->project(x);
}

template<class NonlinearOperator, class Vector>
void project_state(NonlinearOperator*, Vector&, long)
{
}

template<class NonlinearOperator, class Vector>
void project_state(NonlinearOperator* op, Vector& x)
{
    project_state(op, x, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto project_state_relative_to(VectorOperations*, NonlinearOperator* op, const Vector& reference, Vector& x, int)
    -> decltype(op->project_relative_to(reference, x), void())
{
    op->project_relative_to(reference, x);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_state_relative_to(VectorOperations*, NonlinearOperator* op, const Vector&, Vector& x, long)
{
    project_state(op, x);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void project_state_relative_to(VectorOperations* vec_ops, NonlinearOperator* op, const Vector& reference, Vector& x)
{
    project_state_relative_to(vec_ops, op, reference, x, 0);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
auto stabilize_for_arclength(VectorOperations*, NonlinearOperator* op, const Vector& reference, const Vector& source, Vector& destination, int)
    -> decltype(op->stabilize_for_arclength(reference, source, destination), void())
{
    op->stabilize_for_arclength(reference, source, destination);
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_for_arclength(VectorOperations* vec_ops, NonlinearOperator*, const Vector&, const Vector& source, Vector& destination, long)
{
    if(&source != &destination)
    {
        vec_ops->assign(source, destination);
    }
}

template<class VectorOperations, class NonlinearOperator, class Vector>
void stabilize_for_arclength(VectorOperations* vec_ops, NonlinearOperator* op, const Vector& reference, const Vector& source, Vector& destination)
{
    stabilize_for_arclength(vec_ops, op, reference, source, destination, 0);
}

template<class Log, class NonlinearOperator>
auto log_projection_diagnostics(Log* log, NonlinearOperator* op, const char* context, int)
    -> decltype(op->log_projection_diagnostics(log, context), void())
{
    op->log_projection_diagnostics(log, context);
}

template<class Log, class NonlinearOperator>
void log_projection_diagnostics(Log*, NonlinearOperator*, const char*, long)
{
}

template<class Log, class NonlinearOperator>
void log_projection_diagnostics(Log* log, NonlinearOperator* op, const char* context)
{
    log_projection_diagnostics(log, op, context, 0);
}

} // namespace detail
} // namespace nonlinear_operators

#endif
