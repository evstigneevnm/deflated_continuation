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

} // namespace detail
} // namespace nonlinear_operators

#endif
