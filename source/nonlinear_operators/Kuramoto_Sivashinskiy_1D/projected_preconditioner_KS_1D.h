#ifndef __PROJECTED_PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__
#define __PROJECTED_PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator>
class projected_preconditioner_KS_1D
{
public:
    using T_vec = typename VectorOperations::vector_type;

    explicit projected_preconditioner_KS_1D(NonlinearOperator* nonlin_op_):
        nonlin_op(nonlin_op_)
    {
    }

    void set_operator(const LinearOperator* op_) const
    {
        lin_op = op_;
        (void)lin_op;
    }

    void apply(T_vec& x) const
    {
        nonlin_op->project_current_tangent(x, x);
        nonlin_op->preconditioner_jacobian_u(x);
        nonlin_op->project_current_tangent(x, x);
    }

private:
    NonlinearOperator* nonlin_op;
    mutable const LinearOperator* lin_op = nullptr;
};

} // namespace nonlinear_operators

#endif
