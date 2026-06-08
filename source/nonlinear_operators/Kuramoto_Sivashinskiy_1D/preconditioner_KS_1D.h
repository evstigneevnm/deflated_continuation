#ifndef __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__
#define __PRECONDITIONER_KURAMOTO_SIVASHINSKIY_1D_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_KS_1D
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit preconditioner_KS_1D(NonlinearOperator* nonlin_op_):
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
        nonlin_op->preconditioner_jacobian_u(x);
    }

private:
    NonlinearOperator* nonlin_op;
    mutable const LinearOperator* lin_op = nullptr;
};

} // namespace nonlinear_operators

#endif
