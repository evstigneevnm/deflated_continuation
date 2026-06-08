#ifndef __PROJECTED_LINEAR_OPERATOR_KURAMOTO_SIVASHINSKIY_1D_H__
#define __PROJECTED_LINEAR_OPERATOR_KURAMOTO_SIVASHINSKIY_1D_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator>
class projected_linear_operator_KS_1D
{
public:
    using T_vec = typename VectorOperations::vector_type;

    explicit projected_linear_operator_KS_1D(NonlinearOperator* nonlin_op_):
        nonlin_op(nonlin_op_)
    {
    }

    void apply(const T_vec& x, T_vec& f) const
    {
        nonlin_op->projected_jacobian_u(x, f);
    }

private:
    NonlinearOperator* nonlin_op;
};

} // namespace nonlinear_operators

#endif
