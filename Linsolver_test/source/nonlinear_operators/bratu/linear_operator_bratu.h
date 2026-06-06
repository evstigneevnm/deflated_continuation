#ifndef __LINEAR_OPERATOR_BRATU_H__
#define __LINEAR_OPERATOR_BRATU_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator>
class linear_operator_bratu
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit linear_operator_bratu(NonlinearOperator* nonlin_op_):
        nonlin_op(nonlin_op_)
    {
    }

    void apply(const T_vec& x, T_vec& f) const
    {
        nonlin_op->jacobian_u(x, f);
    }

private:
    NonlinearOperator* nonlin_op;
};

} // namespace nonlinear_operators

#endif
