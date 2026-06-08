#ifndef __PRECONDITIONER_STAR_SHAPED_H__
#define __PRECONDITIONER_STAR_SHAPED_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator>
class preconditioner_star_shaped
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit preconditioner_star_shaped(NonlinearOperator* nonlin_op_):
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
        nonlin_op->solve_jacobian_system(x);
    }

private:
    NonlinearOperator* nonlin_op;
    mutable const LinearOperator* lin_op = nullptr;
};

} // namespace nonlinear_operators

#endif
