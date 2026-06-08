#ifndef __BRATU_SYSTEM_OPERATOR_H__
#define __BRATU_SYSTEM_OPERATOR_H__

namespace nonlinear_operators
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver>
class system_operator
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    system_operator(VectorOperations* vec_ops_, LinearOperator* lin_op_, LinearSolver* lin_solver_):
        vec_ops(vec_ops_),
        lin_op(lin_op_),
        lin_solver(lin_solver_)
    {
        vec_ops->init_vector(b);
        vec_ops->start_use_vector(b);
    }

    ~system_operator()
    {
        vec_ops->stop_use_vector(b);
        vec_ops->free_vector(b);
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x)
    {
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->F(x, lambda, b);
        vec_ops->add_mul_scalar(T(0), T(-1), b);
        return lin_solver->solve(*lin_op, b, d_x);
    }

private:
    VectorOperations* vec_ops;
    LinearOperator* lin_op;
    LinearSolver* lin_solver;
    T_vec b;
};

} // namespace nonlinear_operators

#endif
