#ifndef __SYSTEM_OPERATOR_TEMPORAL_H__
#define __SYSTEM_OPERATOR_TEMPORAL_H__

#include <utility>
/**
*   System operator class used to solve the linear system for implicit methods
*   Executed in the Newton's method
*
*
*/

namespace time_steppers
{

namespace nonlinear_operators
{


template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver>
class system_operator
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    

    system_operator(VectorOperations* vec_ops_, LinearOperator* lin_op_, LinearSolver* lin_solver_):
    vec_ops(vec_ops_),
    lin_op(lin_op_),
    lin_solver(lin_solver_)
    {
        vec_ops->init_vector(b); vec_ops->start_use_vector(b);
    }
    ~system_operator()
    {
        vec_ops->stop_use_vector(b); vec_ops->free_vector(b);
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x)
    {
        bool flag_lin_solver;
        
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->F(x, lambda, b); // 
        // vec_ops->add_mul_scalar(T(0), T(-1), b); //b=-F(x,lambda)
        flag_lin_solver = lin_solver->solve(*lin_op, b, d_x);
        return flag_lin_solver;
    }
private:
    VectorOperations* vec_ops;
    LinearOperator* lin_op;
    LinearSolver* lin_solver;
    T_vec b;
    T_vec f;
    T_vec c;
    T beta;
    std::pair<T,T> ab_;

};


}
}
#endif