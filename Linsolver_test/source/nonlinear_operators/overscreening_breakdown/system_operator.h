#ifndef __SYSTEM_OPERATOR_H__
#define __SYSTEM_OPERATOR_H__


/**
*   System operator class used to solve the linear system
*   Executed in the Newton's method
*
*
*/


namespace nonlinear_operators
{


template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver>
class system_operator
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    

    system_operator(VectorOperations* vec_ops_p, LinearOperator* lin_op_p, LinearSolver* lin_solver_p):
    vec_ops_(vec_ops_p),
    lin_op_(lin_op_p),
    lin_solver_(lin_solver_p)
    {
        vec_ops_->init_vector(b); vec_ops_->start_use_vector(b);
    }
    ~system_operator()
    {
        vec_ops_->stop_use_vector(b); vec_ops_->free_vector(b);
    }

    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x)
    {
        bool flag_lin_solver_;
        
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->F(x, lambda, b); // 
        vec_ops_->add_mul_scalar(T(0), T(-1), b); //b=-F(x,lambda)
        flag_lin_solver_ = lin_solver_->solve(*lin_op_, b, d_x);
        return flag_lin_solver_;
    }
private:
    VectorOperations* vec_ops_;
    LinearOperator* lin_op_;
    LinearSolver* lin_solver_;
    T_vec b;
    T_vec f;
    T_vec c;
    T beta;

};


}

#endif