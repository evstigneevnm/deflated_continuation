#ifndef __SYSTEM_OPERATOR_H__
#define __SYSTEM_OPERATOR_H__

namespace nonlinear_operators
{

template<class vector_operations, class nonlinear_operator, class linear_operator, class linear_solver>
class system_operator
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    

    system_operator(vector_operations*& vec_ops_, linear_operator*& lin_op_, linear_solver*& lin_solver_):
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

    bool solve(nonlinear_operator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x)
    {
        bool flag_lin_solver;
        
        nonlin_op->set_linearization_point(x, lambda);
        nonlin_op->F(x, lambda, b); // 
        vec_ops->add_mul_scalar(T(0), T(-1), b); //b=-F(x,lambda)
        flag_lin_solver = lin_solver->solve(*lin_op, b, d_x);
        return flag_lin_solver;
    }
private:
    vector_operations* vec_ops;
    linear_operator* lin_op;
    linear_solver* lin_solver;
    T_vec b;
    T_vec f;
    T_vec c;
    T beta;

};

}

#endif