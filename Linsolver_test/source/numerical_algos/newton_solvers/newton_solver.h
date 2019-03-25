#ifndef __newton_solver_H__
#define __newton_solver_H__
/*
    Newton solver in general

*/

namespace numerical_algos
{
namespace newton_method
{

template<class vector_operations, class nonlinear_operator, class system_operator, class convergence_strategy>
class newton_solver
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    newton_solver(vector_operations*& vec_ops_, system_operator*& system_op_, convergence_strategy*& conv_strat_):
    vec_ops(vec_ops_),
    system_op(system_op_),
    conv_strat(conv_strat_)
    {
        vec_ops->init_vector(delta_x); vec_ops->start_use_vector(delta_x);
    }
    
    ~newton_solver()
    {
        vec_ops->stop_use_vector(delta_x); vec_ops->free_vector(delta_x); 

    }

    bool solve(nonlinear_operator* nonlin_op, const T_vec& x0, const T& lambda0, T_vec& x)
    {
        int result_status = 1;
        T delta_lambda = T(1);
        vec_ops->assign(x0, x);
        vec_ops->assign_scalar(T(1), delta_x);
        bool converged = false;
        bool finished = false;
        bool linsolver_converged;
        conv_strat->reset_iterations();
        while(!finished)
        {
            linsolver_converged = system_op->solve(nonlin_op, x, lambda0, delta_x);
            if(linsolver_converged)
            {
                finished = conv_strat->check_convergence(nonlin_op, x, lambda0, delta_x, result_status);
            }
            else
            {
                finished = true;
                result_status = 100;
            }

        }
        if(result_status==0)
            converged = true;


        return converged;
    }




private:
    vector_operations* vec_ops;
    system_operator* system_op;
    convergence_strategy* conv_strat;
    T_vec delta_x;


};



}
}

#endif