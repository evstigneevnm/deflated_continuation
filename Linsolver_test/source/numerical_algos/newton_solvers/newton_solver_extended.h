#ifndef __NEWTON_SOLVER_EXTENDED_H__
#define __NEWTON_SOLVER_EXTENDED_H__
/*
    Newton solver for extended problem (x,lambda) in general
*/

#include <string>
#include <stdexcept>
#include "../detail/str_source_helper.h"


namespace numerical_algos
{
namespace newton_method_extended
{

template<class VectorOperations, class NonlinearOperator, class SystemOperator, class ConvergenceStrategy, class SolutionPoint>
class newton_solver_extended
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    newton_solver_extended(VectorOperations* vec_ops_, SystemOperator* system_op_, ConvergenceStrategy* conv_strat_):
    vec_ops(vec_ops_),
    system_op(system_op_),
    conv_strat(conv_strat_)
    {
        vec_ops->init_vector(delta_x); vec_ops->start_use_vector(delta_x);
    }
    
    ~newton_solver_extended()
    {
        vec_ops->stop_use_vector(delta_x); vec_ops->free_vector(delta_x); 

    }

    //inplace
    bool solve(NonlinearOperator*& nonlin_op, T_vec& x, T& lambda)
    {
        int result_status = 1;
        T delta_lambda = T(0.0);
        vec_ops->assign_scalar(T(0.0), delta_x);
        bool converged = false;
        bool finished = false;
        bool linsolver_converged = false;
        conv_strat->reset_iterations(); //reset iteration count, newton wight and iteration history
        finished = conv_strat->check_convergence(nonlin_op, x, lambda, delta_x, delta_lambda, result_status); //check that the supplied initial guess is not a solution
        while(!finished)
        {
            //reset iterational vectors??!
            delta_lambda = T(0.0);
            vec_ops->assign_scalar(T(0.0), delta_x);     

            linsolver_converged = system_op->solve(nonlin_op, x, lambda, delta_x, delta_lambda);

            finished = conv_strat->check_convergence(nonlin_op, x, lambda, delta_x, delta_lambda, result_status);

        }
        if(result_status==0)
        {
            converged = true;
        }
        if( (result_status == 2)||(result_status == 3) ) //inf or nan
        {
            throw std::runtime_error(std::string("newton_method_extended: " __FILE__ " " __STR(__LINE__) " invalid number returned from update.") );            
        }

        return converged;
    }

    bool solve(NonlinearOperator*& nonlin_op, const T_vec& x0, const T& lambda0, T_vec& x, T& lambda)
    {
        vec_ops->assign(x0, x);
        lambda = lambda0;
        bool converged = false;
        converged = solve(nonlin_op, x, lambda);
        if(!converged)
        {
            vec_ops->assign(x0, x);
            lambda = lambda0;
        }
        return converged;
    }   

    ConvergenceStrategy* get_convergence_strategy_handle()
    {
        return conv_strat;
    }

private:
    VectorOperations* vec_ops;
    SystemOperator* system_op;
    ConvergenceStrategy* conv_strat;
    T_vec delta_x;


};



}
}

#endif