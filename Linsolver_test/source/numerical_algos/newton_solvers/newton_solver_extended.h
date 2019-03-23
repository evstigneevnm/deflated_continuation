#ifndef __NEWTON_SOLVER_EXTENDED_H__
#define __NEWTON_SOLVER_EXTENDED_H__
/*
    Newton solver for extended problem (x,lambda) in general
*/

namespace numerical_algos
{
namespace newton_method_extended
{

template<class vector_operations, class system_operator, class convergence_strategy, class solution_point/*, class nonlinear_operator, class linear_operator*/>
class newton_solver_extended
{
public:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;

    newton_solver_extended(vector_operations*& vec_ops_, system_operator*& system_op_, convergence_strategy*& conv_strat_):
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

    void solve(const T_vec& x0, const T& lambda0, T_vec& x, T& lambda)
    {
        int result_status = 1;
        T delta_lambda = T(1);
        vec_ops->assign(x0, x);
        lambda = lambda0;
        vec_ops->assign_scalar(T(1), delta_x);
        bool finished = false;
        while(!finished)
        {
            system_op->set_linearization_point(x,labda);
            system_op->solve(x, lambda, delta_x, delta_lambda);
            finished = conv_strat->check_convergence(x, lambda, delta_x, delta_lambda, result_status);
        }

    }

    void solve(const solution_point& x0, solution_point& x)
    {
        //to be used for extended system operation with class container for the extended solution.
        //i intend to unpack solution_point and pass to the solve(x0, l0, x, l)
        //...

    }
    


private:
    vector_operations* vec_ops_;
    system_operator* system_op;
    convergence_strategy* conv_strat;
    T_vec delta_x;


};



}
}

#endif