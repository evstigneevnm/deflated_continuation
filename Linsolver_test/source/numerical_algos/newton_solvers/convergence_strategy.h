#ifndef __CONVERGENCE_STRATEGY_H__
#define __CONVERGENCE_STRATEGY_H__
/*
converhence rules for Newton iterator
*/
#include <cmath>
#include <utils/logged_obj_base.h>

namespace numerical_algos
{
namespace newton_method_extended
{


template<class vector_operations, class nonlinear_operator, class logging>
class convergence_strategy
{
private:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    typedef utils::logged_obj_base<logging> logged_obj_t;

public:    
    convergence_strategy(vector_operations*& vec_ops_, nonlinear_operator*& nonlin_op_, logging*& log_, T tolerance_, unsigned int maximum_iterations_, T newton_wight_, bool verbose_):
    vec_ops(vec_ops_),
    nonlin_op(nonlin_op_),
    log_(log),
    iterations(0),
    tolerance(tolerance_),
    maximum_iterations(maximum_iterations_),
    newton_wight(newton_wight_),
    verbose(verbose_)
    {
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(Fx); vec_ops->start_use_vector(Fx);
    }
    ~convergence_strategy()
    {
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    }


    bool check_convergence(T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, int& result_status)
    {
        bool finish = false;
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm(Fx);
        //update solution
        vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        T lambda1 = lambda + newton_wight*delta_lambda;
        nonlin_op->F(x1, lambda1, Fx);
        T normFx1 = vec_ops->norm(Fx);
        
        iterations++;
        logged_obj_t::info_f("iteration %i, previous residual %le, current residual %le",iterations, (double)normFx, (double)normFx1);

        if(std::isnan(normFx))
        {
            logged_obj_t::info("Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isnan(normFx1))
        {
            logged_obj_t::info("Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }else if(std::isinf(normFx))
        {
            logged_obj_t::info("Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }else if(std::isinf(normFx1))
        {
            logged_obj_t::info("Newton update caused inf.");
            finish = true;
            result_status = 2;            
        }
        if(normFx1<tolerance)
        {
            logged_obj_t::info("Newton converged with %le.", (double)normFx1);
            result_status = 0;
            finish = true;
        }
        else if(iterations>=maximum_iterations)
        {
            logged_obj_t::info("Newton max iterations (%i) reached.", iterations);
            result_status = 1;
            finish = true;
        }


        return finish;
    }
    unsigned int get_number_of_iterations()
    {
        return iterations;
    }

private:
    vector_operations* vec_ops;
    nonlinear_operator* nonlin_op;
    logging* log;
    unsigned int maximum_iterations;
    unsigned int iterations;
    T tolerance;
    T_vec x1, Fx;
    T newton_wight;
    bool verbose;
};


#endif