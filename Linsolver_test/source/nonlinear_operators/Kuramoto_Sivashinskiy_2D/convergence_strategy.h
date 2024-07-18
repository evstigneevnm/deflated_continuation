#ifndef __CONVERGENCE_STRATEGY_H__
#define __CONVERGENCE_STRATEGY_H__
/**
*
*convergence rules for Newton iterator
*
*return status:
*    0 - ok
*    1 - max iterations exceeded
*    2 - inf update
*    3 - nan update
*    4 - wight update is too small
*
*/

#include <cmath>
#include <vector>
#include <limits>
#include <utils/logged_obj_base.h>

namespace nonlinear_operators
{
namespace newton_method
{


template<class vector_operations, class NonlinearOperator, class logging>
class convergence_strategy
{
private:
    typedef typename vector_operations::scalar_type  T;
    typedef typename vector_operations::vector_type  T_vec;
    typedef utils::logged_obj_base<logging> logged_obj_t;

public:    
    convergence_strategy(vector_operations*& vec_ops_, logging*& log_, T tolerance_ = T(1.0e-6), unsigned int maximum_iterations_ = 100, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true):
    vec_ops(vec_ops_),
    log(log_),
    iterations(0),
    tolerance(tolerance_),
    maximum_iterations(maximum_iterations_),
    newton_wight(newton_wight_),
    newton_wight_initial(newton_wight_),
    verbose(verbose_),
    store_norms_history(store_norms_history_)
    {
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(Fx); vec_ops->start_use_vector(Fx);
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        }
        maximum_norm_increase_ = 0.0;
        newton_wight_threshold_ = 100.0*std::numeric_limits<T>::epsilon();
    }
    ~convergence_strategy()
    {
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    }

    void set_convergence_constants(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true)
    {
        tolerance = tolerance_;
        maximum_iterations = maximum_iterations_;
        newton_wight = newton_wight_;
        store_norms_history = store_norms_history_;
        verbose = verbose_;
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        }        
    }
    
    T inline update_solution(NonlinearOperator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T_vec& x1)
    {
        vec_ops->assign_mul(static_cast<T>(1.0), x, newton_wight, delta_x, x1);
        // nonlin_op->project(x1); // project to invariant solution subspace. Should be blank if nothing is needed to be projected.
        nonlin_op->F(x1, lambda, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);
        return normFx1;
    }

    bool check_convergence(NonlinearOperator* nonlin_op, T_vec& x, T lambda, T_vec& delta_x, int& result_status, bool lin_solver_converged = true)
    {
        if(!lin_solver_converged)
        {
            result_status = 5;
            // return true;
        }
        bool finish = false;
        reset_wight();
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm_l2(Fx);
        //update solution
        vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        nonlin_op->F(x1, lambda, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);
        if(store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }
        // log->info_f("nonlinear_operators::convergence_strategy::iteration %i, previous residual %le, current residual %le",iterations, (double)normFx, (double)normFx1);
        // if(normFx1>T(2)*normFx)
        // {
        //     newton_wight *= 0.75;
        //     log->info_f("nonlinear_operators::convergence_strategy::adjusting Newton wight to %le and updating...", newton_wight);            
        //     vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        // }
        // if((std::abs(normFx1-normFx)/normFx<T(0.05))&&(iterations>maximum_iterations/3))
        // {
        //     newton_wight *= 0.75;
        //     log->info_f("nonlinear_operators::convergence_strategy::adjusting Newton wight to %le and updating...", newton_wight);   
        //     vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);            
        // }
        {
            while( (normFx1 - normFx) > maximum_norm_increase_ )
            {
                newton_wight *= 0.5;
                normFx1 = update_solution(nonlin_op, x, lambda, delta_x, x1);
                if(newton_wight < newton_wight_threshold_)
                {
                    result_status = 4;
                    finish = true;
                }
                // log->info_f("nonlinear_operators::convergence:wight_update: iteration %i, residuals n: %le, n+1: %le, wight: %le",iterations, (double)normFx, (double)normFx1, (double)newton_wight );
            }
        }        
        iterations++;
        log->info_f("nonlinear_operators::convergence: iteration %i, residuals n: %le, n+1: %le, wight: %le",iterations, (double)normFx, (double)normFx1, (double)newton_wight );
        reset_wight();

        // if(newton_wight<T(1.0e-6))
        // {
        //     log->info_f("nonlinear_operators::convergence_strategy::Newton wight is too small (%le).", newton_wight);   
        //     finish = true;
        //     result_status = 4;
        // }
        if(std::isnan(normFx))
        {
            log->info("nonlinear_operators::convergence_strategy::Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isnan(normFx1))
        {
            log->info("nonlinear_operators::convergence_strategy::Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isinf(normFx))
        {
            log->info("nonlinear_operators::convergence_strategy::Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }
        else if(std::isinf(normFx1))
        {
            log->info("nonlinear_operators::convergence_strategy::Newton update caused inf.");
            finish = true;
            result_status = 2;            
        }
        else
        {   
            //update solution
            vec_ops->assign(x1,x);
        }
        if(normFx1<tolerance)
        {
            log->info_f("nonlinear_operators::convergence_strategy::Newton converged with %le.", (double)normFx1);
            result_status = 0;
            finish = true;
        }
        else if(iterations>=maximum_iterations)
        {
            log->info("nonlinear_operators::convergence_strategy::Newton max iterations (%i) reached.", iterations);
            result_status = 1;
            finish = true;
        }


        return finish;
    }
    unsigned int get_number_of_iterations()
    {
        return iterations;
    }
    void reset_iterations()
    {
        iterations = 0;
        reset_wight();
        norms_evolution.clear();
    }
    void reset_wight()
    {
        newton_wight = newton_wight_initial;
    }
    std::vector<T>* get_norms_history_handle()
    {
        return &norms_evolution;
    }    

private:
    vector_operations* vec_ops;
    logging* log;
    unsigned int maximum_iterations;
    unsigned int iterations;
    T tolerance;
    T_vec x1, Fx;
    T newton_wight, newton_wight_initial;
    bool verbose,store_norms_history;
    std::vector<T> norms_evolution;
    T maximum_norm_increase_;
    T newton_wight_threshold_;
};


}
}

#endif