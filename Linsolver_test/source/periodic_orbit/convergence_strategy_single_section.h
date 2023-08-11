#ifndef __PERIODIC_ORBIT_CONVERGENCE_STRATEGY_H__
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

template<class VectorOperations, class NonlinearOperator, class Log>
class convergence_strategy_single_section
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef utils::logged_obj_base<Log> logged_obj_t;

public:    
    convergence_strategy_single_section(VectorOperations* vec_ops_, Log* log_, T tolerance_ = 1.0e-6, unsigned int maximum_iterations_ = 100, T newton_wight_ = 1.0, bool store_norms_history_ = false, bool verbose_ = true):
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
    }
    ~convergence_strategy_single_section()
    {
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    }

    void set_convergence_constants(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = 1.0, bool store_norms_history_ = false, bool verbose_ = true)
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
    

    bool check_convergence(NonlinearOperator* nonlin_op, T_vec& x, T lambda, T_vec& delta_x, int& result_status)
    {
        reset_wight();
        bool finish = false;
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm(Fx);
        vec_ops->assign_mul(T(1.0), x, newton_wight, delta_x, x1);
        nonlin_op->F(x1, lambda, Fx);
        T normFx1 = vec_ops->norm(Fx);
        //update solution
        // vec_ops->assign_mul(T(1.0), x, newton_wight, delta_x, x1);


        log->info_f("nonlinear_operators::convergence: iteration %i, previous residual %le, initial current residual %le",iterations, (double)normFx, (double)normFx1);
        bool iterations_residual = false; 
        while(true)
        {
            vec_ops->assign_mul(T(1.0), x, newton_wight, delta_x, x1);
            nonlin_op->F(x1, lambda, Fx);
            normFx1 = vec_ops->norm(Fx);
            if(normFx1 < (1.0 + std::numeric_limits<T>::epsilon() )*normFx)
            {
                result_status = 0;
                if(iterations_residual)
                {
                    log->info_f("nonlinear_operators::convergence: iteration %i, previous residual %le, initial current residual %le",iterations, (double)normFx, (double)normFx1);
                }
                break;
            }
            iterations_residual = true;
            if( newton_wight<T(1.0e-6) )
            {
                log->error_f("nonlinear_operators::convergence: Newton wight is too small (%le).", newton_wight);   
                finish = true;
                result_status = 4;
                break;
            }          
            log->warning_f("nonlinear_operators::convergence: adjusting Newton wight to %le and updating...", newton_wight*0.5);     
            newton_wight *= 0.5;             
                      
        }
        if(std::isnan(normFx))
        {
            log->error("nonlinear_operators::convergence: Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isnan(normFx1))
        {
            log->error("nonlinear_operators::convergence: Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isinf(normFx))
        {
            log->error("nonlinear_operators::convergence: Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }
        else if(std::isinf(normFx1))
        {
            log->error("nonlinear_operators::convergence: Newton update caused inf.");
            finish = true;
            result_status = 2;            
        }
        else
        {   
            vec_ops->assign(x1, x);
        }
        if(normFx1<tolerance)
        {
            log->info_f("nonlinear_operators::convergence: Newton converged with %le.", (double)normFx1);
            result_status = 0;
            finish = true;
        }
        else if(iterations>=maximum_iterations)
        {
            log->warning_f("nonlinear_operators::convergence: Newton max iterations (%i) reached.", iterations);
            result_status = 1;
            finish = true;
        }

        if(store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }
        ++iterations;
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
    VectorOperations* vec_ops;
    Log* log;
    unsigned int maximum_iterations;
    unsigned int iterations;
    T tolerance;
    T_vec x1, Fx;
    T newton_wight, newton_wight_initial;
    bool verbose,store_norms_history;
    std::vector<T> norms_evolution;
};


}
}

#endif