#ifndef __CONVERGENCE_STRATEGY_CONTINUATION_H__
#define __CONVERGENCE_STRATEGY_CONTINUATION_H__
/**
convergence rules for Newton iterator for continuation process
*/
#include <cmath>
#include <vector>
#include <utils/logged_obj_base.h>

namespace continuation
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

    bool check_convergence(nonlinear_operator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, int& result_status)
    {
        bool finish = false;
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm_l2(Fx);
        //update solution
        vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        T lambda1 = lambda + newton_wight*delta_lambda;
        nonlin_op->F(x1, lambda1, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);

        iterations++;
        log->info_f("convergence_strategy::iteration %i, previous residual %le, current residual %le ",iterations, (double)normFx, (double)normFx1);
        // if(normFx1>10.0*normFx) //cancel update and decrease the nonlinear update wight if the norm is growing
        // {   
        //     iterations--;
        //     newton_wight*=T(0.75);
        //     lambda1 = lambda;
        //     vec_ops->assign(x, x1);
        //     log->info_f("convergence_strategy::step is cancelled, Newton wight updated to (%le).", double(newton_wight) );
        // }
        // else
        // {
            //store norm only if the step is seccessfull
            if(store_norms_history)
            {
                norms_evolution.push_back(normFx1);
            }
        //}
        if(normFx1<10.0*normFx)
        {
            reset_wight();
            log->info_f("convergence_strategy::fast descent, Newton wight updated to (%le).", double(newton_wight) );
        }
        if(newton_wight<T(1.0e-6))
        {
            log->info_f("convergence_strategy::Newton wight is too small (%le).", double(newton_wight));   
            finish = true;
            result_status = 4;
        }
        if(std::isnan(normFx))
        {
            log->info("convergence_strategy::Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isnan(normFx1))
        {
            log->info("convergence_strategy::Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }else if(std::isinf(normFx))
        {
            log->info("convergence_strategy::Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }else if(std::isinf(normFx1))
        {
            log->info("convergence_strategy::Newton update caused inf.");
            finish = true;
            result_status = 2;            
        }else
        {   
            //update solution
            lambda = lambda1;
            vec_ops->assign(x1,x);
        }
        if(normFx1<tolerance)
        {
            log->info_f("convergence_strategy::Newton converged with %le.", (double)normFx1);
            result_status = 0;
            finish = true;
        }
        else if(iterations>=maximum_iterations)
        {
            log->info("convergence_strategy::Newton max iterations (%i) reached.", iterations);
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
    bool verbose, store_norms_history;
    std::vector<T> norms_evolution;

};


}
}

#endif