#ifndef __CONVERGENCE_STRATEGY_DEFLATION_H__
#define __CONVERGENCE_STRATEGY_DEFLATION_H__
/*
converhence rules for Newton iterator for deflation process
*/
#include <cmath>
#include <vector>
//#include <utils/logged_obj_base.h>

namespace deflation
{
namespace newton_method_extended
{


template<class VectorOperations, class NonlinearOperator, class Logging>
class convergence_strategy
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
//    typedef utils::logged_obj_base<Logging> logged_obj_t;

public:    
    convergence_strategy(VectorOperations*& vec_ops_, Logging*& log_, T tolerance_ = 1.0e-6, unsigned int maximum_iterations_ = 100, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true):
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
        stagnation_max = 10;
    }
    ~convergence_strategy()
    {
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    }

    void set_convergence_constants(T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, unsigned int stagnation_max_ = 10)
    {
        tolerance = tolerance_;
        maximum_iterations = maximum_iterations_;
        newton_wight = newton_wight_;
        store_norms_history = store_norms_history_;
        verbose = verbose_;
        stagnation_max = stagnation_max_;
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        }        
    }

    bool check_convergence(NonlinearOperator* nonlin_op, T_vec& x, T& lambda, T_vec& delta_x, T& delta_lambda, int& result_status)
    {
    
        bool finish = false;
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops->norm_l2(Fx);
        //update solution
        vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        T lambda1 = lambda + newton_wight*delta_lambda;
        nonlin_op->F(x1, lambda1, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);
        if(store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }

        iterations++;
        log->info_f("deflation::convergence: iteration %i, residuals n: %le, n+1: %le",iterations, (double)normFx, (double)normFx1);

        if(std::isnan(normFx))
        {
            log->error("deflation::convergence: Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if(std::isnan(normFx1))
        {
            log->error("deflation::convergence: Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }else if(std::isinf(normFx))
        {
            log->error("deflation::convergence: Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }else if(std::isinf(normFx1))
        {
            log->error("deflation::convergence: Newton update caused inf.");
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
            log->info_f("deflation::convergence: Newton converged with %le for %i iterations.", (double)normFx1, iterations);
            result_status = 0;
            finish = true;
        }
        else if(iterations>=maximum_iterations)
        {
            log->warning_f("deflation::convergence: Newton max iterations (%i) reached with norm %le", iterations, (double)normFx1);
            result_status = 1;
            finish = true;
        }
        if( std::abs(normFx-normFx1)<1.0e-6)
        {
            stagnation++;
        }
        if(stagnation>stagnation_max)
        {   
    
            if(normFx1 < 1.0e-6)
            {
                result_status = 0; 
                log->warning_f("deflation::convergence: Newton stagnated at iteration(%i) with norm %le < 1.0e-6. Assume covergence.", iterations, (double)normFx1);
            }
            else
            {
                result_status = 1;
                log->warning_f("deflation::convergence: Newton stagnated at iteration(%i) with norm %le >= 1.0e-6. Assume covergence failed.", iterations, (double)normFx1);                
            }
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
        stagnation = 0;
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
    unsigned int stagnation = 0;
    unsigned int stagnation_max = 0;
    VectorOperations* vec_ops;
    Logging* log;
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