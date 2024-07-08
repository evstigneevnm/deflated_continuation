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
#include <utils/logged_obj_base.h>

namespace nonlinear_operators
{
namespace newton_method
{


template<class VectorOperations, class NonlinearOperator, class Logging>
class convergence_strategy
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef utils::logged_obj_base<Logging> logged_obj_t;

public:    
    convergence_strategy(VectorOperations* vec_ops_p, Logging* log_p, T tolerance_p = 1.0e-6, unsigned int maximum_iterations_p = 100, T newton_wight_p = T(1), bool store_norms_history_p = false, bool verbose_p = true):
    vec_ops_(vec_ops_p),
    log_(log_p),
    iterations_(0),
    tolerance_(tolerance_p),
    maximum_iterations_(maximum_iterations_p),
    newton_wight_(newton_wight_p),
    newton_wight_initial_(newton_wight_p),
    verbose_(verbose_p),
    store_norms_history_(store_norms_history_p),
    critical_newton_wight_(1.0e-9)
    {
        vec_ops_->init_vector(x1); vec_ops_->start_use_vector(x1);
        vec_ops_->init_vector(Fx); vec_ops_->start_use_vector(Fx);
        if(store_norms_history_)
        {
            norms_evolution_.reserve(maximum_iterations_);
        }
    }
    ~convergence_strategy()
    {
        vec_ops_->stop_use_vector(x1); vec_ops_->free_vector(x1);
        vec_ops_->stop_use_vector(Fx); vec_ops_->free_vector(Fx);
    }

    void set_convergence_constants(T tolerance_p, unsigned int maximum_iterations_p, T newton_wight_p = T(1), bool store_norms_history_p = false, bool verbose_p = true)
    {
        tolerance_ = tolerance_p;
        maximum_iterations_ = maximum_iterations_p;
        newton_wight_ = newton_wight_p;
        store_norms_history_ = store_norms_history_p;
        verbose_ = verbose_p;
        if(store_norms_history_)
        {
            norms_evolution_.reserve(maximum_iterations_);
        }        
    }
    

    bool check_convergence(NonlinearOperator* nonlin_op, T_vec& x, T lambda, T_vec& delta_x, int& result_status, bool lin_solver_converged = true)
    {
        if(!lin_solver_converged)
        {
            result_status = 5;
            // return true;
        }
        bool finish = false;
        nonlin_op->F(x, lambda, Fx);
        T normFx = vec_ops_->norm_l2(Fx);
        //update solution
        vec_ops_->assign_mul(T(1.0), x, newton_wight_, delta_x, x1);
        nonlin_op->F(x1, lambda, Fx);
        T normFx1 = vec_ops_->norm_l2(Fx);
        if(store_norms_history_)
        {
            norms_evolution_.push_back(normFx1);
        }
        log_->info_f("NonlinearOperators::convergence: iteration %i, previous residual %le, current residual %le",iterations_, (double)normFx, (double)normFx1);
        if(normFx1/normFx>T(1.5))
        {
            newton_wight_ = T(1.49)*(normFx/normFx1);
            log_->warning_f("NonlinearOperators::convergence: adjusting Newton wight to %le and updating...", newton_wight_);            
            vec_ops_->assign_mul(T(1), x, newton_wight_, delta_x, x1);    
            if(newton_wight_<critical_newton_wight_)
            {
                log_->error_f("NonlinearOperators::convergence: Newton wight is too small (%le).", newton_wight_);   
                finish = true;
                result_status = 4;
            }
            reset_wight();            
        }
        if(( abs(normFx1-normFx)/normFx<T(0.05))&&(iterations_>maximum_iterations_/3))
        {
            newton_wight_ *= 0.5;
            log_->warning_f("NonlinearOperators::convergence: adjusting Newton wight to %le and updating...", newton_wight_);   
            vec_ops_->assign_mul(T(1), x, newton_wight_, delta_x, x1);            
        }
        iterations_++;

        if(newton_wight_<critical_newton_wight_)
        {
            log_->error_f("NonlinearOperators::convergence: Newton wight is too small (%le).", newton_wight_);   
            finish = true;
            result_status = 4;
        }
        if( isnan(normFx))
        {
            log_->error("NonlinearOperators::convergence: Newton initial vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if( isnan(normFx1))
        {
            log_->error("NonlinearOperators::convergence: Newton updated vector caused nan.");
            finish = true;
            result_status = 3;
        }
        else if( isinf(normFx))
        {
            log_->error("NonlinearOperators::convergence: Newton initial vector caused inf.");
            finish = true;
            result_status = 2;            
        }
        else if( isinf(normFx1))
        {
            log_->error("NonlinearOperators::convergence: Newton update caused inf.");
            finish = true;
            result_status = 2;            
        }
        else
        {   
            vec_ops_->assign(x1,x);
        }
        if(normFx1<tolerance_)
        {
            log_->info_f("NonlinearOperators::convergence: Newton converged with %le.", (double)normFx1);
            result_status = 0;
            finish = true;
        }
        else if(iterations_>=maximum_iterations_)
        {
            log_->warning_f("NonlinearOperators::convergence: Newton max iterations (%i) reached.", iterations_);
            result_status = 1;
            finish = true;
        }


        return finish;
    }
    unsigned int get_number_of_iterations()
    {
        return iterations_;
    }
    void reset_iterations()
    {
        iterations_ = 0;
        reset_wight();
        norms_evolution_.clear();
    }
    void reset_wight()
    {
        newton_wight_ = newton_wight_initial_;
    }
    std::vector<T>* get_norms_history_handle()
    {
        return &norms_evolution_;
    }    

private:
    T critical_newton_wight_;
    VectorOperations* vec_ops_;
    Logging* log_;
    unsigned int maximum_iterations_;
    unsigned int iterations_;
    T tolerance_;
    T_vec x1, Fx;
    T newton_wight_, newton_wight_initial_;
    bool verbose_,store_norms_history_;
    std::vector<T> norms_evolution_;
};


}
}

#endif