#ifndef __CONTINUATION__PREDICTOR_ADAPTIVE_H__
#define __CONTINUATION__PREDICTOR_ADAPTIVE_H__

/**
    predictor class for continuation.
    performs linear prediction of the trjectory in extended (x,\lambda) space
    original point and tangent is set by calling set_tangent_space; reset_tangent_space is used to reset dt and set tangent space
    main apply method sets the linear extrapolated value 
    Retry failures reduce the step monotonically; sustained clean steps grow it up
    to the configured maximum.
*/

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <continuation/corrector_retry_policy.h>

namespace continuation
{

template<class VectorOperations, class Logging>
class predictor_adaptive
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    predictor_adaptive(VectorOperations* vec_ops_, Logging* log_, T ds_0_ = 0.1, T ds_max_ = 0.1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4):
    step_controller(ds_0_, ds_max_, legacy_policy(step_ds_m_, step_ds_p_, attempts_0_)),
    vec_ops(vec_ops_),
    log(log_)
    {
    }
    ~predictor_adaptive()
    {
        

    }
    void set_steps(T ds_0_, T ds_max_, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        step_controller.configure(
            ds_0_, ds_max_, legacy_policy(step_ds_m_, step_ds_p_, attempts_0_));
    }

    void set_steps(
        const T ds_0_,
        const T ds_max_,
        const corrector_retry_policy<T>& policy)
    {
        step_controller.configure(ds_0_, ds_max_, policy);
    }

    void set_verbose(const bool value)
    {
        verbose = value;
    }

    void begin_step()
    {
        step_controller.begin_step();
    }

    void reset()
    {
        begin_step();
        if(verbose)
        {
            log->info_f(
                "predictor::arclength.begin_step: dS = %le, max dS = %le",
                (double)get_ds(),
                (double)get_ds_max());
        }
    }
    
    //resets all, including ds and advance counters
    void reset_all()
    {
        step_controller.reset_semicurve();
        if(verbose)
        {
            log->info_f(
                "predictor::arclength.reset_semicurve: dS = %le, max dS = %le",
                (double)get_ds(),
                (double)get_ds_max());
        }
    }

    void set_tangent_space(const T_vec& x_0_, const T& lambda_0_, const T_vec& x_s_, const T& lambda_s_)
    {
        
        x_0 = x_0_;
        lambda_0 = lambda_0_;
        x_s = x_s_;
        lambda_s = lambda_s_;

    }
    void reset_tangent_space(const T_vec& x_0_, const T& lambda_0_, const T_vec& x_s_, const T& lambda_s_)
    {
        reset();   
        set_tangent_space(x_0_, lambda_0_, x_s_, lambda_s_);
    }

    //apply only returns predictor results:
    // x_0_p = x_0+x_s*x_0
    // lambda_0_p = lambda_s*lambda_0
    // x_1_g = 1.0001*x_0_p
    // lambda_1_g = 0.999*lambda_0_p
    void apply(T_vec& x_0_p, T& lambda_0_p, T_vec& x_1_g, T& lambda_1_g)
    {
//      x0_guess = x0+delta_s1.*x0_s;   
//      lambda0_guess = lambda0+delta_s1.*lambda0_s;        
/*
//  cublas axpy: y=y+mul_x*x;
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
*/
        vec_ops->assign(x_0, x_0_p);
        const T ds = get_ds();
        vec_ops->add_mul(ds, x_s, x_0_p);
        lambda_0_p=lambda_0 + ds*lambda_s;
        if(verbose)
        {
            log->info_f("predictor::apply: dS = %le, max dS = %le", (double)ds, (double)get_ds_max());
        }
/*
    //calc: y := mul_x*x
    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y)const;
*/
        vec_ops->assign_mul(T(1), x_0_p, x_1_g);
        lambda_1_g = lambda_0_p;

    }

    void apply(T_vec& x_0_p, T& lambda_0_p)
    {
//      x0_guess = x0+delta_s1.*x0_s;   
//      lambda0_guess = lambda0+delta_s1.*lambda0_s;        
/*
//  cublas axpy: y=y+mul_x*x;
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
*/
        vec_ops->assign(x_0, x_0_p);
        const T ds = get_ds();
        vec_ops->add_mul(ds, x_s, x_0_p);
        
        lambda_0_p=lambda_0 + ds*lambda_s;
        if(verbose)
        {
            log->info_f("predictor::apply: dS = %le, max dS = %le", (double)ds, (double)get_ds_max());
        }
    }

    step_retry_result retry_after_failure()
    {
        const auto result = step_controller.retry_after_failure();
        if(result == step_retry_result::retry)
        {
            log->warning_f(
                "predictor::arclength: corrector retry %u of %u reduced dS to %le",
                step_controller.retries(),
                step_controller.policy().maximum_retries,
                (double)get_ds());
        }
        return result;
    }

    step_retry_result retry_after_chart_rejection(const T factor)
    {
        const auto result = step_controller.retry_with_factor(factor);
        if(result == step_retry_result::retry)
        {
            log->warning_f(
                "predictor::arclength: chart retry %u of %u reduced dS to %le",
                step_controller.retries(),
                step_controller.policy().maximum_retries,
                (double)get_ds());
        }
        return result;
    }

    step_retry_result retry_at_step(const T requested_step)
    {
        const auto result = step_controller.retry_at_step(requested_step);
        if(result == step_retry_result::retry)
        {
            log->warning_f(
                "predictor::arclength: event-bracket retry %u of %u set dS to %le",
                step_controller.retries(),
                step_controller.policy().maximum_retries,
                (double)get_ds());
        }
        return result;
    }

    step_retry_result reduce_next_step(const T factor)
    {
        const auto result = step_controller.reduce_next_step(factor);
        if(result == step_retry_result::retry)
        {
            log->warning_f(
                "predictor::arclength: branch-event refinement reduced the next dS to %le",
                (double)get_ds());
        }
        return result;
    }

    // Compatibility wrappers for legacy callers. New continuation code uses the
    // typed retry result above.
    bool decrease_ds()
    {
        return retry_after_chart_rejection(T(0.2)) != step_retry_result::retry;
    }

    bool decrease_ds_monotone(const T factor)
    {
        return retry_after_chart_rejection(factor) != step_retry_result::retry;
    }

    bool decrease_ds_adaptive()
    {
        return retry_after_failure() != step_retry_result::retry;
    }

    void increase_ds()
    {
        accept_step(false);
    }

    void accept_step(const bool recovered)
    {
        const T previous_ds = get_ds();
        step_controller.accept_step(recovered);
        if(verbose && get_ds() != previous_ds)
        {
            log->info_f("predictor::arclength: increased dS to %le", (double)get_ds());
        }
    }

    T get_ds() const
    {
        return step_controller.step();
    }
    T get_ds_max() const
    {
        return step_controller.maximum_step();
    }
    T get_initial_ds() const
    {
        return step_controller.initial_step();
    }
    T get_minimum_ds() const
    {
        return step_controller.minimum_step();
    }
    unsigned int get_retry_count() const
    {
        return step_controller.retries();
    }

private:
    static corrector_retry_policy<T> legacy_policy(
        const T step_ds_m,
        const T step_ds_p,
        const unsigned int attempts)
    {
        corrector_retry_policy<T> policy;
        policy.maximum_retries = attempts;
        policy.failure_reduction_factor = T(1)-step_ds_m;
        // The legacy success path ignored step_ds_p and multiplied dS by 1.25
        // after attempts_increase became greater than five.
        policy.successes_before_growth = 6;
        policy.success_growth_factor = T(1.25);
        policy.validate();
        return policy;
    }

    adaptive_step_controller<T> step_controller;
    VectorOperations* vec_ops;
    T_vec x_s, x_0;
    T lambda_s, lambda_0;
    Logging* log;
    bool verbose = true;
    
};

}

#endif
