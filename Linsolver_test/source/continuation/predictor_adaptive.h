#ifndef __CONTINUATION__PREDICTOR_ADAPTIVE_H__
#define __CONTINUATION__PREDICTOR_ADAPTIVE_H__

/**
    predictor class for continuation.
    performs linear prediction of the trjectory in extended (x,\lambda) space
    original point and tangent is set by calling set_tangent_space; reset_tangent_space is used to reset dt and set tangent space
    main apply method sets the linear extrapolated value 
    modify_ds modifies ds either increases or dereases step, thus using a swinging tries.
*/

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <common/scalar_math.h>

namespace continuation
{

template<class VectorOperations, class Logging>
class predictor_adaptive
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    predictor_adaptive(VectorOperations* vec_ops_, Logging* log_, T ds_0_ = 0.1, T ds_max_ = 0.1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4):
    vec_ops(vec_ops_),
    log(log_),
    ds_0(ds_0_),
    ds_max(ds_max_),
    step_ds_p(step_ds_p_),
    step_ds_m(step_ds_m_),
    attempts_0(attempts_0_)
    {
        validate_steps();
        attempts = 0;
        attempts_increase = 0;
        ds = ds_0;
        ds_p = ds;
        ds_m = ds;

    }
    ~predictor_adaptive()
    {
        

    }
    void set_steps(T ds_0_, T ds_max_, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        ds_0 = ds_0_;
        ds_max = ds_max_;
        step_ds_p = step_ds_p_;
        step_ds_m = step_ds_m_;
        attempts_0 = attempts_0_; 
        validate_steps();

        attempts = 0;
        attempts_increase = 0;
        ds = ds_0;
        ds_p = ds;
        ds_m = ds;

    }
    void reset()
    {
        // ds = ds_0;
        // ds_p = ds;
        // ds_m = ds;     

        attempts = 0;
        log->info_f("predictor::arclength.reset: step dS = %le, max dS = %le", (double)ds, (double)ds_max);
    }
    
    //resets all, including ds and advance counters
    void reset_all()
    {
        ds = ds_0;
        ds_p = ds_0;
        ds_m = ds_0;  
        attempts = 0;
        attempts_increase = 0; 
        log->info_f("predictor::arclength.reset_all: step dS = %le, max dS = %le", (double)ds, (double)ds_max);
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
        vec_ops->add_mul(ds, x_s, x_0_p);
        lambda_0_p=lambda_0 + ds*lambda_s;
        log->info_f("predictor::apply: dS = %le, max dS = %le", (double)ds, (double)ds_max);
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
        vec_ops->add_mul(ds, x_s, x_0_p);
        
        lambda_0_p=lambda_0 + ds*lambda_s;
        log->info_f("predictor::apply: dS = %le, max dS = %le", (double)ds, (double)ds_max);
    }

    bool modify_ds()
    {
        if(attempts<4*attempts_0)
        {        

            if(attempts%2==0)
            {
                ds_p = std::min(ds_p*T(1+step_ds_p), ds_max);
                ds = ds_p;
            }
            else
            {
                ds_m = ds_m*T(1-step_ds_m);            
                ds = ds_m;
            }
            log->info_f("predictor::ds_modified to %le", (double)ds);

        }
        else if(attempts == 4*attempts_0)
        {
            ds = ds_0;
            log->info_f("predictor::ds_modified. Max attempts now, ds = %le", (double)ds);
        }
        

        
        if(attempts>4*attempts_0)
        {
            log->info_f("predictor::ds_modified. Max attempts reached");
            return true;
        }
        else
        {
            attempts++;
            return false;
        }
    }


    bool decrease_ds()
    {
        attempts_increase = 0;
        ds = ds*0.2;
        log->info_f("predictor::ds is decreased to %le", (double)ds);
        attempts++;
        if(ds<ds_0*T(1.0e-10))
        {
            log->info_f("predictor::ds decrease. Max attempts reached with ds = %le", (double)ds);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool decrease_ds_adaptive()
    {
        bool modify = modify_ds();
        bool decrease = false;
        if(modify)
        {
            decrease = decrease_ds();
        }
        return decrease;
    }

    void increase_ds()
    {
        attempts_increase++;
        attempts = 0; //reset decrease attempts!!!
        if(attempts_increase>5)
        {
            
            if(ds >= ds_max)
            {
                ds = ds_max;
                log->info_f("predictor::ds is maximized to %le", (double)ds);
            }
            else
            {
                ds = std::min(ds*T(1.25), ds_max);
                log->info_f("predictor::ds is increased to %le", (double)ds);
            }
            attempts_increase = 0;
            
        }

    }


    T get_ds()
    {
        return ds;    
    }
    T get_ds_max()
    {
        return ds_max;
    }

private:
    void validate_steps()
    {
        if((!common::scalar_math::isfinite(ds_0)) || (ds_0 <= T(0)))
        {
            throw std::runtime_error("predictor_adaptive: initial arclength step ds_0 must be positive and finite.");
        }
        if((!common::scalar_math::isfinite(ds_max)) || (ds_max <= T(0)))
        {
            throw std::runtime_error("predictor_adaptive: maximum arclength step ds_max must be positive and finite.");
        }
        if(ds_max < ds_0)
        {
            throw std::runtime_error("predictor_adaptive: maximum arclength step ds_max must be greater than or equal to ds_0.");
        }
        if((!common::scalar_math::isfinite(step_ds_m)) || (step_ds_m < T(0)) || (step_ds_m >= T(1)))
        {
            throw std::runtime_error("predictor_adaptive: step_ds_m must be finite and in [0,1).");
        }
        if((!common::scalar_math::isfinite(step_ds_p)) || (step_ds_p < T(0)))
        {
            throw std::runtime_error("predictor_adaptive: step_ds_p must be finite and non-negative.");
        }
        if(attempts_0 == 0)
        {
            throw std::runtime_error("predictor_adaptive: attempts_0 must be positive.");
        }
    }

    T ds_max, ds_0, ds, step_ds_p, step_ds_m, ds_p, ds_m;
    unsigned int attempts_0, attempts, attempts_increase;
    VectorOperations* vec_ops;
    T_vec x_s, x_0;
    T lambda_s, lambda_0;
    Logging* log;
    
};

}

#endif
