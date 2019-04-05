#ifndef __CONTINUATION__PREDICTOR_ADAPTIVE_H__
#define __CONTINUATION__PREDICTOR_ADAPTIVE_H__

namespace continuation
{

template<class VectorOperations>
class predictor_adaptive
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    predictor_adaptive(VectorOperations* vec_ops_, T ds_0_, T step_ds_ = 0.25, unsigned int attempts_0_ = 4):
    vec_ops(vec_ops_),
    ds_0(ds_0_),
    step_ds(step_ds_),
    attempts_0(attempts_0_)
    {
        attempts = 0;
        ds = ds_0;

    }
    ~predictor_adaptive()
    {


    }
    void reset()
    {
        ds = ds_0;
        attempts = 0;
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
    //cublas axpy: y=y+mul_x*x;
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
*/    
        vec_ops->assign(x_0, x_0_p);
        vec_ops->add_mul(ds, x_s, x_0_p);
        lambda_0_p=lambda_0 + ds*lambda_s;
/*
    //calc: y := mul_x*x
    void assign_mul(const scalar_type mul_x, const vector_type& x, vector_type& y)const;
*/      
        vec_ops->assign_mul(T(1.0001), x_0_p, x_1_g);
        lambda_1_g = T(0.999)*lambda_0_p;

    }

    bool modify_ds()
    {
        if(attempts<2*attempts_0)
        {        
            if(attempts<attempts_0)
            {
                ds = ds*T(1+step_ds);
            }
            else if(attempts==attempts_0)
            {
                ds = ds_0*T(1-step_ds);
            }
            else
            {
                ds = ds*T(1-step_ds);
            }
        }
        attempts++;
        
        if(attempts>2*attempts_0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


private:
    T ds_0, ds, step_ds;
    unsigned int attempts_0, attempts;
    VectorOperations* vec_ops;
    T_vec x_s, x_0;
    T lambda_s, lambda_0;
    
};

}

#endif