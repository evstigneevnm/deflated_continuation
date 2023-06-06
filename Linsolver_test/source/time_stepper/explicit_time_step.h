#ifndef __TIME_STEPPER_EXPLICIT_TIME_STEP_H__
#define __TIME_STEPPER_EXPLICIT_TIME_STEP_H__

#include <string>
#include <vector>
#include <stdexcept>

namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, class Log>

class explicit_time_step
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    explicit_time_step(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_, Log* log_):
    vec_ops(vec_ops_), nonlin_op(nonlin_op_), log(log_)
    {
        vec_ops->init_vector(v1_helper); vec_ops->start_use_vector(v1_helper);
        vec_ops->init_vector(f_helper); vec_ops->start_use_vector(f_helper);

    }
    ~explicit_time_step()
    {
        vec_ops->stop_use_vector(v1_helper); vec_ops->free_vector(v1_helper);
        vec_ops->stop_use_vector(f_helper); vec_ops->free_vector(f_helper);

    }
    
    void scheme(const std::string& ts_scheme_)
    {
        bool found_allowed = false;
        for(auto &scheme_allowed: ts_scheme_permisive)
        {
            if(scheme_allowed == ts_scheme_)
            {
                ts_scheme = ts_scheme_; 
                found_allowed = true;
                break;
            }    
        }
        if(!found_allowed)
        {
            throw std::logic_error("explicit_time_step::scheme: No allowed time marching scheme is found, user provided " + ts_scheme_);
        }

    }

    void set_parameter(const T param_)
    {
        param = param_;
    }


    //TODO: execute an automatic timestepping algorithm
    void set_time_step(const T time_step_)
    {
        time_step = time_step_;
    }
    void override_single_time_step(const T time_step_override_)
    {
        time_step_override = time_step_override_;
    }
    T get_time_step()
    {
        return time_step;
    }

    void execute(const T_vec in_, T_vec out_)
    {
        T dt = time_step;
        if(time_step_override > T(0.0) )
        {
            dt = time_step_override;
            time_step_override = T(-1.0);
        }
        

        if(ts_scheme == "RK3ssp")
        {
            //step1
            nonlin_op->F(in_, param, f_helper);
            //calc: z := mul_x*x + mul_y*y
            vec_ops->assign_mul(T(1.0),in_, f_sign*dt, f_helper, out_);
            // step2
            nonlin_op->F(out_, param, f_helper);
            vec_ops->add_mul(T(0.75), in_, T(0.25)*f_sign*dt, f_helper, T(0.25), out_);
            // step3
            nonlin_op->F(out_, param, f_helper);
            vec_ops->add_mul(T(1.0/3.0), in_, T(2.0/3.0)*f_sign*dt, f_helper, T(2.0/3.0), out_);

        }


    }


private:
    T f_sign = T(1.0);
    T time_step;  
    T param;
    T time_step_override = T(-1.0);
    T_vec v1_helper = nullptr;
    T_vec f_helper = nullptr;

    std::string ts_scheme = "RK3ssp";
    const std::vector<std::string> ts_scheme_permisive = {"RK3ssp", "RK4", "RK4ssp", "DP8", "RK3exp"};

    VectorOperations* vec_ops;
    NonlinearOperator* nonlin_op;
    Log* log;

};



}



#endif