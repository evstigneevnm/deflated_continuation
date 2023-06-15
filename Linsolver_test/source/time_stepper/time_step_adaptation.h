#ifndef __TIME_STEPPER_TIME_STEP_ADAPTATION_H__
#define __TIME_STEPPER_TIME_STEP_ADAPTATION_H__

#include <utility>

namespace time_steppers
{

//TODO: 
// add three more classes:
// class time_step_adaptation_constantl //<-used constant step ode integration
// class time_step_adaptation_error_control //<-used for normal ode integration
// class time_step_adaptation_globalization //<-used for globalization process in steady state solvers

template<class VectorOperations, class Log>
class time_step_adaptation
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_step_adaptation(VectorOperations* vec_ops_p, Log* log_p, std::pair<T,T> time_interval_p = {0,1},  T dt_p = 1.0):
    vec_ops_(vec_ops_p),
    log_(log_p),
    time_interval_(time_interval_p),
    dt_(dt_p),
    dt_initial_(dt_p),
    current_step_(0),
    fail_flag_(false)
    {
        current_time_ = time_interval_.first;
        T final_time = time_interval_.second;
        dt_ = ((final_time- current_time_)>3.0*dt_p)?dt_p:(final_time- current_time_)/3.0;

        if( std::abs(current_time_)<=std::numeric_limits<T>::epsilon() )
        {
            dt_min_ = static_cast<T>(16.0)*std::numeric_limits<T>::denorm_min();
        }
        else
        {
            dt_min_ = static_cast<T>(16.0)*std::numeric_limits<T>::epsilon();
        }
    }
    
    virtual ~time_step_adaptation()
    {}
        
    T get_dt()const
    {
        return dt_;
    }
    T get_time()const
    {
        return current_time_;
    }
    size_t get_step()const
    {
        return current_step_;
    }
    time_step_adaptation& operator++()
    {
        current_time_ += dt_;
        current_step_++;
        if((current_time_+dt_)>time_interval_.second)
        {
            dt_ = time_interval_.second - current_time_;
        }
        return *this;
    }
    std::pair<bool,bool> chech_finished()const
    {
        if(fail_flag_)
        {
            return {fail_flag_, false};
        }
        else if(current_time_-time_interval_.second >= 0)
        {
            return {fail_flag_, true};
        }
        else
        {
            return{false, false};
        }

    }

    virtual bool estimate_timestep(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)const = 0;


protected:
    T dt_;
    T dt_min_;
    T dt_initial_;
    std::pair<T,T> time_interval_;
    mutable T current_time_;
    VectorOperations* vec_ops_;
    Log* log_;
    size_t current_step_;
    bool fail_flag_;

};
}

#endif