#ifndef __TIME_STEPPER_TIME_STEP_ADAPTATION_H__
#define __TIME_STEPPER_TIME_STEP_ADAPTATION_H__

#include <utility>
#include <time_stepper/detail/positive_preserving_dummy.h>

namespace time_steppers
{

//TODO: 
// add three more classes:
// class time_step_adaptation_constantl //<-used constant step ode integration
// class time_step_adaptation_error_control //<-used for normal ode integration
// class time_step_adaptation_globalization //<-used for globalization process in steady state solvers

template<class VectorOperations, class Log, class PositivePreservingCheck= detail::positive_preserving_dummy<VectorOperations> >
class time_step_adaptation
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_step_adaptation(VectorOperations* vec_ops_p, Log* log_p, std::pair<T,T> time_interval_p = {0,1},  T dt_p = 1.0, PositivePreservingCheck* positive_check_p = new detail::positive_preserving_dummy<VectorOperations>() ):
    vec_ops_(vec_ops_p),
    log_(log_p),
    time_interval_(time_interval_p),
    dt_(dt_p),
    dt_prev_(dt_p),
    dt_initial_(dt_p),
    current_step_(0),
    fail_flag_(false),
    positive_check_(positive_check_p)
    {
        current_time_ = time_interval_.first;
        T final_time = time_interval_.second;
        dt_max_ = 0.1*(final_time - current_time_); //default in matlab

        dt_ = ((final_time- current_time_)>3.0*dt_p)?dt_p:(final_time- current_time_)/3.0;
        calcualte_dt_min();
    
    }
    
    virtual ~time_step_adaptation()
    {}
      
    T get_dt()const
    {
        return dt_prev_;
    }
    T get_time()const
    {
        return current_time_;
    }

    size_t get_iteration()const
    {
        return current_step_;
    }
    time_step_adaptation& operator++()
    {
        current_time_ += dt_prev_;
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
        else if(current_time_-time_interval_.second >= 0.0)
        {
            return {fail_flag_, true};
        }
        else
        {
            return{false, false};
        }

    }

    bool check_solution_consistency(const T_vec& sol_p)const
    {
        return positive_check_->apply(sol_p);
    }

    virtual void pre_execte_step()const = 0;
    virtual bool check_reject_step()const = 0;
    virtual void init_steps(const T_vec& x_p, const T_vec& fx_p) = 0;
    virtual bool estimate_timestep(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p) = 0;


protected:
    mutable T dt_;
    mutable T dt_prev_;
    mutable T dt_min_;
    T dt_initial_;
    std::pair<T,T> time_interval_;
    mutable T current_time_;
    VectorOperations* vec_ops_;
    PositivePreservingCheck* positive_check_;
    Log* log_;
    size_t current_step_;
    bool fail_flag_;
    T dt_max_;

    void calcualte_dt_min() const
    {
        if( std::abs(current_time_)<=std::numeric_limits<T>::epsilon() )
        {
            dt_min_ = static_cast<T>(16.0)*std::numeric_limits<T>::denorm_min();
        }
        else
        {
            dt_min_ = static_cast<T>(16.0)*std::numeric_limits<T>::epsilon();
        }
    }

};
}

#endif