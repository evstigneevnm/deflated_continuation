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
    dt_accepted_(dt_p),
    dt_initial_(dt_p),
    current_step_(0),
    fail_flag_(false),
    positive_check_(positive_check_p)
    {
        calculate_initial_dt();
    }
    
    virtual ~time_step_adaptation()
    {}
      
    
    void set_initial_time_interval(const std::pair<T,T> time_interval_p)
    {
        time_interval_=time_interval_p;
        calculate_initial_dt();
    }
    void set_initial_dt(const T dt_p)
    {
        dt_ = dt_p;
        dt_accepted_ = dt_p;
        dt_initial_ = dt_p;
    }

    T get_dt()const
    {
        return dt_accepted_;
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
        previous_time_ = current_time_;
        current_time_ += dt_accepted_;
        previous_step_ = current_step_;
        current_step_++;
        if((current_time_+dt_accepted_)>time_interval_.second)
        {
            dt_ = time_interval_.second - current_time_;
        }
        previous_dt_accepted_ = dt_accepted_;
        return *this;
    }

    void force_set_timestep(const T dt_p)
    {
        dt_accepted_ = dt_p;
        dt_ = dt_p;
    }  

    void force_undo_step()
    {
        current_time_ =  previous_time_;
        current_step_ = previous_step_;
        dt_accepted_ = previous_dt_accepted_;
    }


    void reset_steps()
    {
        current_time_ = 0;
        current_step_ = 0;
        dt_ = dt_initial_;
        dt_accepted_ = dt_initial_;
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
    virtual bool is_adaptive()const=0;

protected:
    mutable T dt_;
    mutable T dt_accepted_;
    mutable T dt_min_;
    mutable T previous_dt_accepted_;
    T dt_initial_;
    std::pair<T,T> time_interval_;
    mutable T current_time_;
    VectorOperations* vec_ops_;
    PositivePreservingCheck* positive_check_;
    Log* log_;
    size_t current_step_;
    bool fail_flag_;
    T dt_max_;

    mutable T previous_time_;
    size_t previous_step_;

    void calculate_initial_dt()
    {
        current_time_ = time_interval_.first;
        T final_time = time_interval_.second;
        dt_max_ = 0.1*(final_time - current_time_); //default in matlab

        dt_ = ((final_time- current_time_)>3.0*dt_initial_)?dt_initial_:(final_time- current_time_)/3.0;
        calcualte_dt_min();        
    }

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