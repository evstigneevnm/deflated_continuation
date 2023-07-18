#ifndef __TIME_STEPPER_TIME_STEP_ADAPTATION_CONSTANT_H__
#define __TIME_STEPPER_TIME_STEP_ADAPTATION_CONSTANT_H__

#include <utility>
#include <time_stepper/detail/positive_preserving_dummy.h>
#include <time_stepper/time_step_adaptation.h>


namespace time_steppers
{

template<class VectorOperations, class Log, class PositivePreservingCheck = detail::positive_preserving_dummy<VectorOperations> >
class time_step_adaptation_constant: public time_step_adaptation<VectorOperations, Log, PositivePreservingCheck>
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
private:
    using parent_t = time_step_adaptation<VectorOperations, Log>;

public:
    time_step_adaptation_constant(VectorOperations* vec_ops_p, Log* log_p, std::pair<T,T> time_interval_p = {0,1},  T dt_p = 0.0, PositivePreservingCheck* positive_check_p = new detail::positive_preserving_dummy<VectorOperations>() ):
    parent_t(vec_ops_p, log_p, time_interval_p, dt_p, positive_check_p)
    {}
    
    ~time_step_adaptation_constant()
    {}
        

    void init_steps(const T_vec& x_p, const T_vec& fx_p)
    {
        //do nothing here
    }
    void pre_execte_step()const
    {
        //do nothing here
    }
    bool check_reject_step()const
    {
        return false;
    }
    bool estimate_timestep(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)
    {
        //this implements timestep strategy adaptation. It is void for the constant time step.
        if(vec_ops_->check_is_valid_number(x_new_p))
        {
            return true;     
        }
        else
        {
            log_->error_f("time_step_adaptation::estimate_timestep: returned nan at step %i at time %e for dt %e", current_step_, current_time_, dt_);
            bool fail_flag_ = true;
            return false;
        }
    }
    bool is_adaptive()const
    {
        return false;
    }

protected:
    using parent_t::dt_;
    using parent_t::time_interval_;
    using parent_t::current_time_;
    using parent_t::current_step_;
    using parent_t::vec_ops_;
    using parent_t::log_;
    using parent_t::fail_flag_;
    using parent_t::dt_max_;
    using parent_t::dt_min_;


};



}



#endif