#ifndef __TIME_STEPPER_TIME_STEP_ADAPTATION_ERROR_CONTROL_H__
#define __TIME_STEPPER_TIME_STEP_ADAPTATION_ERROR_CONTROL_H__

#include <utility>
#include <time_stepper/detail/positive_preserving_dummy.h>
#include <time_stepper/time_step_adaptation.h>


namespace time_steppers
{

template<class VectorOperations, class Log, class PositivePreservingCheck = detail::positive_preserving_dummy<VectorOperations> >
class time_step_adaptation_error_control: public time_step_adaptation<VectorOperations, Log, PositivePreservingCheck>
{
public:

    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
private:
    using parent_t = time_step_adaptation<VectorOperations, Log>;

public:
    time_step_adaptation_error_control(VectorOperations* vec_ops_p, Log* log_p, std::pair<T,T> time_interval_p = {0.0,1.0},  T dt_p = 0.1, PositivePreservingCheck* positive_check_p = new detail::positive_preserving_dummy<VectorOperations>() ):
    parent_t(vec_ops_p, log_p, time_interval_p, dt_p, positive_check_p),
    relative_tolerance_(1.0e-6), //default values
    absolute_tolerance_(1.0e-12), //default values
    norm_control_(false),
    nofailed_(true),
    reject_step_(false),
    nfailed_(0),
    converged_(true),
    dt_prev_(dt_p)
    {
        vec_ops_->init_vectors(f_help_, x_help_); vec_ops_->start_use_vectors(f_help_, x_help_);
        threshold_ = absolute_tolerance_/relative_tolerance_;
    }
    
    ~time_step_adaptation_error_control()
    {
        vec_ops_->stop_use_vectors(f_help_, x_help_); vec_ops_->free_vectors(f_help_, x_help_);
    }
        

    void set_parameters(const T relative_tolerance_p, const T absolute_tolerance_p, bool norm_control_ = false)
    {
        relative_tolerance_ = relative_tolerance_p;
        absolute_tolerance_ = absolute_tolerance_p;
        if(relative_tolerance_ == 0.0)
        {
            throw std::logic_error("time_step_adaptation::set_parameters: relative_tolerance_ can't be zero.");
        }
        threshold_ = absolute_tolerance_/relative_tolerance_;
    }

    void init_steps(const T_vec& x_p, const T_vec& fx_p)
    {
        
        T abst = std::min(dt_max_, dt_);
        T rh = 0;
        T dinum = 0.8*std::pow(relative_tolerance_, power_);
        if(norm_control_)
        {
            T norm_x = vec_ops_->norm(x_p);
            T norom_f0 = vec_ops_->norm(fx_p); 
            rh = (norom_f0/std::max(norm_x, threshold_))/dinum;
        }
        else
        {
            vec_ops_->make_abs_copy(x_p, x_help_); 
            vec_ops_->add_mul_scalar(threshold_, 1.0, x_help_); // |x_p|+threshold_
            vec_ops_->div_pointwise(1.0,fx_p,1.0,x_help_,f_help_); // f_help = fx_p/ (|x_p| + threshold)
            T norm_f_inf_sc = vec_ops_->norm_inf(f_help_);
            rh = norm_f_inf_sc/dinum;
        }
        if(abst*rh>1.0)
        {
            abst = 1.0/rh;
        }
        dt_ = std::max(abst, dt_min_);
        log_->info_f("time_step_adaptation::init_steps: rel tol = %le, abs tol = %le, initial timestep = %e", relative_tolerance_, absolute_tolerance_, dt_);
    }


    void pre_execte_step()const
    {
        parent_t::calcualte_dt_min();
        T dt_min_l = dt_min_*current_time_;
        dt_ = std::min(dt_max_, std::max(dt_min_l, dt_));  
        nofailed_ = true; 
        reject_step_ = false;
        converged_ = true;
    }
    bool check_reject_step()const
    {
        return reject_step_;
    }


    bool estimate_timestep(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)
    {

        estimate_error_matlab(x_p, x_new_p, f_err_p);
        if(converged_||reject_step_)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


protected:
    using parent_t::dt_;
    T dt_prev_;
    using parent_t::time_interval_;
    using parent_t::current_time_;
    using parent_t::current_step_;
    using parent_t::vec_ops_;
    using parent_t::log_;
    using parent_t::fail_flag_;
    using parent_t::dt_max_;
    using parent_t::dt_min_;
    bool norm_control_; //equivalent to the matlab's definition: Control error relative to the norm of the solution. When NormControl is 'on' (true), the solvers control the error e at each step using the norm of the solution rather than its absolute value: norm(e(i)) <= max(RelTol*norm(y(i)),AbsTol(i)). If the NormControl if 'off' (false), the solvers control the error e using the absolute value: 
    T power_ = 0.2; // matlab predefined
    T relative_tolerance_;
    T absolute_tolerance_;
    T threshold_;
    T_vec f_help_;
    T_vec x_help_;
    mutable bool nofailed_;
    mutable bool reject_step_;
    mutable bool converged_;
    size_t nfailed_;


private:

    void estimate_error_matlab(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)
    {
        T err = 0.0;
        if(!parent_t::check_solution_consistency(x_new_p))
        {
            reject_step_ = true;
            fail_flag_ = true; //?
        }
        if(norm_control_)
        {
            T norm_x = vec_ops_->norm(x_p);
            T norm_x_new = vec_ops_->norm(x_new_p);
            T norm_f_err = vec_ops_->norm(f_err_p);

            T errwt = std::max(std::max(norm_x,norm_x_new),threshold_);
            err = dt_*( norm_f_err/errwt);
        }
        else
        {
            vec_ops_->make_abs_copy(x_p, x_help_); 
            vec_ops_->make_abs_copy(x_new_p, f_help_); 
            vec_ops_->max_pointwise(threshold_, f_help_, x_help_);
            vec_ops_->div_pointwise(1.0, f_err_p, 1.0, x_help_, f_help_);
            T norm_f_inf_sc = vec_ops_->norm_inf(f_help_);
            err = dt_*norm_f_inf_sc;
        }
        if(err>relative_tolerance_)
        {
            converged_ = false;
            nfailed_ = nfailed_ + 1;  
            if( std::abs(dt_-dt_min_) <= std::numeric_limits<T>::epsilon() )
            {
                log_->error_f("time_step_adaptation::estimate_error: timestep %le is bellow minimum value %le on step %d at time %le. Termination.", dt_, dt_min_, current_step_, current_time_);
                fail_flag_ = true;
            }   
            if(nofailed_)
            {
                nofailed_ = false;
                if(reject_step_)
                {
                    dt_ = std::max(dt_min_, static_cast<T>(0.5)*dt_);      
                }
                else
                {
                    dt_ = std::max(dt_min_, dt_*std::max(static_cast<T>(0.1), static_cast<T>(0.8)*std::pow( (relative_tolerance_/err), power_) ) );
                }
            }
            else
            {
                dt_ = std::max(dt_min_, static_cast<T>(0.5)*dt_);    
            }
                        

        }
        else
        {
            converged_ = true;
            dt_prev_ = dt_;
            if(nofailed_)
            {
                T temp = static_cast<T>(1.25)*std::pow((err/relative_tolerance_), power_);
                if(temp > 0.2) //matlab predefined
                {
                    dt_ = dt_/temp;
                }
                else
                {
                    dt_ = 5.0*dt_;
                }
            }
        }

    }


};



}



#endif