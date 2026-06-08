#ifndef __TIME_STEPPER_TIME_STEP_ADAPTATION_TOLERANCE_H__
#define __TIME_STEPPER_TIME_STEP_ADAPTATION_TOLERANCE_H__

#include <cmath>
#include <utility>
#include <deque>
#include <time_stepper/detail/positive_preserving_dummy.h>
#include <time_stepper/time_step_adaptation.h>


namespace time_steppers
{

template<class VectorOperations, class Log, class PositivePreservingCheck = detail::positive_preserving_dummy<VectorOperations> >
class time_step_adaptation_tolerance: public time_step_adaptation<VectorOperations, Log, PositivePreservingCheck>
{
public:

    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
private:
    using parent_t = time_step_adaptation<VectorOperations, Log>;

public:
    time_step_adaptation_tolerance(VectorOperations* vec_ops_p, Log* log_p, std::pair<T,T> time_interval_p = {0.0,1.0},  T dt_p = 1.0, PositivePreservingCheck* positive_check_p = new detail::positive_preserving_dummy<VectorOperations>() ):
    parent_t(vec_ops_p, log_p, time_interval_p, dt_p, positive_check_p),
    relative_tolerance_(1.0e-6), //default values
    absolute_tolerance_(1.0e-12), //default values
    nofailed_(true),
    reject_step_(false),
    nfailed_(0),
    converged_(true),
    adaptation_name_("PID"),
    k_safe(0.9),
    number_of_ok_steps_(0)
    {
        // vec_ops_->init_vectors(delta_np1_, delta_n_, delta_nm1_);
        // vec_ops_->start_use_vectors(delta_np1_, delta_n_, delta_nm1_);
        threshold_ = absolute_tolerance_/relative_tolerance_;
        set_controller_constants();

    }
    
    ~time_step_adaptation_tolerance()
    {
        
        // vec_ops_->stop_use_vectors(delta_np1_, delta_n_, delta_nm1_);
        // vec_ops_->free_vectors(delta_np1_, delta_n_, delta_nm1_);
        // log_->info_f("~time_step_adaptation(): total number of failed restarts = %d", nfailed_);        
    }
        

    void set_parameters(const T relative_tolerance_p, const T absolute_tolerance_p = 1.0e-12)
    {
        relative_tolerance_ = relative_tolerance_p;
        absolute_tolerance_ = absolute_tolerance_p;
        if(relative_tolerance_ == 0.0)
        {
            throw std::logic_error("time_step_adaptation::set_parameters: relative_tolerance_ can't be zero.");
        }
        threshold_ = absolute_tolerance_/relative_tolerance_;
    }

    //accepts initial conditions and initial tanget
    void init_steps(const T_vec& x_p, const T_vec& fx_p)
    {
        
        T abst = std::min(dt_max_, dt_);
        T rh = 0;
        T dinum = 0.8*std::pow(relative_tolerance_, power_);
        T norm_x = vec_ops_->norm_l2(x_p);
        T norom_f0 = vec_ops_->norm_l2(fx_p); 
        rh = (norom_f0/std::max(norm_x, threshold_))/dinum;
        if(abst*rh>1.0)
        {
            abst = 1.0/rh;
        }
        dt_ = std::max(abst, dt_min_);
        dt_accepted_ = dt_;
        log_->info_f("time_step_adaptation::init_steps: rel tol = %le, abs tol = %le, initial timestep = %e", relative_tolerance_, absolute_tolerance_, dt_);

        auto norm_dtf0 = dt_*rh;
        deltas.push_back( norm_dtf0 );
        deltas.push_back( norm_dtf0 );
        deltas.push_back( norm_dtf0 );
        timesteps.push_back(dt_);
        timesteps.push_back(dt_);
        timesteps.push_back(dt_);
    }

    void pre_execte_step()const
    {
        parent_t::calcualte_dt_min();
        T dt_min_l = dt_min_*current_time_;
        dt_ = std::min(dt_max_, std::max(dt_min_l, dt_));  
        dt_accepted_ = dt_;
        nofailed_ = true; 
        reject_step_ = false;
        converged_ = true;
    }

    bool check_reject_step()const
    {
        return reject_step_;
    }

    void reject_step() const
    {
        reject_step_ = true;
    }

    bool estimate_timestep(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)
    {

        if(reject_step_)
        {
            number_of_ok_steps_ = 0;
            dt_ *= 0.5; //divide by 2 if the step failed and try again.

            if( std::abs(dt_-dt_min_) <= std::numeric_limits<T>::epsilon() )
            {
                log_->error_f("time_step_adaptation::reject_step: timestep %le is bellow minimum value %le on step %d at time %le. Termination.", dt_, dt_min_, current_step_, current_time_);
                fail_flag_ = true;
                converged_ = false;
            }             
        }
        else if(force_globalization_)
        {
            if( (number_of_ok_steps_++) > 5)
            {
                dt_ *= 1.5; //some primitive flobalization method.
                number_of_ok_steps_ = 0;
                log_->info_f("time_step_adaptation::force_globalization: timestep increased to %le", static_cast<double>(dt_) );
            }
        }
        else
        {
            estimate_error(x_p, x_new_p, f_err_p);
        }
        if(converged_ || reject_step_ || fail_flag_)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool is_adaptive()const
    {
        return true;
    }

    void set_adaptation_method(const std::string& adaptation_name_p, std::size_t ode_solver_order_p)
    {
        adaptation_name_ = adaptation_name_p;
        parent_t::set_ode_stepper_order(ode_solver_order_p);

        set_controller_constants();
    }   


protected:
    using parent_t::dt_;
    using parent_t::dt_accepted_;
    using parent_t::time_interval_;
    using parent_t::current_time_;
    using parent_t::current_step_;
    using parent_t::vec_ops_;
    using parent_t::log_;
    using parent_t::fail_flag_;
    using parent_t::dt_max_;
    using parent_t::dt_min_;
    using parent_t::base_ode_solver_order_;
    using parent_t::force_globalization_;
    const T power_ = 0.2; // matlab predefined
    T relative_tolerance_;
    T absolute_tolerance_;
    T threshold_;
    mutable bool nofailed_;
    mutable bool reject_step_;
    mutable bool converged_;
    std::size_t nfailed_;
    std::size_t number_of_ok_steps_;
    std::string adaptation_name_;
    std::deque<T> deltas;
    std::deque<T> timesteps;
    T k_safe;
    // controller constants:
    T alpha, beta, gamma, aa, bb;
    

private:
    
    void set_controller_constants()
    {
        if(adaptation_name_ == "I")
        {
            alpha = 1.0/(base_ode_solver_order_-1.0 + 1.0);
            beta = 0.0; gamma = 0.0; aa = 0.0; bb = 0.0;
        }
        else if(adaptation_name_ == "H211")
        {
            alpha = 0.25/(base_ode_solver_order_-1.0);
            beta = -alpha; gamma = 0.0; aa = -0.25; bb = 0.0;
        }
        else if(adaptation_name_ == "PC")
        {
            alpha = 2.0/(base_ode_solver_order_-1.0);
            beta = 1.0/(base_ode_solver_order_-1.0);
            gamma = 0.0; aa = 1.0; bb = 0.0;
        }        
        else if(adaptation_name_ == "PID")
        {
            alpha = 1.0/(18.0*(base_ode_solver_order_-1.0));
            beta = -1.0/(9.0*(base_ode_solver_order_-1.0));
            gamma = 1.0/(18.0*(base_ode_solver_order_-1.0));
            aa = 0.0; bb = 0.0;
        } 
        else if(adaptation_name_ == "H312")
        {
            alpha = 1.0/(8.0*(base_ode_solver_order_-1.0));
            beta = -1.0/(4.0*(base_ode_solver_order_-1.0));
            gamma = 1.0/(8.0*(base_ode_solver_order_-1.0));
            aa = -3.0/8.0; 
            bb = -1.0/8.0;
        }    
        else if(adaptation_name_ == "PPID")
        {
            alpha = 6.0/(20.0*(base_ode_solver_order_-1.0));
            beta = -1.0/(20.0*(base_ode_solver_order_-1.0));
            gamma = -5.0/(20.0*(base_ode_solver_order_-1.0));
            aa = 1.0; 
            bb = 0;
        }             
        else if(adaptation_name_ == "H321")
        {
            alpha = 1.0/(3.0*(base_ode_solver_order_-1.0));
            beta = -1.0/(18.0*(base_ode_solver_order_-1.0));
            gamma = -5.0/(18.0*(base_ode_solver_order_-1.0));
            aa = 5.0/6.0; 
            bb = 1.0/6.0;
        }
        else
        {
            throw std::logic_error("time_step_adaptation_tolerance::set_controller_constants: incorrect controller name.");
        }
    }

    void increase_step()
    {

    }

    void estimate_error(const T_vec& x_p, const T_vec& x_new_p, const T_vec& f_err_p)
    {
        reject_step_ = false;
        if(!parent_t::check_solution_consistency(x_new_p))
        {
            reject_step_ = true;
            fail_flag_ = true; //?
        }
        else
        {
            T norm_x = vec_ops_->norm_l2(x_p);
            T norm_x_new = vec_ops_->norm_l2(x_new_p);
            T norm_f_err = vec_ops_->norm_l2(f_err_p);

            T errwt = std::max(std::max(norm_x,norm_x_new), threshold_);
            T err = dt_*( norm_f_err/errwt);
            // std::cout << "err_norm = " << err_norm << std::endl;
            deltas.pop_front();
            deltas.push_back(err); // delta_{n-1} = deltas[0], delta_{n} = delta[1], delta_{n+1} = delta[2]
            T eps = relative_tolerance_;
            // std::cout << "0: " << deltas[0] << ", 1: " << deltas[1] << ", 2: " << deltas[2] << std::endl;
            auto m1 = std::pow(eps/deltas[2], alpha);
            auto m2 = std::pow(deltas[1]/eps, beta);
            auto m3 = std::pow(eps/deltas[0], gamma);
            auto m4 = std::pow(timesteps[2]/timesteps[1], aa);
            auto m5 = std::pow(timesteps[1]/timesteps[0], bb);
            dt_ = k_safe*dt_*m1*m2*m3*m4*m5;
            timesteps.pop_front();
            timesteps.push_back(dt_);
            reject_step_ = false;
            fail_flag_ = false;
            converged_ = true;
            dt_accepted_ = dt_;
        }

    }


};



}



#endif