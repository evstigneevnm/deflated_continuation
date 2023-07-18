#ifndef __TIME_STEPPER_H__
#define __TIME_STEPPER_H__

#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <type_traits>
#include "detail/external_management_dummy.h"

namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, class SingleStepper, class Log, class ExternalManagement = time_steppers::detail::external_management_dummy<VectorOperations, SingleStepper> >
class time_stepper
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_stepper(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_p, SingleStepper* stepper_, Log* log_, ExternalManagement* external_management_p = new time_steppers::detail::external_management_dummy<VectorOperations, SingleStepper>(), unsigned int skip_p = 1000):
    vec_ops(vec_ops_), nonlin_op_(nonlin_op_p), stepper(stepper_), log(log_), skip_(skip_p),external_management_(external_management_p)
    {
        vec_ops->init_vector(v_in); vec_ops->start_use_vector(v_in);
        vec_ops->init_vector(v_out); vec_ops->start_use_vector(v_out);
        vec_ops->init_vector(v_helper); vec_ops->start_use_vector(v_helper);
    }
    ~time_stepper()
    {
        if(std::is_same<ExternalManagement, time_steppers::detail::external_management_dummy<VectorOperations, SingleStepper>>::value )
        {
            delete external_management_; //if it is a default dummy structure, then we must delete it
        }
        vec_ops->stop_use_vector(v_in); vec_ops->free_vector(v_in);
        vec_ops->stop_use_vector(v_out); vec_ops->free_vector(v_out);
        vec_ops->stop_use_vector(v_helper); vec_ops->free_vector(v_helper);
    }
    
    void set_skip(unsigned int skip_p)
    {
        skip_ = skip_p;
    }
    void set_parameter(const T param_)
    {
        param = param_;
        stepper->set_parameter(param);
        param_set = true;
    }
    
    void set_initial_conditions(const T_vec& x_, T perturbations_ = 0.0)
    {
        vec_ops->assign(x_, v_in);
        // nonlin_op_->randomize_vector(v_helper);
        // vec_ops->add_mul(perturbations_, v_helper, v_in);
        vec_ops->assign(v_in, v_out); //as initialization
        initials_set = true;
    }

    void get_results(T_vec& x_)
    {
        vec_ops->assign(v_out, x_);
    }

    T get_simulated_time()const
    {
        return simulated_time;
    }
    // execution in a single call
    // x_ <- initial conditions and returing result
    // param_p:=lambda <- parameter for F(x,lambda)
    // time_interval_p is a caluclating interval in the form {Tstart,Tend}, Tstart>Tend
    void execute(T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p )
    {
        set_initial_conditions(x_p);
        stepper->set_initial_time_interval(time_interval_p);
        set_parameter(param_p);
        execute();
        get_results(x_p);
    }

    void execute()
    {
        // check logic
        if(!param_set) throw std::logic_error("time_stepper::execute: parameter value not set.");
        if(!initials_set) throw std::logic_error("time_stepper::execute: initial conditions not set.");
        
        bool finish = false;
        bool finish_management = false;
        log->info_f("time_stepper::execute: starting time stepper execution, parameter value = %e", param);
        T bif_priv = 0.0;
        
        stepper->init_steps(v_in);
        
        {
            auto dt = stepper->get_dt();
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op_->norm_bifurcation_diagram(v_in, bif_norms_at_t_);
            bif_norms_at_t_.push_back(dt);
            solution_norms.push_back(bif_norms_at_t_);
        }

        while(!finish && !finish_management)
        {
            finish = stepper->execute(v_in, v_out);
            auto iteraiton = stepper->get_iteration();
            auto dt = stepper->get_dt();
            T simulated_time_p = simulated_time;

            //can be used for the external control of the time stepping process
            finish_management = external_management_->apply(simulated_time, dt, v_in, v_out);

            simulated_time = stepper->get_time();
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op_->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
            bif_norms_at_t_.push_back(simulated_time-simulated_time_p);
            solution_norms.push_back(bif_norms_at_t_);
            T bif_now = bif_norms_at_t_.at(2);
            if(iteraiton%skip_ == 0)
            {
                T solution_quality = nonlin_op_->check_solution_quality(v_out);
                log->info_f("time_stepper::execute: step %d, time %.2f, dt %.2le, norm %.2le, d_norm %.2le, quality %.2le", iteraiton, simulated_time, dt, bif_now, (bif_now - bif_priv), solution_quality );
                bif_priv = bif_now;
            }
            if(finish_management||finish)
            {
                T solution_quality = nonlin_op_->check_solution_quality(v_out);
                log->info_f("time_stepper::execute finished: step %d, time %.2f, dt %.2le, norm %.2le, d_norm %.2le, quality %.2le", iteraiton, simulated_time, dt, bif_now, (bif_now - bif_priv), solution_quality );                
                bif_priv = bif_now;
                finish = true;
            }

            vec_ops->assign(v_out, v_in);
        } 

    }

    void reset()
    {
        stepper->reset_steps();
        simulated_time = T(0.0);
        solution_norms.clear();
    }
    void save_norms(const std::string& file_name_)const
    {
        std::ofstream file_;
        file_ = std::ofstream(file_name_.c_str(), std::ofstream::out);
        if(!file_) throw std::runtime_error("time_stepper::save_norms: failed to open file " + file_name_ + " for output.");

        for(auto &x_: solution_norms)
        {
            for(auto &y_: x_)
            {
                if(!(file_ << std::scientific << y_ << " ")) throw std::runtime_error("time_stepper::save_norms: failed to write to " + file_name_);
            }
            if(!(file_ << std::endl)) throw std::runtime_error("time_stepper::save_norms: failed to write to " + file_name_);
        }
        file_.close();

    }

private:
    T param;
    T simulated_time = T(0.0);
    std::vector< std::vector<T> > solution_norms;
    VectorOperations* vec_ops;
    SingleStepper* stepper;
    NonlinearOperator* nonlin_op_;
    ExternalManagement* external_management_;
    Log* log;
    bool param_set = false;
    bool time_set = false;
    bool initials_set = false;
    unsigned int skip_;

    T_vec v_in;
    T_vec v_out;
    T_vec v_helper;


};
}


#endif