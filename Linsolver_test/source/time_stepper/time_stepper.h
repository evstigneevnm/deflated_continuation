#ifndef __TIME_STEPPER_H__
#define __TIME_STEPPER_H__

#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, class SingleStepper, class Log>
class time_stepper
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_stepper(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_, SingleStepper* stepper_, Log* log_):
    vec_ops(vec_ops_), nonlin_op(nonlin_op_), stepper(stepper_), log(log_)
    {
        vec_ops->init_vector(v_in); vec_ops->start_use_vector(v_in);
        vec_ops->init_vector(v_out); vec_ops->start_use_vector(v_out);
        vec_ops->init_vector(v_helper); vec_ops->start_use_vector(v_helper);
    }
    ~time_stepper()
    {
        
        vec_ops->stop_use_vector(v_in); vec_ops->free_vector(v_in);
        vec_ops->stop_use_vector(v_out); vec_ops->free_vector(v_out);
        vec_ops->stop_use_vector(v_helper); vec_ops->free_vector(v_helper);

    }
    
    void set_skip(unsigned int skip_)
    {
        skip = skip_;
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
        // nonlin_op->randomize_vector(v_helper);
        // vec_ops->add_mul(perturbations_, v_helper, v_in);
        vec_ops->assign(v_in, v_out); //as initialization
        initials_set = true;
    }

    void get_results(T_vec& x_)
    {
        vec_ops->assign(v_out, x_);
    }


    void execute()
    {
        // check logic
        if(!param_set) throw std::logic_error("time_stepper::execute: parameter value not set.");
        if(!initials_set) throw std::logic_error("time_stepper::execute: initial conditions not set.");
        
        {
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op->norm_bifurcation_diagram(v_in, bif_norms_at_t_);
            solution_norms.push_back(bif_norms_at_t_);
        }

        bool finish = false;
        log->info_f("time_stepper::execute: starting time stepper execution with total time = %f and parameter value = %e", time, param);
        T bif_priv = 0.0;
        
        stepper->init_steps(v_in);

        while(!finish)
        {
            stepper->pre_execte_step();
            finish = stepper->execute(v_in, v_out);

            auto iteraiton = stepper->get_iteration();
            T norm_out = vec_ops->norm(v_out);
            if(std::isnan(norm_out))
            {
                throw std::runtime_error("time_stepper::execute: nan returned at iteration " + std::to_string(iteraiton));
            }
            auto dt = stepper->get_dt();
            simulated_time = stepper->get_time();
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
            
            solution_norms.push_back(bif_norms_at_t_);
            T bif_now = bif_norms_at_t_.back();
            if(iteraiton%skip == 0)
            {
                log->info_f("time_stepper::execute: step %d, time %.2f, dt %.2e, norm %.2e, d_norm %.2e", iteraiton, simulated_time, dt, bif_now, (bif_now - bif_priv) );
                bif_priv = bif_now;
            }
            vec_ops->assign(v_out, v_in);
        } 

    }

    void reset()
    {
        simulated_time = T(0.0);
        solution_norms.clear();
    }
    void save_norms(const std::string& file_name_)
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
    T time;
    T param;
    T simulated_time = T(0.0);
    std::vector< std::vector<T> > solution_norms;
    VectorOperations* vec_ops;
    SingleStepper* stepper;
    NonlinearOperator* nonlin_op;
    Log* log;
    bool param_set = false;
    bool time_set = false;
    bool initials_set = false;
    unsigned int skip = 1000;

    T_vec v_in = nullptr;
    T_vec v_out = nullptr;
    T_vec v_helper = nullptr;


};
}


#endif