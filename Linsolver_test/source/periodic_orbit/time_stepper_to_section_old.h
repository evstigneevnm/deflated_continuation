// 230712 TO BE DELETED! Keep for now

#ifndef __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H
#define __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H


#include <utility>
#include <tuple>


namespace periodic_orbit
{
namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, class SingleStepper, class Hyperplane, class Log>
class time_stepper_to_section
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_stepper_to_section(VectorOperations* vec_ops_, NonlinearOperator* nonlin_op_p, SingleStepper* stepper_p, Hyperplane* hyperplane_p, Log* log_, std::pair<T,size_t> bisection_params_p = {1.0e-12,100}, unsigned int skip_p = 1000):
    vec_ops(vec_ops_), 
    nonlin_op_(nonlin_op_p), 
    stepper_(stepper_p), 
    hyperplane_(hyperplane_p),
    log(log_), 
    bisection_tolerance_(bisection_params_p.first),
    bisection_iters_(bisection_params_p.second),
    skip_(skip_p)
    {
        vec_ops->init_vector(v_in); vec_ops->start_use_vector(v_in);
        vec_ops->init_vector(v_out); vec_ops->start_use_vector(v_out);
        vec_ops->init_vector(v_helper); vec_ops->start_use_vector(v_helper);
        bisection_tolerance_ *= vec_ops->get_size(v_in);
    }
    ~time_stepper_to_section()
    {
        
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
        stepper_->set_parameter(param);
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
    void set_time_before_section(const T time_b4_section_check_p)
    {
        time_b4_section_check_ = time_b4_section_check_p;
    }

    void get_results(T_vec& x_)const
    {
        vec_ops->assign(v_out, x_);
    }

    // execution in a single call
    // x_ <- initial conditions and returing result
    // param_p:=lambda <- parameter for F(x,lambda)
    // time_interval_p is a caluclating interval in the form {Tstart,Tend}, Tstart>Tend
    void execute(T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p,  const T time_b4_section_check_p = 0.0)
    {
        set_time_before_section(time_b4_section_check_p);
        set_initial_conditions(x_p);
        stepper_->set_initial_time_interval(time_interval_p);
        set_parameter(param_p);
        execute();
        get_results(x_p);
    }

    void execute()
    {
        // check logic
        if(!param_set) throw std::logic_error("time_stepper_to_section::execute: parameter value not set.");
        if(!initials_set) throw std::logic_error("time_stepper_to_section::execute: initial conditions not set.");
        
        bool finish = false;
        bool finish_besection = false;
        log->info_f("time_stepper_to_section::execute: starting time stepper_ execution with total time = %f and parameter value = %e", time, param);
        T bif_priv = 0.0;
        
        stepper_->init_steps(v_in);
        
        {
            auto dt = stepper_->get_dt();
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op_->norm_bifurcation_diagram(v_in, bif_norms_at_t_);
            bif_norms_at_t_.push_back(dt);
            solution_norms.push_back(bif_norms_at_t_);
        }
        while(!finish)
        {
            finish = stepper_->execute(v_in, v_out);
            auto iteraiton = stepper_->get_iteration();
            auto dt = stepper_->get_dt();
            //TODO:
            // add intersection check and bisection callback to time_step_adaptation?
            T dt_mod = dt;
            if(simulated_time >= time_b4_section_check_)
            {
                if(periodic_time == 0.0)
                {
                    if(time_b4_section_check_>0.0)
                    {
                        hyperplane_->update(time_b4_section_check_, v_in, param);
                    }
                    std::vector<T> bif_norms_at_t_;
                    bif_norms_at_t_.push_back(periodic_time);
                    nonlin_op_->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
                    bif_norms_at_t_.push_back(dt_mod);
                    solution_period_estmate_norms.push_back(bif_norms_at_t_);
                }
                if(hyperplane_->is_crossed_in_normal_direction(v_in, v_out))
                {
                    finish_besection = bisect_intersection(v_in, dt_mod, v_out);
                }
                periodic_time += dt_mod;
                std::vector<T> bif_norms_at_t_;
                bif_norms_at_t_.push_back(periodic_time);
                nonlin_op_->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
                bif_norms_at_t_.push_back(dt_mod);
                solution_period_estmate_norms.push_back(bif_norms_at_t_);
            }
            T simulated_time_p = simulated_time;
            simulated_time = stepper_->get_time();
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(simulated_time);
            nonlin_op_->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
            bif_norms_at_t_.push_back(simulated_time-simulated_time_p);
            solution_norms.push_back(bif_norms_at_t_);
            T bif_now = bif_norms_at_t_.at(1);
            if(iteraiton%skip_ == 0)
            {
                T solution_quality = nonlin_op_->check_solution_quality(v_out);
                log->info_f("time_stepper_to_section::execute: step %d, time %.2f, dt %.2le, norm %.2le, d_norm %.2le, quality %.2le", iteraiton, simulated_time, dt, bif_now, (bif_now - bif_priv), solution_quality );
                bif_priv = bif_now;
            }
            if(finish_besection)
            {
                T solution_quality = nonlin_op_->check_solution_quality(v_out);
                log->info_f("time_stepper_to_section::execute bisection finished: step %d, time %.2f, dt %.2le, norm %.2le, d_norm %.2le, quality %.2le", iteraiton, simulated_time, dt_mod, bif_now, (bif_now - bif_priv), solution_quality );                
                bif_priv = bif_now;
                finish = true;
            }
            vec_ops->assign(v_out, v_in);
        } 

    }

    T get_simulated_time()const
    {
        return simulated_time;
    }
    T get_period_estmate_time()const
    {
        return periodic_time;
    }    
    void reset()
    {
        periodic_time = T(0.0);
        simulated_time = T(0.0);
        solution_norms.clear();
    }
    void save_norms(const std::string& file_name_)const
    {
        std::ofstream file_;
        file_ = std::ofstream(file_name_.c_str(), std::ofstream::out);
        if(!file_) throw std::runtime_error("time_stepper_to_section::save_norms: failed to open file " + file_name_ + " for output.");

        for(auto &x_: solution_norms)
        {
            for(auto &y_: x_)
            {
                if(!(file_ << std::scientific << y_ << " ")) throw std::runtime_error("time_stepper_to_section::save_norms: failed to write to " + file_name_);
            }
            if(!(file_ << std::endl)) throw std::runtime_error("time_stepper_to_section::save_norms: failed to write to " + file_name_);
        }
        file_.close();

    }
    void save_period_estmate_norms(const std::string& file_name_)const
    {
        std::ofstream file_;
        file_ = std::ofstream(file_name_.c_str(), std::ofstream::out);
        if(!file_) throw std::runtime_error("time_stepper_to_section::save_period_estmate_norms: failed to open file " + file_name_ + " for output.");

        for(auto &x_: solution_period_estmate_norms)
        {
            for(auto &y_: x_)
            {
                if(!(file_ << std::scientific << y_ << " ")) throw std::runtime_error("time_stepper_to_section::save_period_estmate_norms: failed to write to " + file_name_);
            }
            if(!(file_ << std::endl)) throw std::runtime_error("time_stepper_to_section::save_period_estmate_norms: failed to write to " + file_name_);
        }
        file_.close();
    }

private:
    

    
    T time;
    T param;
    T simulated_time = T(0.0);
    T periodic_time = T(0.0);
    std::vector< std::vector<T> > solution_norms;
    std::vector< std::vector<T> > solution_period_estmate_norms;
    VectorOperations* vec_ops;
    SingleStepper* stepper_;
    NonlinearOperator* nonlin_op_;
    Hyperplane* hyperplane_;
    Log* log;
    bool param_set = false;
    bool time_set = false;
    bool initials_set = false;
    unsigned int skip_;
    T bisection_tolerance_;
    size_t bisection_iters_;
    T_vec v_in;
    T_vec v_out;
    T_vec v_helper;
    T time_b4_section_check_;

    bool bisect_intersection(const T_vec& v_in, T& dt_mod, T_vec& v_out)
    {
        bool ret_flag = false;
        T error = 1.0;
        T distance = 1.0;
        size_t iters = 0;
        while((std::abs(error) > bisection_tolerance_)&&(iters < bisection_iters_))
        {
            std::tie(error, distance) = hyperplane_->intersection(v_in, v_out);
            dt_mod = dt_mod - error*dt_mod/distance;
            stepper_->force_undo_step();
            stepper_->execute_forced_dt(dt_mod, v_in, v_out);
            ++iters;
        }
        if(std::abs(error) <= bisection_tolerance_)
        {
            ret_flag = true;
            log->info_f("time_stepper::bisect_intersection: intersection tolerance %.2le on %i iteration with error %.2le using timestep %.2le ", bisection_tolerance_, iters, error, dt_mod);
        }  
        else
        {
            log->warning_f("time_stepper::bisect_intersection: failed to obtain intersection on %i iteration with error %.2le using timestep %.2le ",  iters, error, dt_mod);
        }      
        return ret_flag;
    }




};

}
}

#endif // __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H

