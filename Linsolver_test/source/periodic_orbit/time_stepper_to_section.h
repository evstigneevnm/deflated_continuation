#ifndef __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H
#define __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H


#include <utility>
#include <tuple>
#include <time_stepper/time_stepper.h>

namespace periodic_orbit
{
namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, class SingleStepper, class Hyperplane, class Log>
class time_stepper_to_section
{
private:
    struct external_management
    {
        using T = typename VectorOperations::scalar_type;
        using T_vec = typename VectorOperations::vector_type;   

        external_management(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, SingleStepper* stepper_p, Log* log_p, std::pair<T,size_t> bisection_params_p = {1.0e-12, 200}):
        periodic_time_(0.0),
        time_b4_section_check_(0.0),
        vec_ops_(vec_ops_p), 
        nonlin_op_(nonlin_op_p), 
        stepper_(stepper_p), 
        log_(log_p), 
        bisection_tolerance_(bisection_params_p.first),
        bisection_iters_(bisection_params_p.second),
        hyperplane_(nullptr)
        {}

        ~external_management()
        {}

        void set_hyperplane(Hyperplane* hyperplane_p)
        {
            hyperplane_ = hyperplane_p;
        }

        //main callback function
        bool apply(T simulated_time, T& dt, T_vec& v_in, T_vec& v_out)
        {
            if(hyperplane_ == nullptr)
            {
                throw std::logic_error("time_stepper_to_section::external_management: hyperplane is not set!");
            }
            bool finish_besection = false;
            if(simulated_time >= time_b4_section_check_)
            {
                if(periodic_time_ == 0.0)
                {
                    if(time_b4_section_check_>0.0)
                    {
                        hyperplane_->update(time_b4_section_check_, v_in, param_);
                    }
                    std::vector<T> bif_norms_at_t_;
                    bif_norms_at_t_.push_back(periodic_time_);
                    nonlin_op_->norm_bifurcation_diagram(v_in, bif_norms_at_t_);
                    bif_norms_at_t_.push_back(0.0);
                    solution_period_estmate_norms.push_back(bif_norms_at_t_);
                }
                if(hyperplane_->is_crossed_in_normal_direction(v_in, v_out))
                {
                    finish_besection = bisect_intersection(v_in, dt, v_out);
                }
                periodic_time_ += dt;
                std::vector<T> bif_norms_at_t_;
                bif_norms_at_t_.push_back(periodic_time_);
                nonlin_op_->norm_bifurcation_diagram(v_out, bif_norms_at_t_);
                bif_norms_at_t_.push_back(dt);
                solution_period_estmate_norms.push_back(bif_norms_at_t_);
            } 
            return finish_besection;             
        }

        void set_time_before_section(const T time_b4_section_check_p)
        {
            time_b4_section_check_ = time_b4_section_check_p;
        }
        
        T get_period_estmate_time()const
        {
            return periodic_time_;
        }
        void set_parameter(const T param_p)
        {
            param_ = param_p;
        }
        void reset()
        {
            time_b4_section_check_ = 0;
            solution_period_estmate_norms.resize(0);
            periodic_time_ = 0;

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
        VectorOperations* vec_ops_;
        Hyperplane* hyperplane_;
        SingleStepper* stepper_;
        NonlinearOperator* nonlin_op_;
        Log* log_;
        T param_;
        T bisection_tolerance_;
        size_t bisection_iters_;
        T time_b4_section_check_;
        T periodic_time_;
        std::vector< std::vector<T> > solution_period_estmate_norms;

        bool bisect_intersection(const T_vec& v_in, T& dt_mod, T_vec& v_out)
        {
            bool ret_flag = false;
            T error = 1.0;
            T distance = 1.0;
            size_t iters = 0;
            T size_l2_correction = vec_ops_->get_l2_size();

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
                log_->info_f("time_stepper_to_section::external_management: intersection tolerance %.2le on %i iteration with error %.2le using timestep %.2le", bisection_tolerance_, iters, error, dt_mod);
            }  
            else if(std::abs(error) <= bisection_tolerance_*size_l2_correction)
            {
                ret_flag = true;
                log_->warning_f("time_stepper_to_section::external_management: intersection ACCEPTED RELAXED tolerance %.2le on %i iteration with error %.2le using timestep %.2le", bisection_tolerance_*size_l2_correction, iters, error, dt_mod);                
            }
            else
            {
                log_->warning_f("time_stepper_to_section::external_management: failed to obtain intersection tolerance %.2le on %i iteration with error %.2le using timestep %.2le",  bisection_tolerance_, iters, error, dt_mod);
            }      
            return ret_flag;
        }        

    };


    using external_management_t = external_management;

    using time_stepper_t = ::time_steppers::time_stepper<VectorOperations, NonlinearOperator, SingleStepper, Log, external_management_t>;

    external_management_t* em = nullptr;
    time_stepper_t* time_stepper = nullptr;

public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    time_stepper_to_section(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, SingleStepper* stepper_p, Log* log_p, std::pair<T,size_t> bisection_params_p = {1.0e-12,100}, unsigned int skip_p = 1000)
    {
        em = new external_management_t(vec_ops_p, nonlin_op_p, stepper_p, log_p, bisection_params_p); 
        time_stepper = new time_stepper_t(vec_ops_p, nonlin_op_p, stepper_p, log_p, em, skip_p);

    }
    ~time_stepper_to_section()
    {
        delete time_stepper;
        delete em;
    }
    
    void set_skip(unsigned int skip_p)
    {
        time_stepper->set_skip(skip_p);
    }
    void set_parameter(const T param_p)
    {
        time_stepper->set_parameter(param_p);
        em->set_parameter(param_p);
    }
    
    void set_initial_conditions(const T_vec& x_, T perturbations_ = 0.0)
    {
        time_stepper->set_initial_conditions(x_, perturbations_);
    }
    void set_time_before_section(const T time_b4_section_check_p)
    {
        em->set_time_before_section(time_b4_section_check_p);
    }

    void get_results(T_vec& x_)const
    {
        time_stepper->get_results(x_);
    }

    void set_hyperplane(Hyperplane* hyperplane_p)
    {
        em->set_hyperplane(hyperplane_p);
    }

    // execution in a single call
    // hyperplane_p <- target hyperplane used for section.
    // x_p <- initial conditions and returing result
    // param_p:=lambda <- parameter for F(x,lambda)
    // time_interval_p is a caluclating interval in the form {Tstart,Tend}, Tstart>Tend
    void execute(Hyperplane* hyperplane_p, T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p,  const T time_b4_section_check_p = 0.0)
    {
        set_time_before_section(time_b4_section_check_p);
        set_initial_conditions(x_p);
        set_parameter(param_p);
        set_hyperplane(hyperplane_p);
        time_stepper->execute(x_p, param_p, time_interval_p);
        get_results(x_p);
    }

    void execute()
    {
        time_stepper->execute();
    }

    void execute_with_no_section(T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p)
    {
        set_time_before_section(time_interval_p.second);
        set_initial_conditions(x_p);
        set_parameter(param_p);
        time_stepper->execute(x_p, param_p, time_interval_p);
        get_results(x_p);
    }

    T get_simulated_time()const
    {
        return time_stepper->get_simulated_time();
    }
    T get_period_estmate_time()const
    {
        return em->get_period_estmate_time();
    }    
    void reset()
    {
        time_stepper->reset();
        em->reset();
    }
    void save_norms(const std::string& file_name_)const
    {
        time_stepper->save_norms(file_name_);
    }
    void save_period_estmate_norms(const std::string& file_name_)const
    {
        em->save_period_estmate_norms(file_name_);
    }


};

}
}

#endif // __PERIODIC_ORBIT_TIME_STEPPER_TO_SECTION_H

