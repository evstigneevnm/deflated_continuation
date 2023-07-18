#ifndef __PERIODIC_ORBIT_GLUED_POINCARE_MAP_LINEAR_OPERATOR_H__
#define __PERIODIC_ORBIT_GLUED_POINCARE_MAP_LINEAR_OPERATOR_H__

#include <time_stepper/time_stepper.h>
#include "detail/glued_nonlinear_operator_and_jacobian.h"
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/positive_preserving_dummy.h>
namespace periodic_orbit
{

//
//class VectorOperations, class NonlinearOperator, class Log, class TimeStepAdaptation
template<class VectorOperations, class NonlinearOperator, template<class, class, class>class TimeStepAdaptation, template<class, class, class, class> class SingleStepper, class Hyperplane, class Log >
class glued_poincare_map_linear_operator
{

    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;   

    using glued_nonlinear_operator_t = detail::glued_nonlinear_operator_and_jacobian<VectorOperations, NonlinearOperator>;
    using gvec_ops_t = typename glued_nonlinear_operator_t::glued_vector_operations_type;

    using T_gvec = typename glued_nonlinear_operator_t::vector_type;

    template<class SingleStepperCustom>
    struct external_management
    {

        external_management(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, SingleStepperCustom* stepper_p, Log* log_p, std::pair<T,size_t> bisection_params_p = {1.0e-12, 100}):
        periodic_time_(0.0),
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
        bool apply(T simulated_time, T& dt, T_gvec& v_in, T_gvec& v_out)
        {
            if(hyperplane_ == nullptr)
            {
                throw std::logic_error("glued_poincare_map_linear_operator::external_management: hyperplane is not set!");
            }
            bool finish_besection = false;
            if(periodic_time_ == 0.0)
            {
                std::vector<T> bif_norms_at_t_;
                bif_norms_at_t_.push_back(periodic_time_);
                nonlin_op_->norm_bifurcation_diagram(v_in.comp(0), bif_norms_at_t_);
                bif_norms_at_t_.push_back(0.0);
                solution_period_estmate_norms.push_back(bif_norms_at_t_);
            }            
            if(hyperplane_->is_crossed_in_normal_direction(v_in.comp(0), v_out.comp(0) ))
            {
                finish_besection = bisect_intersection(v_in, dt, v_out);
            }
            periodic_time_ += dt;
            std::vector<T> bif_norms_at_t_;
            bif_norms_at_t_.push_back(periodic_time_);
            nonlin_op_->norm_bifurcation_diagram(v_out.comp(0), bif_norms_at_t_);
            bif_norms_at_t_.push_back(dt);
            solution_period_estmate_norms.push_back(bif_norms_at_t_);
    
            return finish_besection;             
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
            periodic_time_ = 0.0;   
            solution_period_estmate_norms.resize(0);
        }
        // do we need it here?
        void save_period_estmate_norms(const std::string& file_name_)const
        {
            std::ofstream file_;
            file_ = std::ofstream(file_name_.c_str(), std::ofstream::out);
            if(!file_) throw std::runtime_error("glued_poincare_map_linear_operator::save_period_estmate_norms: failed to open file " + file_name_ + " for output.");

            for(auto &x_: solution_period_estmate_norms)
            {
                for(auto &y_: x_)
                {
                    if(!(file_ << std::scientific << y_ << " ")) throw std::runtime_error("glued_poincare_map_linear_operator::save_period_estmate_norms: failed to write to " + file_name_);
                }
                if(!(file_ << std::endl)) throw std::runtime_error("glued_poincare_map_linear_operator::save_period_estmate_norms: failed to write to " + file_name_);
            }
            file_.close();            
        }


    private:
        VectorOperations* vec_ops_;
        Hyperplane* hyperplane_;
        SingleStepperCustom* stepper_;
        NonlinearOperator* nonlin_op_;
        Log* log_;
        T param_;
        T bisection_tolerance_;
        size_t bisection_iters_;
        T periodic_time_;
        std::vector< std::vector<T> > solution_period_estmate_norms;

        bool bisect_intersection(const T_gvec& v_in, T& dt_mod, T_gvec& v_out)
        {
            bool ret_flag = false;
            T error = 1.0;
            T distance = 1.0;
            size_t iters = 0;
            while((std::abs(error) > bisection_tolerance_)&&(iters < bisection_iters_))
            {
                std::tie(error, distance) = hyperplane_->intersection(v_in.comp(0), v_out.comp(0));
                dt_mod = dt_mod - error*dt_mod/distance;
                stepper_->force_undo_step();
                stepper_->execute_forced_dt(dt_mod, v_in, v_out);
                ++iters;
            }
            if(std::abs(error) <= bisection_tolerance_)
            {
                ret_flag = true;
                log_->info_f("glued_poincare_map_linear_operator::external_management: intersection tolerance %.2le on %i iteration with error %.2le using timestep %.2le", bisection_tolerance_, iters, error, dt_mod);
            }  
            else
            {
                log_->warning_f("glued_poincare_map_linear_operator::external_management: failed to obtain intersection on %i iteration with error %.2le using timestep %.2le",  iters, error, dt_mod);
            }      
            return ret_flag;
        }        

    };    

    using time_step_adopt_t = TimeStepAdaptation<gvec_ops_t, Log, ::time_steppers::detail::positive_preserving_dummy<gvec_ops_t> >;
    using single_step_t = SingleStepper<gvec_ops_t, glued_nonlinear_operator_t, Log, time_step_adopt_t>;

    using external_management_t = external_management<single_step_t>;
    //making a custom timestepper to section
    using time_stepper_t = ::time_steppers::time_stepper<gvec_ops_t, glued_nonlinear_operator_t, single_step_t, Log, external_management_t>;

    using method_type = ::time_steppers::detail::methods;
    
    time_step_adopt_t* time_step_adopt_;
    single_step_t* single_step_;
    ::time_steppers::detail::methods method_;

    external_management_t* external_;
    time_stepper_t* time_advance_;
    glued_nonlinear_operator_t* glued_nonlin_op_;
    gvec_ops_t* g_vec_ops_;

public:
//VectorOperations* vec_ops_p, TimeStepAdaptation* time_step_adapt_p, Log* log_, NonlinearOperator* nonlin_op_p = nullptr, T param_p = 1.0,  method_type method_p = method_type::RKDP45

    glued_poincare_map_linear_operator(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p,  Log* log_p, T max_time, T param_p = 1.0, method_type method_p = method_type::RKDP45, T dt_initial_p = 1.0/500.0):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p),
    log_(log_p)
    {   
        
        glued_nonlin_op_ = new glued_nonlinear_operator_t(vec_ops_, nonlin_op_);
        g_vec_ops_ = glued_nonlin_op_->get_glued_vec_ops();

        time_step_adopt_ = new time_step_adopt_t(g_vec_ops_, log_, {0,max_time}, dt_initial_p );
        single_step_ = new single_step_t(g_vec_ops_, time_step_adopt_, log_, glued_nonlin_op_, param_p, method_p);
        external_ = new external_management_t(vec_ops_, nonlin_op_, single_step_, log_);

        time_advance_ = new time_stepper_t(g_vec_ops_, glued_nonlin_op_, single_step_, log_, external_);
        
        set_parameter(param_p);

        g_vec_ops_->init_vector(glued_vec_); g_vec_ops_->start_use_vector(glued_vec_);

    }
    ~glued_poincare_map_linear_operator()
    {
        g_vec_ops_->stop_use_vector(glued_vec_); g_vec_ops_->free_vector(glued_vec_); 
        delete external_;
        delete single_step_;
        delete time_step_adopt_;
        delete glued_nonlin_op_;
        delete time_advance_;

    }

    void set_method(method_type method_p)
    {
        single_step_->scheme(method_p);
    }

    void set_parameter(const T param_p)
    {
        external_->set_parameter(param_p);
        time_advance_->set_parameter(param_p);
    }

    T get_period_estmate_time()const 
    {
        return external_->get_period_estmate_time();
    }

    void set_hyperplanes(std::pair<Hyperplane*, Hyperplane*> hyperplane_pair_p)
    {
        hyperplane_pair_ = hyperplane_pair_p;
        external_->set_hyperplane(hyperplane_pair_.second);
    }
    
    void apply(const T_vec& v_in, T_vec& v_out)const
    {   
        Hyperplane* plane_0 = hyperplane_pair_.first;
        Hyperplane* plane_1 = hyperplane_pair_.second;
        plane_0->get_initial_point( glued_vec_.comp(0) );
        vec_ops_->assign(v_in, glued_vec_.comp(1));
        plane_0->restore_from( glued_vec_.comp(1) );
        time_advance_->reset();
        external_->reset();
        time_advance_->set_initial_conditions(glued_vec_);
        time_advance_->execute();
        time_advance_->get_results(glued_vec_);
        auto period_time = time_advance_->get_simulated_time();

        plane_1->project_to(period_time, glued_vec_.comp(0), glued_vec_.comp(1));
        vec_ops_->assign_mul(-1.0, v_in, 1.0, glued_vec_.comp(1), v_out);

    }
    void save_period_estmate_norms(const std::string& file_name_)const
    {
        external_->save_period_estmate_norms(file_name_);
    }

private:
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    Log* log_;
    std::pair<Hyperplane*, Hyperplane*> hyperplane_pair_;
    mutable T_gvec glued_vec_;

};

}

#endif // __PERIODIC_ORBIT_GLUED_POINCARE_MAP_LINEAR_OPERATOR_H__