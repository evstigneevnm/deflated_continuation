#ifndef __PERIODIC_ORBIT_POINCARE_MAP_OPERATOR_H__
#define __PERIODIC_ORBIT_POINCARE_MAP_OPERATOR_H__

#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/positive_preserving_dummy.h>
#include "time_stepper_to_section.h"


namespace periodic_orbit
{
template<class VectorOperations, class NonlinearOperator, template<class, class, class>class TimeStepAdaptation, template<class, class, class, class> class SingleStepper, class Hyperplane, class Log>
class poincare_map_operator
{
    
    using T_vec = typename VectorOperations::vector_type;
    using T = typename VectorOperations::scalar_type;

    //TODO:  move "time_steppers::detail::positive_preserving_dummy<gvec_ops_t>" to template parameters
    using time_step_adopt_t = TimeStepAdaptation<VectorOperations, Log, ::time_steppers::detail::positive_preserving_dummy<VectorOperations> >;
    using single_step_t = SingleStepper<VectorOperations, NonlinearOperator, Log, time_step_adopt_t>;
    using time_stepper_t = time_steppers::time_stepper_to_section<VectorOperations, NonlinearOperator, single_step_t, Hyperplane, Log>;
    using method_type = ::time_steppers::detail::methods;    

public:

    poincare_map_operator(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, Log* log_p, T max_time, T param_p = 1.0, method_type method_p = method_type::RKDP45, T dt_initial_p = 1.0/500.0):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p),
    log_(log_p)
    {
        time_step_adopt_ = new time_step_adopt_t(vec_ops_, log_, {0, max_time}, dt_initial_p);
        single_step_ = new single_step_t(vec_ops_, time_step_adopt_, log_, nonlin_op_, param_p, method_p);
        time_advance_ = new time_stepper_t(vec_ops_, nonlin_op_, single_step_, log_);

        set_parameter(param_p);        
    }
    ~poincare_map_operator()
    {
        delete time_step_adopt_;
        delete single_step_;
        delete time_advance_;
    }


    void set_method(method_type method_p)
    {
        single_step_->scheme(method_p);
    }

    void set_parameter(const T param_p)
    {
        time_advance_->set_parameter(param_p);
    }

    void set_hyperplanes(std::pair<Hyperplane*, Hyperplane*>& hyperplane_pair_p)
    {
        hyperplane_pair_ = hyperplane_pair_p;
        time_advance_->set_hyperplane(hyperplane_pair_.second);
    }
    
    void F(const T_vec& x_in_p, const T lambda, T_vec& x_out_p)const
    {
        hyperplane_pair_.first->update(0.0, x_in_p, lambda);
        time_advance_->reset();
        time_advance_->set_initial_conditions(x_in_p);
        time_advance_->execute();
        time_advance_->get_results(x_out_p);
    }

    void time_stepper(T_vec& x_p, const T param_p, const std::pair<T,T> time_interval_p)
    {
        time_advance_->execute_with_no_section(x_p, param_p, time_interval_p);
    }

    T get_simulated_time()const
    {
        return time_advance_->get_simulated_time();
    }
    T get_period_estmate_time()const
    {
        return time_advance_->get_period_estmate_time();
    }
    void save_norms(const std::string& file_name_)const
    {
        time_advance_->save_norms(file_name_);
    }
    void save_period_estmate_norms(const std::string& file_name_)const
    {
        time_advance_->save_period_estmate_norms(file_name_);
    }


private:
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    Log* log_;
    single_step_t* single_step_;
    time_step_adopt_t* time_step_adopt_;
    time_stepper_t* time_advance_;


    std::pair<Hyperplane*, Hyperplane*> hyperplane_pair_;


};

}


#endif // __PERIODIC_ORBIT_POINCARE_MAP_OPERATOR_H__