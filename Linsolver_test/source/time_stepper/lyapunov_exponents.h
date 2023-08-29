#ifndef __TIME_STEPPER_LYAPUNOV_EXPONENTS_H__
#define __TIME_STEPPER_LYAPUNOV_EXPONENTS_H__

#include <time_stepper/time_stepper.h>
#include "detail/glued_nonlinear_operator_and_multiple_jacobian.h"
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/positive_preserving_dummy.h>


namespace time_steppers
{

template<class VectorOperations, class NonlinearOperator, template<class, class, class>class TimeStepAdaptation, template<class, class, class, class> class SingleStepper, class Log, size_t GluedVectors >
class lyapunov_exponents
{

    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;   

    using glued_nonlinear_operator_t = detail::glued_nonlinear_operator_and_multiple_jacobian<VectorOperations, NonlinearOperator, GluedVectors>;
    using gvec_ops_t = typename glued_nonlinear_operator_t::glued_vector_operations_type;

    using T_gvec = typename glued_nonlinear_operator_t::vector_type;


    using time_step_adopt_t = TimeStepAdaptation<gvec_ops_t, Log, ::time_steppers::detail::positive_preserving_dummy<gvec_ops_t> >;
    using single_step_t = SingleStepper<gvec_ops_t, glued_nonlinear_operator_t, Log, time_step_adopt_t>;

    using time_stepper_t = ::time_steppers::time_stepper<gvec_ops_t, glued_nonlinear_operator_t, single_step_t, Log>;
    using method_type = ::time_steppers::detail::methods;
    




    time_step_adopt_t* time_step_adopt_;
    single_step_t* single_step_;
    method_type method_;

    time_stepper_t* time_advance_;
    glued_nonlinear_operator_t* glued_nonlin_op_;
    gvec_ops_t* g_vec_ops_;

    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    Log* log_;
    mutable T_gvec glued_vec_;
    T iteration_time_;

public:
	lyapunov_exponents(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p,  Log* log_p, T iteration_time, T param_p = 1.0, method_type method_p = method_type::RKDP45, T dt_initial_p = 1.0/500.0):
    vec_ops_(vec_ops_p),
    nonlin_op_(nonlin_op_p),
    log_(log_p),
    iteration_time_(iteration_time)
	{
		glued_nonlin_op_ = new glued_nonlinear_operator_t(vec_ops_, nonlin_op_);
		g_vec_ops_ = glued_nonlin_op_->get_glued_vec_ops();

		time_step_adopt_ = new time_step_adopt_t(g_vec_ops_, log_, {0, iteration_time}, dt_initial_p );
		single_step_ = new single_step_t(g_vec_ops_, time_step_adopt_, log_, glued_nonlin_op_, param_p, method_p);
		time_advance_ = new time_stepper_t(g_vec_ops_, glued_nonlin_op_, single_step_, log_);
		set_parameter(param_p);
		g_vec_ops_->init_vector(glued_vec_); g_vec_ops_->start_use_vector(glued_vec_);
		for(size_t l = 1; l < GluedVectors; l++)
		{
			nonlin_op_->randomize_vector( glued_vec_.comp(l) );
		}
	}
	~lyapunov_exponents()
	{
        g_vec_ops_->stop_use_vector(glued_vec_); g_vec_ops_->free_vector(glued_vec_); 
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
        time_advance_->set_parameter(param_p);
    }

    void run_single_time(const T_vec& v_in, T_vec& v_out) const
    {
        time_advance_->reset();
    	vec_ops_->assign(v_in, glued_vec_.comp(0));
    	time_advance_->set_initial_conditions(glued_vec_);
    	time_advance_->execute();
    	time_advance_->get_results(glued_vec_);
    	vec_ops_->assign(glued_vec_.comp(0), v_out);
    }




};

}


#endif