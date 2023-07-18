#ifndef __TIME_STEPPER_IMPLICIT_TIME_STEP_H__
#define __TIME_STEPPER_IMPLICIT_TIME_STEP_H__

template<class VectorOperations, class NonlinearOperator, class Log, class TimeStepAdaptation>
class implicit_time_step
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using method_type = detail::methods;
    using table_t = time_steppers::detail::tableu;

    implicit_time_step(VectorOperations* vec_ops_p, NonlinearOperator* nonlin_op_p, TimeStepAdaptation* time_step_adapt_p, Log* log_, T param_p = 1.0,  method_type method_p = method_type::RKDP45):
    vec_ops_(vec_ops_p), 
    nonlin_op_(nonlin_op_p), 
    time_step_adapt_(time_step_adapt_p),
    log(log_), 
    param_(param_p),
    method_(method_p)
    {
        set_table_and_reinit_storage();
        vec_ops_->init_vectors(v1_helper_, f_helper_); vec_ops_->start_use_vectors(v1_helper_, f_helper_);
        numeric_eps_ = std::numeric_limits<T>::epsilon();
    }
    ~implicit_time_step()
    {
        vec_ops_->stop_use_vectors(v1_helper_, f_helper_); vec_ops_->free_vectors(v1_helper_, f_helper_);
        clear_storage();
    }
    
    void scheme(method_type method_p)
    {
        if(method_ != method_p)
        {
            method_ = method_p;
            set_table_and_reinit_storage();
        }
    }
};

}


#endif