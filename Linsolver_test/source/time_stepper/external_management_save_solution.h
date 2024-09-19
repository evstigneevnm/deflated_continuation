#ifndef __TIME_STEPPER_EXTERNAL_MANAGEMENT_SAVE_SOLUTION_H__
#define __TIME_STEPPER_EXTERNAL_MANAGEMENT_SAVE_SOLUTION_H__

#include <string>
#include <sstream>


namespace time_steppers{
namespace detail{

template<class VectorOperations, class NonlinearOperator>
struct external_management_save_solution
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;   

    external_management_save_solution(NonlinearOperator* nonlin_op, T temporal_interval, std::string&& file_name_and_extension):
    nonlin_op_(nonlin_op),
    temporal_interval_(temporal_interval),
    file_name_and_extension_(file_name_and_extension),
    internal_time_counter_(0.0),
    internal_counter_(0.0)
    {}
    ~external_management_save_solution() = default;
    
    bool apply(T simulated_time, T& dt, T_vec& v_in, T_vec& v_out)
    {
        internal_time_counter_ += dt;
        if (internal_time_counter_ >= temporal_interval_)
        {
            internal_time_counter_ = 0.0;
            std::stringstream ss;
            ss << internal_counter_ << "_" << file_name_and_extension_;
            nonlin_op_->write_solution_abs(ss.str(), v_out);
            internal_counter_++;
        }
        return false; //returns external control of the finish flag
    }


private:
    NonlinearOperator* nonlin_op_;
    T internal_time_counter_;
    T temporal_interval_;
    std::string file_name_and_extension_;
    std::size_t internal_counter_;

};


}
}


#endif