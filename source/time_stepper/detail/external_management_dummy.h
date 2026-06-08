#ifndef __TIME_STEPPER_EXTERNAL_MANAGEMENT_DUMMY_H__
#define __TIME_STEPPER_EXTERNAL_MANAGEMENT_DUMMY_H__


namespace time_steppers{
namespace detail{

template<class VectorOperations, class SingleStepper>
struct external_management_dummy
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;   

    external_management_dummy() = default;
    ~external_management_dummy() = default;
    
    bool apply(T simulated_time, T& dt, T_vec& v_in, T_vec& v_out)
    {
        return false; //returns external control of the finish flag
    }

};


}
}


#endif