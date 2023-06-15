#ifndef __TIME_STEPPER_ALL_METHODS_ENUM_H__
#define __TIME_STEPPER_ALL_METHODS_ENUM_H__

namespace time_steppers
{
namespace detail
{

enum methods {
    EXPLICIT_EULER, 
    RK3SSP, 
    RKDP45};
}
}

#endif