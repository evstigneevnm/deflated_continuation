#ifndef __TIME_STEPPER_ALL_METHODS_ENUM_H__
#define __TIME_STEPPER_ALL_METHODS_ENUM_H__

#include <vector>
#include <string>

namespace time_steppers
{
namespace detail
{

enum methods {
    EXPLICIT_EULER = 0,  
    HEUN_EULER,
    RK33SSP, 
    RK43SSP,
    RKDP45,
    RK64SSP};
}

std::vector<std::string> methods_str{"EXPLICIT_EULER", "HEUN_EULER", "RK33SSP", "RK43SSP", "RKDP45", "RK64SSP"};
}

#endif