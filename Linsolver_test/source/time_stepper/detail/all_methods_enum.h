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
    RK64SSP,
    IMEX_EULER,
    IMEX_TR2,
    IMEX_ARS3,
    IMEX_AS2,
    IMEX_KOTO2,
    IMEX_SSP222,
    IMEX_SSP322,
    IMEX_SSP332,
    IMEX_SSP333,
    IMEX_SSP433,
    IMPLICIT_EULER,
    IMPLICIT_MIDPOINT,
    CRANK_NICOLSON,
    SDIRK2A1,
    ESDIRK3A2,
    SDIRK3A3
    };
}

}

#endif