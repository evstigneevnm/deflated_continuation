#ifndef __TIME_STEPPER_ALL_METHODS_ENUM_H__
#define __TIME_STEPPER_ALL_METHODS_ENUM_H__

#include <vector>
#include <string>

namespace nmfd
{
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
} // namespace detail
} // namespace time_steppers
} // namespace nmfd

// Compatibility for steppers and periodic-orbit code not yet migrated to NMFD.
namespace time_steppers
{
namespace detail
{
using nmfd::time_steppers::detail::methods;
using nmfd::time_steppers::detail::EXPLICIT_EULER;
using nmfd::time_steppers::detail::HEUN_EULER;
using nmfd::time_steppers::detail::RK33SSP;
using nmfd::time_steppers::detail::RK43SSP;
using nmfd::time_steppers::detail::RKDP45;
using nmfd::time_steppers::detail::RK64SSP;
using nmfd::time_steppers::detail::IMEX_EULER;
using nmfd::time_steppers::detail::IMEX_TR2;
using nmfd::time_steppers::detail::IMEX_ARS3;
using nmfd::time_steppers::detail::IMEX_AS2;
using nmfd::time_steppers::detail::IMEX_KOTO2;
using nmfd::time_steppers::detail::IMEX_SSP222;
using nmfd::time_steppers::detail::IMEX_SSP322;
using nmfd::time_steppers::detail::IMEX_SSP332;
using nmfd::time_steppers::detail::IMEX_SSP333;
using nmfd::time_steppers::detail::IMEX_SSP433;
using nmfd::time_steppers::detail::IMPLICIT_EULER;
using nmfd::time_steppers::detail::IMPLICIT_MIDPOINT;
using nmfd::time_steppers::detail::CRANK_NICOLSON;
using nmfd::time_steppers::detail::SDIRK2A1;
using nmfd::time_steppers::detail::ESDIRK3A2;
using nmfd::time_steppers::detail::SDIRK3A3;
} // namespace detail
} // namespace time_steppers

#endif
