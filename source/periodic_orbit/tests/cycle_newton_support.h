#ifndef PERIODIC_ORBIT_CYCLE_NEWTON_SUPPORT_H
#define PERIODIC_ORBIT_CYCLE_NEWTON_SUPPORT_H

#include "prepared_cycle_support.h"
#include "newton_test_support.h"

namespace cycle_tests
{
using types             = section_tests::newton_types<blocks, orbit_problem>;
using cycle_log         = types::log;
using cycle_monitor     = types::monitor;
using cycle_linear      = types::linear;
using cycle_iteration   = types::iteration;
using cycle_convergence = types::convergence;
using cycle_solver      = types::solver;
using section_tests::make_newton;
}
#endif
