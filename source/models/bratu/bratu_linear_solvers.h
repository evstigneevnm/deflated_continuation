#ifndef __BRATU_LINEAR_SOLVERS_H__
#define __BRATU_LINEAR_SOLVERS_H__

#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <nmfd/solvers/sherman_morrison.h>

namespace bratu_model
{
template <class Space, class Log>
using monitor = nmfd::solvers::default_monitor<Space, Log>;
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using direct_solver = nmfd::solvers::direct_solver<Space, Monitor, Log, Operator, Preconditioner>;
template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using linear_system =
    nmfd::solvers::sherman_morrison<Space, Monitor, Log, Operator, Preconditioner, nmfd::solvers::direct_solver>;
} // namespace bratu_model

#endif
