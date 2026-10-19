#ifndef __CIRCLE_LINEAR_SOLVERS_H__
#define __CIRCLE_LINEAR_SOLVERS_H__

#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>

namespace circle_model
{

// Template-order aliases for workflows that still support unmigrated models.
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using cgs = nmfd::solvers::cgs<Space, Monitor, Log, Operator, Preconditioner>;

template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using bicgstabl = nmfd::solvers::bicgstabl<Space, Monitor, Log, Operator, Preconditioner>;

} // namespace circle_model

#endif
