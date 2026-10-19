#ifndef __KS1D_LINEAR_SOLVERS_H__
#define __KS1D_LINEAR_SOLVERS_H__

#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>

namespace ks1d_model
{

// Preserve the workflow's template ordering while using the canonical solver.
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using bicgstabl = nmfd::solvers::bicgstabl<Space, Monitor, Log, Operator, Preconditioner>;

} // namespace ks1d_model

#endif
