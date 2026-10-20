#ifndef __STAR_SHAPED_LINEAR_SOLVERS_H__
#define __STAR_SHAPED_LINEAR_SOLVERS_H__

#if defined( STAR_SHAPED_VECTOR_BACKEND_VAR_PREC )
#    include <numerical_algos/lin_solvers/default_monitor.h>
#    include <numerical_algos/lin_solvers/exact_wrapper.h>
#    include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#else
#    include <nmfd/solvers/default_monitor.h>
#    include <nmfd/solvers/direct_solver.h>
#    include <nmfd/solvers/sherman_morrison.h>
#endif

namespace star_shaped_model
{
#if defined( STAR_SHAPED_VECTOR_BACKEND_VAR_PREC )
template <class Space, class Log>
using monitor = numerical_algos::lin_solvers::default_monitor<Space, Log>;
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using direct_solver = numerical_algos::lin_solvers::exact_wrapper<Operator, Preconditioner, Space, Monitor, Log>;
template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using linear_system = numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
    Operator, Preconditioner, Space, Monitor, Log, direct_solver>;
#else
template <class Space, class Log>
using monitor = nmfd::solvers::default_monitor<Space, Log>;
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using direct_solver = nmfd::solvers::direct_solver<Space, Monitor, Log, Operator, Preconditioner>;
template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using linear_system =
    nmfd::solvers::sherman_morrison<Space, Monitor, Log, Operator, Preconditioner, nmfd::solvers::direct_solver>;
#endif
} // namespace star_shaped_model

#endif
