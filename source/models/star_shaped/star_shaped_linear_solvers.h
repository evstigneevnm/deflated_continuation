#ifndef __STAR_SHAPED_LINEAR_SOLVERS_H__
#define __STAR_SHAPED_LINEAR_SOLVERS_H__

#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <main/nmfd_direct_newton_types.h>
#include <nonlinear_operators/star_shaped/augmented_inverse_star_shaped.h>

namespace star_shaped_model
{
template <class Space, class Log>
using monitor = nmfd::solvers::default_monitor<Space, Log>;
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using direct_solver = nmfd::solvers::direct_solver<Space, Monitor, Log, Operator, Preconditioner>;
template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using linear_system =
    nmfd::solvers::sherman_morrison<Space, Monitor, Log, Operator, Preconditioner, nmfd::solvers::direct_solver>;

template <class Space, class Log, class Problem, class Jacobian, class LinearSystem, class Storage>
using newton_types = main_classes::nmfd_direct_newton_types<
    Space, Log, Problem, Jacobian, LinearSystem, Storage, nonlinear_operators::augmented_inverse_star_shaped>;
} // namespace star_shaped_model

#endif
