#ifndef __BRATU_LINEAR_SOLVERS_H__
#define __BRATU_LINEAR_SOLVERS_H__

#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <main/nmfd_newton_types.h>
#include <nonlinear_operators/bratu/augmented_inverse_bratu.h>

namespace bratu_model
{
template <class Space, class Log>
using monitor = nmfd::solvers::default_monitor<Space, Log>;
template <class Operator, class Preconditioner, class Space, class Monitor, class Log>
using direct_solver = nmfd::solvers::direct_solver<Space, Monitor, Log, Operator, Preconditioner>;
template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using linear_system =
    nmfd::solvers::sherman_morrison<Space, Monitor, Log, Operator, Preconditioner, nmfd::solvers::direct_solver>;

template <class Space, class Log, class Problem, class Jacobian>
struct augmented_solver_types
{
    using corrector_type = continuation::corrector_operator<Space, Problem, Jacobian>;
    using space_type     = typename corrector_type::vector_space_type;
    using operator_type  = typename corrector_type::jacobi_operator_type;
    using inverse_type   = nonlinear_operators::augmented_inverse_bratu<space_type, operator_type>;
    using solver_type =
        nmfd::solvers::direct_solver<space_type, monitor<space_type, Log>, Log, operator_type, inverse_type>;
    using linear_type = continuation::augmented_linear_solver<space_type, operator_type, solver_type>;
};

template <class Space, class Log, class Problem, class Jacobian, class LinearSystem, class Storage>
struct newton_types
    : main_classes::nmfd_newton_types<
          Space, Log, Problem, Jacobian, LinearSystem, Storage, main_classes::detail::newton_convergence,
          main_classes::detail::continuation_convergence, main_classes::detail::deflation_convergence,
          typename augmented_solver_types<Space, Log, Problem, Jacobian>::linear_type>
{
    using assembly_type = augmented_solver_types<Space, Log, Problem, Jacobian>;
    using space_type    = typename assembly_type::space_type;

    template <class Preconditioner>
    static auto make_bordered(
        const std::shared_ptr<space_type> &space, const std::shared_ptr<LinearSystem> &,
        const std::shared_ptr<Preconditioner> &, Log *log
    )
    {
        using solver_type = typename assembly_type::solver_type;
        auto inverse      = std::make_shared<typename assembly_type::inverse_type>( space );
        auto solver       = std::make_shared<solver_type>( space, log, typename solver_type::params(), inverse );
        return std::make_shared<typename assembly_type::linear_type>( space, std::move( solver ) );
    }
};
} // namespace bratu_model

#endif
