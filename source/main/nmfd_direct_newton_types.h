#ifndef __MAIN_NMFD_DIRECT_NEWTON_TYPES_H__
#define __MAIN_NMFD_DIRECT_NEWTON_TYPES_H__

#include <main/nmfd_newton_types.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>

namespace main_classes
{

template <class Space, class Log, class Problem, class Jacobian, template <class, class> class Inverse>
struct direct_augmented_solver_types
{
    using corrector_type = continuation::corrector_operator<Space, Problem, Jacobian>;
    using space_type     = typename corrector_type::vector_space_type;
    using operator_type  = typename corrector_type::jacobi_operator_type;
    using inverse_type   = Inverse<space_type, operator_type>;
    using solver_type    = nmfd::solvers::direct_solver<
           space_type, nmfd::solvers::default_monitor<space_type, Log>, Log, operator_type, inverse_type>;
    using linear_type = continuation::augmented_linear_solver<space_type, operator_type, solver_type>;
};

template <
    class Space, class Log, class Problem, class Jacobian, class LinearSystem, class Storage,
    template <class, class> class Inverse,
    template <class, class, class, class, class> class FixedConvergence        = detail::newton_convergence,
    template <class, class, class, class, class> class ContinuationConvergence = detail::continuation_convergence,
    template <class, class, class, class, class> class DeflationConvergence    = detail::deflation_convergence>
struct nmfd_direct_newton_types
    : nmfd_newton_types<
          Space, Log, Problem, Jacobian, LinearSystem, Storage, FixedConvergence, ContinuationConvergence,
          DeflationConvergence,
          typename direct_augmented_solver_types<Space, Log, Problem, Jacobian, Inverse>::linear_type>
{
    using assembly_type = direct_augmented_solver_types<Space, Log, Problem, Jacobian, Inverse>;
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

}

#endif
