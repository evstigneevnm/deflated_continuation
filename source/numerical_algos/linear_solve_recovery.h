#ifndef __LINEAR_SOLVE_RECOVERY_H__
#define __LINEAR_SOLVE_RECOVERY_H__

#include <utility>

namespace nmfd
{
namespace solvers
{
template <class Space, class Monitor, class Log, class Operator, class Inverse>
class direct_solver;
}
}

namespace numerical_algos
{
namespace lin_solvers
{
namespace recovery
{

// An inverse-only solve has no unpreconditioned fallback, regardless of its setters.
template <class Space, class Monitor, class Log, class Operator, class Inverse>
bool supports_unpreconditioned_retry(
    const nmfd::solvers::direct_solver<Space, Monitor, Log, Operator, Inverse> *, int
)
{
    return false;
}

template <class Solver>
auto supports_unpreconditioned_retry( const Solver *solver, int )
    -> decltype( solver->supports_unpreconditioned_retry() )
{
    return solver->supports_unpreconditioned_retry();
}

// Preserve the existing recovery contract for callers without an explicit capability.
template <class Solver>
bool supports_unpreconditioned_retry( const Solver *, long )
{
    return true;
}

template <class Solver>
auto set_preconditioner_bypass( Solver *solver, const bool enabled, int )
    -> decltype( solver->set_preconditioner_bypass( enabled ), bool() )
{
    solver->set_preconditioner_bypass( enabled );
    return true;
}

template <class Solver>
bool set_preconditioner_bypass( Solver *, const bool, long )
{
    return false;
}

template <class Solver, class Solve, class Reset, class OnRetry>
bool solve_with_unpreconditioned_retry( Solver *solver, Solve &&solve, Reset &&reset, OnRetry &&on_retry )
{
    if ( solve() )
    {
        return true;
    }
    if ( !supports_unpreconditioned_retry( solver, 0 ) || !set_preconditioner_bypass( solver, true, 0 ) )
    {
        return false;
    }

    struct bypass_guard
    {
        Solver *solver;
        ~bypass_guard()
        {
            set_preconditioner_bypass( solver, false, 0 );
        }
    } guard{ solver };

    on_retry();
    reset();
    return solve();
}

template <class Solver, class Solve, class Reset>
bool solve_with_unpreconditioned_retry( Solver *solver, Solve &&solve, Reset &&reset )
{
    return solve_with_unpreconditioned_retry( solver, std::forward<Solve>( solve ), std::forward<Reset>( reset ), []() {
    } );
}

} // namespace recovery
} // namespace lin_solvers
} // namespace numerical_algos

#endif // __LINEAR_SOLVE_RECOVERY_H__
