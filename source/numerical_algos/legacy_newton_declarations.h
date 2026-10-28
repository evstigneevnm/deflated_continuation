#ifndef __LEGACY_NEWTON_DECLARATIONS_H__
#define __LEGACY_NEWTON_DECLARATIONS_H__

// Historical assembly is available only to explicitly opted-in build targets.
#if defined( PROJECT_ENABLE_LEGACY_NEWTON )
#    include <numerical_algos/newton_solvers/newton_solver.h>
#    include <numerical_algos/newton_solvers/newton_solver_extended.h>
namespace numerical_algos
{
inline constexpr bool legacy_newton_enabled = true;
}
#else
namespace numerical_algos
{
inline constexpr bool legacy_newton_enabled = false;
namespace newton_method
{
template <class VectorOperations, class NonlinearOperator, class SystemOperator, class ConvergenceStrategy>
class newton_solver;
}
namespace newton_method_extended
{
template <
    class VectorOperations, class NonlinearOperator, class SystemOperator, class ConvergenceStrategy,
    class SolutionPoint>
class newton_solver_extended;
}
}
#endif

#endif
