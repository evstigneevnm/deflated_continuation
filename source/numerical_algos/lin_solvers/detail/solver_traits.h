#ifndef __SCFD_SOLVER_TRAITS_H__
#define __SCFD_SOLVER_TRAITS_H__

#include <type_traits>

namespace numerical_algos
{
namespace lin_solvers 
{
namespace detail
{

template <typename Solver, typename = int>
struct has_hierarchy : std::false_type { };

template <typename Solver>
struct has_hierarchy<Solver, decltype((void)(typename Solver::params_hierarchy()),(void)(Solver::has_hierarchy),int(0))> : 
    std::integral_constant<bool,Solver::has_hierarchy> { };

} // namespace detail
} // namespace lin_solvers
} // namespace numerical_algos

#endif
