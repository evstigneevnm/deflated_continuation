#ifndef __MAIN_SOLVER_CONSTRUCTION_H__
#define __MAIN_SOLVER_CONSTRUCTION_H__

#include <memory>
#include <type_traits>

namespace main_classes
{
namespace detail
{

template <class Component, class Problem>
std::shared_ptr<Component> make_problem_component( Problem *problem )
{
    return std::make_shared<Component>( problem );
}

template <class Component, class Problem>
std::shared_ptr<Component> make_problem_component( const std::shared_ptr<Problem> &problem )
{
    if constexpr ( std::is_constructible<Component, std::shared_ptr<Problem>>::value )
        return std::make_shared<Component>( problem );
    else
    {
        auto *borrowed = problem.get();
        return make_problem_component<Component>( borrowed );
    }
}

template <class Solver, class Space, class Operator, class Preconditioner, class Log>
std::shared_ptr<Solver> make_linear_system(
    Space *space, const std::shared_ptr<Operator> &, const std::shared_ptr<Preconditioner> &prec, Log *log
)
{
    auto *borrowed = prec.get();
    return std::make_shared<Solver>( borrowed, space, log );
}

template <class Solver, class Space, class Operator, class Preconditioner, class Log>
std::shared_ptr<Solver> make_linear_system(
    const std::shared_ptr<Space> &space, const std::shared_ptr<Operator> &op,
    const std::shared_ptr<Preconditioner> &prec, Log *log
)
{
    if constexpr ( std::is_constructible<
                       Solver, std::shared_ptr<Space>, std::shared_ptr<const Operator>, std::shared_ptr<Preconditioner>,
                       Log *>::value )
        return std::make_shared<Solver>( space, op, prec, log );
    else
        return make_linear_system<Solver>( space.get(), op, prec, log );
}

template <class Solver, class Space, class Operator, class Preconditioner, class Log>
std::shared_ptr<Solver> make_linear_solver(
    Space *space, const std::shared_ptr<Operator> &, const std::shared_ptr<Preconditioner> &prec, Log *log
)
{
    auto solver = std::make_shared<Solver>( space, log );
    solver->set_preconditioner( prec.get() );
    return solver;
}

template <class Solver, class Space, class Operator, class Preconditioner, class Log>
std::shared_ptr<Solver> make_linear_solver(
    const std::shared_ptr<Space> &space, const std::shared_ptr<Operator> &op,
    const std::shared_ptr<Preconditioner> &prec, Log *log
)
{
    if constexpr ( std::is_constructible<Solver, std::shared_ptr<Space>, Log *>::value )
    {
        auto solver = std::make_shared<Solver>( space, log, typename Solver::params(), prec );
        solver->set_operator( op );
        return solver;
    }
    else
        return make_linear_solver<Solver>( space.get(), op, prec, log );
}

} // namespace detail
} // namespace main_classes

#endif
