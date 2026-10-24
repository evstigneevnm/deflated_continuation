#ifndef __MAIN_NMFD_NEWTON_TYPES_H__
#define __MAIN_NMFD_NEWTON_TYPES_H__

#include <memory>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nonlinear_operators/fixed_parameter_operator.h>
#include <numerical_algos/parameterized_correction.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <continuation/corrector_operator.h>
#include <continuation/bordered_linear_solver.h>
#include <deflation/deflated_operator.h>
#include <deflation/fixed_damping_convergence_strategy.h>
#include <deflation/rank_one_linear_solver.h>
#include <main/solver_construction.h>

namespace main_classes
{
namespace detail
{

// Match template arity explicitly: NVCC does not accept the defaulted sixth argument here.
template <class Space, class Log, class Operator, class Projector, class Quality>
using newton_convergence = nmfd::solvers::default_convergence_strategy<Space, Log, Operator, Projector, Quality>;

template <class Space, class Log, class Operator, class Projector, class Quality>
using continuation_convergence = nmfd::solvers::default_convergence_strategy<
    Space, Log, Operator, Projector, Quality, continuation::corrector_residual_norm>;

template <class Space, class Log, class Operator, class Projector, class Quality>
using deflation_convergence = nmfd::solvers::default_convergence_strategy<
    Space, Log, Operator, Projector, Quality, deflation::deflated_residual_norm>;

template <class Space, class Log, class Operator, class Projector, class Quality>
using fixed_damping_deflation_convergence =
    deflation::fixed_damping_convergence_strategy<Space, Log, Operator, Projector, Quality>;

template <class Strategy, class Parameters>
void configure_deflation_strategy( Strategy *strategy, const Parameters &parameters )
{
    using T = typename Strategy::vector_space_type::scalar_type;
    strategy->set_convergence_constants(
        parameters.tolerance, parameters.newton_max_it, T( 1 ), 0, parameters.newton_wight,
        parameters.store_norms_history, parameters.verbose, 10, T( 1e16 )
    );
}

template <class Space, class Log, class Operator, class Projector, class Quality, class Parameters>
void configure_deflation_strategy(
    fixed_damping_deflation_convergence<Space, Log, Operator, Projector, Quality> *strategy,
    const Parameters                                                              &parameters
)
{
    strategy->set_convergence_constants(
        parameters.tolerance, parameters.newton_max_it, parameters.newton_wight, parameters.store_norms_history,
        parameters.verbose
    );
}

// One construction path for all roles; convergence remains a replaceable type.
template <
    class StateSpace, class Space, class Log, class Problem, class Operator, class Linear, class Projector,
    template <class, class, class, class, class> class Convergence>
struct nmfd_correction_method
{
    using recovery_type    = numerical_algos::recovering_linear_solver<Space, Linear>;
    using iteration_type   = nmfd::solvers::newton_iteration<Space, Operator, recovery_type>;
    using quality_type     = nmfd::operations::zero_functional<Space>;
    using convergence_type = Convergence<Space, Log, Operator, Projector, quality_type>;
    using solver_type      = nmfd::solvers::nonlinear_solver<
             Space, Log, Operator, iteration_type, Projector, quality_type, convergence_type>;
    using correction_type = numerical_algos::parameterized_correction<StateSpace, Problem, Operator, solver_type>;

    static auto make(
        const std::shared_ptr<StateSpace> &operations, const std::shared_ptr<Problem> &problem,
        const std::shared_ptr<Operator> &op, const std::shared_ptr<Space> &space, const std::shared_ptr<Linear> &linear,
        Log *log, std::shared_ptr<Projector> projector
    )
    {
        auto recovery    = std::make_shared<recovery_type>( space, linear );
        auto iteration   = std::make_shared<iteration_type>( space, recovery );
        auto convergence = std::make_shared<convergence_type>( space, log );
        auto solver      = std::make_shared<solver_type>( space, log, iteration, convergence );
        solver->set_nonlinear_operator( op );
        solver->set_project_operator( std::move( projector ) );
        return std::make_shared<correction_type>( operations, problem, op, space, solver );
    }
};

}

template <
    class Space, class Log, class Problem, class Jacobian, class Linear,
    template <class, class, class, class, class> class Convergence = detail::newton_convergence>
struct nmfd_fixed_newton_types
{
    using scalar_type         = typename Space::scalar_type;
    using fixed_operator_type = nonlinear_operators::fixed_parameter_operator<Space, Problem, Jacobian>;
    using projector_type      = numerical_algos::problem_projector<Problem, typename Space::vector_type>;
    using fixed_method        = detail::nmfd_correction_method<
               Space, Space, Log, Problem, fixed_operator_type, Linear, projector_type, Convergence>;
    using fixed_convergence_type = typename fixed_method::convergence_type;
    using fixed_correction_type  = typename fixed_method::correction_type;

    template <class Parameters>
    static void configure_fixed( fixed_convergence_type *strategy, const Parameters &parameters )
    {
        strategy->set_convergence_constants(
            parameters.tolerance, parameters.newton_max_it, scalar_type( 1 ), 0, parameters.newton_wight,
            parameters.store_norms_history, parameters.verbose
        );
    }
};

template <
    class Space, class Log, class Problem, class Jacobian, class LinearSystem, class Storage,
    template <class, class, class, class, class> class FixedConvergence        = detail::newton_convergence,
    template <class, class, class, class, class> class ContinuationConvergence = detail::continuation_convergence,
    template <class, class, class, class, class> class DeflationConvergence    = detail::deflation_convergence>
struct nmfd_newton_types
    : nmfd_fixed_newton_types<
          Space, Log, Problem, Jacobian, typename LinearSystem::linear_solver_original_type, FixedConvergence>
{
    using scalar_type             = typename Space::scalar_type;
    using corrector_operator_type = continuation::corrector_operator<Space, Problem, Jacobian>;
    using pair_space_type         = typename corrector_operator_type::vector_space_type;
    using bordered_type =
        continuation::bordered_linear_solver<typename corrector_operator_type::jacobi_operator_type, LinearSystem>;
    using continuation_method = detail::nmfd_correction_method<
        Space, pair_space_type, Log, Problem, corrector_operator_type, bordered_type,
        typename corrector_operator_type::trial_projector, ContinuationConvergence>;
    using continuation_convergence_type = typename continuation_method::convergence_type;
    using continuation_correction_type  = typename continuation_method::correction_type;
    using deflated_operator_type        = deflation::deflated_operator<Space, Problem, Jacobian, Storage>;
    using rank_one_type =
        deflation::rank_one_linear_solver<typename deflated_operator_type::jacobi_operator_type, LinearSystem>;
    using deflation_method = detail::nmfd_correction_method<
        Space, Space, Log, Problem, deflated_operator_type, rank_one_type,
        typename deflated_operator_type::trial_projector, DeflationConvergence>;
    using deflation_convergence_type = typename deflation_method::convergence_type;
    using deflation_correction_type  = typename deflation_method::correction_type;

    struct state
    {
        std::shared_ptr<corrector_operator_type>      corrector_operator;
        std::shared_ptr<continuation_correction_type> corrector;
        std::shared_ptr<deflated_operator_type>       deflated_operator;
        std::shared_ptr<deflation_correction_type>    deflated;
    };

    template <class Parameters>
    static void configure_continuation( continuation_convergence_type *strategy, const Parameters &parameters )
    {
        strategy->set_convergence_constants(
            parameters.tolerance, parameters.newton_max_it, parameters.relax_tolerance_factor,
            parameters.relax_tolerance_steps, parameters.newton_wight, parameters.store_norms_history,
            parameters.verbose, parameters.stagnation_max, parameters.maximum_norm_increase,
            parameters.newton_wight_threshold
        );
    }

    template <class Parameters>
    static void configure_deflation( deflation_convergence_type *strategy, const Parameters &parameters )
    {
        detail::configure_deflation_strategy( strategy, parameters );
    }
};

}

#endif
