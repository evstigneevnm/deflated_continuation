#ifndef __NUMERICAL_ALGOS_PARAMETERIZED_CORRECTION_H__
#define __NUMERICAL_ALGOS_PARAMETERIZED_CORRECTION_H__

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nonlinear_operators/projected_operator_helpers.h>

namespace numerical_algos
{

template <class Problem, class Vector>
struct problem_projector
{
    std::shared_ptr<Problem> problem;
    explicit problem_projector( std::shared_ptr<Problem> op )
        : problem( op ? std::move( op ) : throw std::invalid_argument( "Missing nonlinear problem" ) )
    {
    }
    void apply( Vector &state )
    {
        nonlinear_operators::detail::project_state( problem.get(), state );
    }
};

// Application boundary only: bind parameters or pack (state, parameter), then
// commit the candidate on success. Newton and globalization belong to Solver.
template <class StateSpace, class Problem, class Operator, class Solver>
class parameterized_correction
{
public:
    using scalar_type                       = typename StateSpace::scalar_type;
    using vector_type                       = typename StateSpace::vector_type;
    using solver_type                       = Solver;
    using solver_space_type                 = typename Solver::vector_space_type;
    using solver_vector_type                = typename Solver::vector_type;
    static constexpr bool coupled_parameter = !std::is_same_v<vector_type, solver_vector_type>;

    parameterized_correction(
        std::shared_ptr<StateSpace> operations, std::shared_ptr<Problem> problem, std::shared_ptr<Operator> op,
        std::shared_ptr<solver_space_type> space, std::shared_ptr<Solver> solver
    )
        : operations_( operations ? std::move( operations ) : throw std::invalid_argument( "Missing state space" ) ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "Missing nonlinear problem" ) ),
          operator_( op ? std::move( op ) : throw std::invalid_argument( "Missing correction operator" ) ),
          space_( space ? std::move( space ) : throw std::invalid_argument( "Missing solver space" ) ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing nonlinear solver" ) ),
          candidate_( *space_ )
    {
        candidate_.init();
        candidate_.start_use();
    }

    bool solve( Problem *problem, vector_type &state, scalar_type &parameter )
    {
        return solve( problem, state, parameter, state, parameter );
    }

    bool solve( Problem *problem, vector_type &state, const scalar_type &parameter )
    {
        static_assert( !coupled_parameter, "A continuation parameter must be mutable" );
        scalar_type fixed = parameter;
        return solve( problem, state, parameter, state, fixed );
    }

    bool solve( Problem *problem, const vector_type &initial, const scalar_type &parameter, vector_type &state )
    {
        scalar_type fixed = parameter;
        return solve( problem, initial, parameter, state, fixed );
    }

    bool solve(
        Problem *problem, const vector_type &initial, const scalar_type &parameter, vector_type &state,
        scalar_type &result_parameter
    )
    {
        if ( problem != problem_.get() )
            throw std::invalid_argument( "Correction is bound to a different nonlinear problem" );
        if constexpr ( coupled_parameter )
        {
            operations_->assign( initial, ( *candidate_ ).first );
            ( *candidate_ ).second[0] = parameter;
        }
        else
        {
            operator_->set_parameter( parameter );
            operations_->assign( initial, *candidate_ );
        }
        if ( !solver_->solve( *candidate_ ) )
        {
            if ( &initial != &state )
                operations_->assign( initial, state );
            return false;
        }
        if constexpr ( coupled_parameter )
        {
            operations_->assign( ( *candidate_ ).first, state );
            result_parameter = ( *candidate_ ).second[0];
        }
        else
        {
            operations_->assign( *candidate_, state );
            result_parameter = parameter;
        }
        return true;
    }

    void set_tangent_space(
        const vector_type &state, scalar_type parameter, const vector_type &tangent, scalar_type tangent_parameter,
        scalar_type step, char type
    )
    {
        operator_->set_tangent_space( state, parameter, tangent, tangent_parameter, step, type );
    }

    auto get_convergence_strategy_handle()
    {
        return solver_->get_convergence_strategy_handle();
    }

private:
    std::shared_ptr<StateSpace>                                operations_;
    std::shared_ptr<Problem>                                   problem_;
    std::shared_ptr<Operator>                                  operator_;
    std::shared_ptr<solver_space_type>                         space_;
    std::shared_ptr<Solver>                                    solver_;
    nmfd::detail::vector_wrap<solver_space_type, false, false> candidate_;
};

}

#endif
