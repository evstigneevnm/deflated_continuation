#ifndef __NUMERICAL_ALGOS_RECOVERING_LINEAR_SOLVER_H__
#define __NUMERICAL_ALGOS_RECOVERING_LINEAR_SOLVER_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <numerical_algos/linear_solve_recovery.h>

namespace numerical_algos
{

// Keep project retry policy outside NMFD's ordinary Newton iteration.
template <class VectorOperations, class LinearSolver, class ConvergenceStrategy = void>
class recovering_linear_solver
{
public:
    using scalar_type          = typename VectorOperations::scalar_type;
    using vector_type          = typename VectorOperations::vector_type;
    using linear_operator_type = typename LinearSolver::linear_operator_type;

    recovering_linear_solver(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<LinearSolver> solver,
        std::shared_ptr<ConvergenceStrategy> convergence = nullptr
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Missing vector operations" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing linear solver" ) ),
          convergence_( std::move( convergence ) )
    {
    }

    void set_operator( std::shared_ptr<const linear_operator_type> op )
    {
        solver_->set_operator( op );
        operator_ = std::move( op );
    }

    // Retry the same linear system once without preconditioning, from zero.
    bool solve( const vector_type &rhs, vector_type &correction )
    {
        return lin_solvers::recovery::solve_with_unpreconditioned_retry(
            solver_.get(), [this, &rhs, &correction]() { return solve_attempt( rhs, correction ); },
            [this, &correction]() { operations_->assign_scalar( scalar_type( 0 ), correction ); }
        );
    }

private:
    template <class Strategy, class = void>
    struct has_inexact_assessment : std::false_type
    {
    };
    template <class Strategy>
    struct has_inexact_assessment<
        Strategy, std::void_t<decltype( std::declval<Strategy &>().accept_inexact_increment(
                      std::declval<const linear_operator_type &>(), std::declval<const vector_type &>(),
                      std::declval<const vector_type &>()
                  ) )>> : std::true_type
    {
    };

    bool solve_attempt( const vector_type &rhs, vector_type &correction )
    {
        if ( solver_->solve( rhs, correction ) )
            return true;
        // The retry clears correction, so assess the primary result first.
        if constexpr ( has_inexact_assessment<ConvergenceStrategy>::value )
        {
            if ( convergence_ && operator_ )
                return convergence_->accept_inexact_increment( *operator_, rhs, correction );
        }
        return false;
    }

    std::shared_ptr<VectorOperations>           operations_;
    std::shared_ptr<LinearSolver>               solver_;
    std::shared_ptr<ConvergenceStrategy>        convergence_;
    std::shared_ptr<const linear_operator_type> operator_;
};

}

#endif
