#ifndef __NUMERICAL_ALGOS_RECOVERING_LINEAR_SOLVER_H__
#define __NUMERICAL_ALGOS_RECOVERING_LINEAR_SOLVER_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <numerical_algos/linear_solve_recovery.h>

namespace numerical_algos
{

// Keep project retry policy outside NMFD's ordinary Newton iteration.
template <class VectorOperations, class LinearSolver>
class recovering_linear_solver
{
public:
    using scalar_type          = typename VectorOperations::scalar_type;
    using vector_type          = typename VectorOperations::vector_type;
    using linear_operator_type = typename LinearSolver::linear_operator_type;

    recovering_linear_solver( std::shared_ptr<VectorOperations> operations, std::shared_ptr<LinearSolver> solver )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Missing vector operations" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing linear solver" ) )
    {
    }

    void set_operator( std::shared_ptr<const linear_operator_type> op )
    {
        solver_->set_operator( std::move( op ) );
    }

    // Retry the same linear system once without preconditioning, from zero.
    bool solve( const vector_type &rhs, vector_type &correction )
    {
        return lin_solvers::recovery::solve_with_unpreconditioned_retry(
            solver_.get(), [this, &rhs, &correction]() { return solver_->solve( rhs, correction ); },
            [this, &correction]() { operations_->assign_scalar( scalar_type( 0 ), correction ); }
        );
    }

private:
    std::shared_ptr<VectorOperations> operations_;
    std::shared_ptr<LinearSolver>     solver_;
};

}

#endif
