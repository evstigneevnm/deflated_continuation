#ifndef __DEFLATION_RANK_ONE_LINEAR_SOLVER_H__
#define __DEFLATION_RANK_ONE_LINEAR_SOLVER_H__

#include <memory>
#include <stdexcept>
#include <utility>

namespace deflation
{

// Bind the existing Sherman-Morrison rank-one solve, without another Krylov loop.
template <class DeflatedOperator, class LinearSystemSolver>
class rank_one_linear_solver
{
public:
    using scalar_type          = typename DeflatedOperator::scalar_type;
    using vector_type          = typename DeflatedOperator::vector_type;
    using linear_operator_type = DeflatedOperator;

    explicit rank_one_linear_solver( std::shared_ptr<LinearSystemSolver> solver )
        : solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing rank-one solver" ) )
    {
    }

    void set_operator( std::shared_ptr<const linear_operator_type> op )
    {
        if ( !op )
            throw std::invalid_argument( "Missing deflated Jacobian" );
        solver_->get_linsolver_handle_original()->set_operator( op->state_operator );
        operator_ = std::move( op );
    }

    bool solve( const vector_type &rhs, vector_type &correction )
    {
        if ( !operator_ )
            throw std::logic_error( "Bind the rank-one operator before solving" );
        const auto &op = *operator_;
        return solver_->solve(
            op.multiplier, *op.state_operator, scalar_type( 1 ), *op.negative_gradient, *op.residual, rhs, correction
        );
    }

    void set_preconditioner_bypass( bool enabled )
    {
        solver_->set_preconditioner_bypass( enabled );
    }

private:
    std::shared_ptr<LinearSystemSolver>         solver_;
    std::shared_ptr<const linear_operator_type> operator_;
};

}

#endif
