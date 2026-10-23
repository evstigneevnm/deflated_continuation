#ifndef __CONTINUATION_BORDERED_LINEAR_SOLVER_H__
#define __CONTINUATION_BORDERED_LINEAR_SOLVER_H__

#include <memory>
#include <stdexcept>
#include <utility>

namespace continuation
{

// Bind the existing scalar-bordered solver to NMFD's set_operator/solve interface.
// The underlying system must be assembled with the same original state operator.
template <class BorderedOperator, class LinearSystemSolver>
class bordered_linear_solver
{
public:
    using scalar_type          = typename BorderedOperator::scalar_type;
    using vector_type          = typename BorderedOperator::vector_type;
    using linear_operator_type = BorderedOperator;

    explicit bordered_linear_solver( std::shared_ptr<LinearSystemSolver> solver )
        : solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing bordered solver" ) )
    {
    }

    void set_operator( std::shared_ptr<const linear_operator_type> op )
    {
        if ( !op )
            throw std::invalid_argument( "Missing bordered Jacobian" );
        solver_->get_linsolver_handle_original()->set_operator( op->state_operator );
        operator_ = std::move( op );
    }

    bool solve( const vector_type &rhs, vector_type &correction )
    {
        if ( !operator_ )
            throw std::logic_error( "Bind the bordered operator before solving" );
        const auto &op        = *operator_;
        scalar_type parameter = 0;
        if ( !solver_->solve(
                 *op.state_operator, *op.constraint_row, *op.parameter_column, op.constraint_parameter, rhs.first,
                 rhs.second[0], correction.first, parameter
             ) )
            return false;
        correction.second[0] = parameter;
        return true;
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
