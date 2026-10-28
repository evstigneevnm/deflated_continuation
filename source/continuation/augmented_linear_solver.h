#ifndef CONTINUATION_AUGMENTED_LINEAR_SOLVER_H
#define CONTINUATION_AUGMENTED_LINEAR_SOLVER_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/backend/math.h>
#include <nmfd/solvers/krylov/operator_apply.h>
#include <numerical_algos/bypassable_preconditioner.h>

namespace continuation
{

// Application solve boundary: the operator is already augmented; no scalar elimination.
template <class Space, class Operator, class LinearSolver>
class augmented_linear_solver
{
    using math_type = nmfd::operations::backend::math<void, typename Space::norm_type>;

public:
    using scalar_type          = typename Space::scalar_type;
    using vector_type          = typename Space::vector_type;
    using linear_operator_type = Operator;
    using linear_solver_type   = LinearSolver;

    augmented_linear_solver(
        std::shared_ptr<Space> space, std::shared_ptr<LinearSolver> solver,
        std::shared_ptr<numerical_algos::preconditioner_bypass> retry = nullptr
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing augmented space" ) ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing augmented linear solver" ) ),
          retry_( std::move( retry ) ), candidate_( *space_ ), residual_( *space_ )
    {
        candidate_.init();
        candidate_.start_use();
        residual_.init();
        residual_.start_use();
    }

    auto &monitor()
    {
        return solver_->monitor();
    }

    // Configure the selected algorithm directly; no forwarding setters or ownership copies.
    LinearSolver &linear_solver()
    {
        return *solver_;
    }

    void set_operator( std::shared_ptr<const Operator> op )
    {
        if ( !op )
            throw std::invalid_argument( "Missing augmented Jacobian" );
        solver_->set_operator( op );
        operator_ = std::move( op );
    }

    bool supports_unpreconditioned_retry() const
    {
        return retry_ && !retry_->enabled;
    }

    void set_preconditioner_bypass( bool enabled )
    {
        if ( !retry_ && enabled )
            throw std::logic_error( "This solve has no bypassable preconditioner" );
        if ( retry_ )
            retry_->enabled = enabled;
    }

    bool preconditioner_bypass_enabled() const
    {
        return retry_ && retry_->enabled;
    }

    // Recovery is selected by the caller; failed attempts leave correction unchanged.
    bool solve( const vector_type &rhs, vector_type &correction )
    {
        if ( !operator_ )
            throw std::logic_error( "Bind the augmented Jacobian before solving" );
        if ( !space_->is_valid_number( rhs ) )
            return false;
        space_->assign_scalar( scalar_type( 0 ), *candidate_ );
        if ( !solver_->solve( rhs, *candidate_ ) || !space_->is_valid_number( *candidate_ ) ||
             !nmfd::solvers::krylov::apply_operator( *operator_, *candidate_, *residual_ ) )
            return false;
        space_->add_lin_comb( scalar_type( -1 ), rhs, scalar_type( 1 ), *residual_ );
        const auto tolerance = monitor().abs_tol() + monitor().rel_tol() * space_->norm( rhs );
        const auto norm      = space_->norm( *residual_ );
        if ( !math_type::is_finite( norm ) || !math_type::is_finite( tolerance ) || !( norm <= tolerance ) )
            return false;
        space_->assign( *candidate_, correction );
        return true;
    }

private:
    std::shared_ptr<Space>                                  space_;
    std::shared_ptr<LinearSolver>                           solver_;
    std::shared_ptr<numerical_algos::preconditioner_bypass> retry_;
    nmfd::detail::vector_wrap<Space, false, false>          candidate_, residual_;
    std::shared_ptr<const Operator>                         operator_;
};

}

#endif
