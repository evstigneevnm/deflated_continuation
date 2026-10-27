#ifndef CONTINUATION_AUGMENTED_LINEAR_SOLVER_H
#define CONTINUATION_AUGMENTED_LINEAR_SOLVER_H

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/preconditioners/block_schur.h>
#include <nmfd/preconditioners/small_schur_inverse.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include <continuation/corrector_operator.h>

namespace continuation
{

template <class Space, class Monitor, class Log, class Operator, class Preconditioner>
using augmented_gmres = nmfd::solvers::gmres<Space, Monitor, Log, Operator, Preconditioner>;

// Application solve boundary: the operator is already augmented; no scalar elimination.
template <
    class Space, class Operator, class FirstInverse, class Log,
    template <class, class, class, class, class> class Krylov = augmented_gmres>
class augmented_linear_solver
{
    using access_type = corrector_block_access<Space, Operator>;
    using schur_type  = nmfd::preconditioners::schur_complement_operator<Space, Operator, FirstInverse, access_type>;
    using second_space_type  = typename Space::second_space_type;
    using schur_inverse_type = nmfd::preconditioners::small_schur_inverse<second_space_type, schur_type>;
    using schur_preconditioner_type =
        nmfd::preconditioners::block_schur<Space, Operator, FirstInverse, schur_inverse_type, access_type>;
    struct preconditioner_type : schur_preconditioner_type
    {
        using schur_preconditioner_type::schur_preconditioner_type;
        bool bypass = false;
        bool apply( typename Space::vector_type &in_out ) const
        {
            return bypass || schur_preconditioner_type::apply( in_out );
        }
    };
    using monitor_type = nmfd::solvers::monitor_krylov<Space, Log>;
    using solver_type  = Krylov<Space, monitor_type, Log, Operator, preconditioner_type>;

public:
    using scalar_type          = typename Space::scalar_type;
    using vector_type          = typename Space::vector_type;
    using linear_operator_type = Operator;

    augmented_linear_solver( std::shared_ptr<Space> space, std::shared_ptr<FirstInverse> inverse, Log *log = nullptr )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing augmented space" ) ),
          preconditioner_(
              std::make_shared<preconditioner_type>(
                  space_, std::move( inverse ), std::make_shared<schur_inverse_type>( space_->second() )
              )
          ),
          solver_( space_, log, typename solver_type::params(), preconditioner_ ), candidate_( *space_ ),
          residual_( *space_ )
    {
        candidate_.init();
        candidate_.start_use();
        residual_.init();
        residual_.start_use();
    }

    auto &monitor()
    {
        return solver_.monitor();
    }

    void set_operator( std::shared_ptr<const Operator> op )
    {
        if ( !op )
            throw std::invalid_argument( "Missing augmented Jacobian" );
        solver_.set_operator( op );
        operator_ = std::move( op );
    }

    void set_preconditioner_bypass( bool enabled )
    {
        bypass_                 = enabled;
        preconditioner_->bypass = enabled;
    }

    bool preconditioner_bypass_enabled() const
    {
        return bypass_;
    }

    // Recovery is selected by the caller; failed attempts leave correction unchanged.
    bool solve( const vector_type &rhs, vector_type &correction )
    {
        if ( !operator_ )
            throw std::logic_error( "Bind the augmented Jacobian before solving" );
        if ( ( !bypass_ && !preconditioner_->ready() ) || !space_->is_valid_number( rhs ) )
            return false;
        space_->assign_scalar( scalar_type( 0 ), *candidate_ );
        if ( !solver_.solve( rhs, *candidate_ ) || !space_->is_valid_number( *candidate_ ) ||
             !operator_->apply( *candidate_, *residual_ ) )
            return false;
        space_->add_lin_comb( scalar_type( -1 ), rhs, scalar_type( 1 ), *residual_ );
        const auto tolerance = monitor().abs_tol() + monitor().rel_tol() * space_->norm( rhs );
        const auto norm      = space_->norm( *residual_ );
        if ( !std::isfinite( norm ) || !std::isfinite( tolerance ) || !( norm <= tolerance ) )
            return false;
        space_->assign( *candidate_, correction );
        return true;
    }

private:
    std::shared_ptr<Space>                         space_;
    std::shared_ptr<preconditioner_type>           preconditioner_;
    solver_type                                    solver_;
    nmfd::detail::vector_wrap<Space, false, false> candidate_, residual_;
    std::shared_ptr<const Operator>                operator_;
    bool                                           bypass_ = false;
};

}

#endif
