#ifndef CONTINUATION_AUGMENTED_TANGENT_H
#define CONTINUATION_AUGMENTED_TANGENT_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <continuation/tangent_normalization.h>
#include <numerical_algos/linear_solve_recovery.h>

namespace continuation
{

// Application formulation for [J F_parameter; row row_parameter] * tangent = [0; 1].
template <class Corrector, class LinearSolver>
class augmented_tangent
{
    using space_type       = typename Corrector::vector_space_type;
    using state_space_type = typename space_type::first_space_type;

public:
    using scalar_type = typename Corrector::scalar_type;
    using vector_type = typename Corrector::state_vector_type;

    augmented_tangent( std::shared_ptr<Corrector> op, std::shared_ptr<LinearSolver> solver )
        : op_( op ? std::move( op ) : throw std::invalid_argument( "Missing tangent operator" ) ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "Missing tangent solver" ) ),
          rhs_( *op_->space() ), candidate_( *op_->space() )
    {
        rhs_.init();
        rhs_.start_use();
        candidate_.init();
        candidate_.start_use();
    }

    bool apply(
        const vector_type &state, scalar_type parameter, const vector_type &row, scalar_type row_parameter,
        vector_type &tangent, scalar_type &tangent_parameter, scalar_type tolerance, int maximum_iterations
    )
    {
        op_->prepare_tangent( state, parameter, row, row_parameter );
        solver_->set_operator( op_->get_jacobi_operator() );
        auto &space = *op_->space();
        space.assign_scalar( scalar_type( 0 ), *rhs_ );
        ( *rhs_ ).second[0] = scalar_type( 1 );
        space.assign_scalar( scalar_type( 0 ), *candidate_ );
        auto &monitor = solver_->monitor();
        monitor.set_temp_tolerance( tolerance );
        monitor.set_temp_max_iterations( maximum_iterations );
        struct restore_monitor
        {
            decltype( monitor ) reference;
            ~restore_monitor()
            {
                reference.restore_max_iterations();
                reference.restore_tolerance();
            }
        } guard{ monitor };
        const bool solved = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            solver_.get(), [this]() { return solver_->solve( *rhs_, *candidate_ ); },
            [this, &space]() { space.assign_scalar( scalar_type( 0 ), *candidate_ ); }
        );
        auto       &result     = *candidate_;
        const auto &operations = space.first_space();
        if ( !solved || !normalize_rank1_tangent( &operations, result.first, result.second[0] ) )
            return false;
        operations.assign( result.first, tangent );
        tangent_parameter = result.second[0];
        return true;
    }

    // Preserve parameter-oriented initialization when regular; a state row covers a fold.
    bool initial(
        const vector_type &state, scalar_type parameter, scalar_type sign, vector_type &tangent,
        scalar_type &tangent_parameter
    )
    {
        const auto &operations = op_->space()->first_space();
        auto       &row        = ( *rhs_ ).first;
        operations.assign_scalar( scalar_type( 0 ), row );
        const scalar_type tolerance = scalar_type( 1e-8 ) * operations.get_l2_size();
        if ( apply( state, parameter, row, sign, tangent, tangent_parameter, tolerance, 10000 ) )
            return true;
        operations.assign_scalar( sign, row );
        return apply( state, parameter, row, scalar_type( 0 ), tangent, tangent_parameter, tolerance, 10000 );
    }

private:
    std::shared_ptr<Corrector>                          op_;
    std::shared_ptr<LinearSolver>                       solver_;
    nmfd::detail::vector_wrap<space_type, false, false> rhs_, candidate_;
};

}

#endif
