#ifndef __CONTINUATION_CORRECTOR_OPERATOR_H__
#define __CONTINUATION_CORRECTOR_OPERATOR_H__

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <scfd/static_vec/vec.h>
#include <common/scalar_math.h>
#include <continuation/arclength_constraint.h>
#include <continuation/chart_helpers.h>
#include <nonlinear_operators/projected_operator_helpers.h>

namespace continuation
{

// Match the legacy corrector merit without changing the bordered algebra.
struct corrector_residual_norm
{
    template <class Space, class Operator>
    static auto evaluate(
        const Space &space, Operator &op, const typename Space::vector_type &state,
        typename Space::vector_type &residual
    )
    {
        op.apply( state, residual );
        const auto state_norm      = space.first()->norm_l2( residual.first );
        const auto constraint_norm = space.second()->norm_l2( residual.second );
        return common::scalar_math::sqrt( state_norm * state_norm + constraint_norm * constraint_norm );
    }
};

template <
    class VectorOperations, class NonlinearOperator, class LinearOperator,
    class Constraint = arclength_constraint<VectorOperations>>
class corrector_operator
{
public:
    using scalar_type       = typename VectorOperations::scalar_type;
    using state_vector_type = typename VectorOperations::vector_type;
    using scalar_space_type =
        nmfd::operations::static_vector_space<scalar_type, 1, scfd::static_vec::vec<scalar_type, 1>>;
    using vector_space_type    = nmfd::operations::pair_vector_space<VectorOperations, scalar_space_type>;
    using vector_type          = typename vector_space_type::vector_type;
    using residual_vector_type = vector_type;

    struct jacobi_operator_type
    {
        using scalar_type  = typename VectorOperations::scalar_type;
        using vector_type  = typename corrector_operator::vector_type;
        using storage_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
        std::shared_ptr<VectorOperations>     operations;
        std::shared_ptr<const LinearOperator> state_operator;
        storage_type                          parameter_column, constraint_row;
        scalar_type                           constraint_parameter = 0;

        jacobi_operator_type( std::shared_ptr<VectorOperations> space, std::shared_ptr<const LinearOperator> op )
            : operations( std::move( space ) ), state_operator( std::move( op ) ), parameter_column( *operations ),
              constraint_row( *operations )
        {
            parameter_column.init();
            parameter_column.start_use();
            constraint_row.init();
            constraint_row.start_use();
        }

        // Apply [J F_parameter; constraint_state constraint_parameter] without aliasing.
        bool apply( const vector_type &in, vector_type &out ) const
        {
            if ( &in == &out )
                throw std::invalid_argument( "Bordered Jacobian requires distinct input and output" );
            if constexpr ( std::is_void_v<decltype( state_operator->apply( in.first, out.first ) )> )
                state_operator->apply( in.first, out.first );
            else if ( !state_operator->apply( in.first, out.first ) )
                return false;
            operations->add_lin_comb( in.second[0], *parameter_column, out.first );
            out.second[0] = operations->scalar_prod( *constraint_row, in.first ) + constraint_parameter * in.second[0];
            return true;
        }
    };

    corrector_operator(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<NonlinearOperator> problem,
        std::shared_ptr<const LinearOperator> jacobian
    )
        : corrector_operator(
              operations, std::move( problem ), std::move( jacobian ), std::make_shared<Constraint>( operations )
          )
    {
    }

    corrector_operator(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<NonlinearOperator> problem,
        std::shared_ptr<const LinearOperator> jacobian, std::shared_ptr<Constraint> constraint
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Missing vector operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "Missing nonlinear problem" ) ),
          constraint_(
              constraint ? std::move( constraint ) : throw std::invalid_argument( "Missing continuation constraint" )
          ),
          space_( std::make_shared<vector_space_type>( operations_, std::make_shared<scalar_space_type>() ) ),
          jacobian_(
              std::make_shared<jacobi_operator_type>(
                  operations_,
                  jacobian ? std::move( jacobian ) : throw std::invalid_argument( "Missing Jacobian operator" )
              )
          )
    {
    }

    const std::shared_ptr<vector_space_type> &space() const
    {
        return space_;
    }

    // Inputs are in the selected corrector coordinates; do not move the chart during trials.
    void set_tangent_space(
        const state_vector_type &reference, scalar_type parameter, const state_vector_type &tangent,
        scalar_type tangent_parameter, scalar_type step, char type = 'S'
    )
    {
        prepared_  = false;
        reference_ = nullptr;
        constraint_->set_tangent_space( reference, parameter, tangent, tangent_parameter, step, type );
    }

    void apply( const vector_type &state, vector_type &residual )
    {
        if ( prepared_ )
            nonlinear_operators::detail::residual_in_frozen_chart(
                problem_.get(), state.first, state.second[0], residual.first
            );
        else
            nonlinear_operators::detail::residual( problem_.get(), state.first, state.second[0], residual.first );
        residual.second[0] = constraint_->apply( state.first, state.second[0] );
    }

    void set_linearization_point( const vector_type &state )
    {
        prepared_  = false;
        reference_ = nullptr;
        nonlinear_operators::detail::set_linearization_point( problem_.get(), state.first, state.second[0] );
        nonlinear_operators::detail::jacobian_alpha( problem_.get(), *jacobian_->parameter_column );
        constraint_->linearize(
            state.first, state.second[0], *jacobian_->constraint_row, jacobian_->constraint_parameter
        );
        reference_           = &state.first;
        reference_parameter_ = state.second[0];
        prepared_            = true;
    }

    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        if ( !prepared_ )
            throw std::logic_error( "Prepare the continuation Jacobian before requesting it" );
        return jacobian_;
    }

    // Borrow the unchanged base iterate only during Newton's trial processing.
    void project_trial( vector_type &trial )
    {
        if ( !prepared_ || reference_ == &trial.first )
            throw std::logic_error( "Trial projection requires a distinct, live Newton base iterate" );
        chart::stabilize_corrector_trial(
            operations_.get(), static_cast<void *>( nullptr ), problem_.get(), *reference_, reference_parameter_,
            trial.first, trial.second[0]
        );
    }

    struct trial_projector
    {
        std::shared_ptr<corrector_operator> operation;
        explicit trial_projector( std::shared_ptr<corrector_operator> op )
            : operation( op ? std::move( op ) : throw std::invalid_argument( "Missing continuation operator" ) )
        {
        }
        void apply( vector_type &trial )
        {
            operation->project_trial( trial );
        }
    };

private:
    std::shared_ptr<VectorOperations>     operations_;
    std::shared_ptr<NonlinearOperator>    problem_;
    std::shared_ptr<Constraint>           constraint_;
    std::shared_ptr<vector_space_type>    space_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    const state_vector_type              *reference_           = nullptr;
    scalar_type                           reference_parameter_ = 0;
    bool                                  prepared_            = false;
};

}

#endif
