#ifndef __DEFLATION_DEFLATED_OPERATOR_H__
#define __DEFLATION_DEFLATED_OPERATOR_H__

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nonlinear_operators/projected_operator_helpers.h>

namespace deflation
{

// Select the physical residual for NMFD's convergence and trial assessment only.
struct original_residual_norm
{
    template <class Space, class Operator>
    static auto
    evaluate( Space &, Operator &op, const typename Space::vector_type &state, typename Space::vector_type &work )
    {
        return op.original_residual_norm( state, work );
    }
};

template <class VectorOperations, class NonlinearOperator, class LinearOperator, class SolutionStorage>
class deflated_operator
{
public:
    using scalar_type          = typename VectorOperations::scalar_type;
    using vector_type          = typename VectorOperations::vector_type;
    using residual_vector_type = vector_type;
    using storage_type         = nmfd::detail::vector_wrap<VectorOperations, false, false>;

    struct jacobi_operator_type
    {
        using scalar_type = typename VectorOperations::scalar_type;
        using vector_type = typename VectorOperations::vector_type;
        std::shared_ptr<VectorOperations>     operations;
        std::shared_ptr<const LinearOperator> state_operator;
        storage_type                          residual, negative_gradient;
        scalar_type                           multiplier = 1;

        jacobi_operator_type( std::shared_ptr<VectorOperations> space, std::shared_ptr<const LinearOperator> op )
            : operations( std::move( space ) ), state_operator( std::move( op ) ), residual( *operations ),
              negative_gradient( *operations )
        {
            residual.init();
            residual.start_use();
            negative_gradient.init();
            negative_gradient.start_use();
        }

        // J_D v = m J_F v - F <c,v>, where storage supplies c = -grad(m).
        bool apply( const vector_type &in, vector_type &out ) const
        {
            if ( &in == &out )
                throw std::invalid_argument( "Deflated Jacobian requires distinct input and output" );
            if constexpr ( std::is_void_v<decltype( state_operator->apply( in, out ) )> )
                state_operator->apply( in, out );
            else if ( !state_operator->apply( in, out ) )
                return false;
            operations->assign_lin_comb(
                multiplier, out, -operations->scalar_prod( *negative_gradient, in ), *residual, out
            );
            return true;
        }
    };

    deflated_operator(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<NonlinearOperator> problem,
        std::shared_ptr<const LinearOperator> jacobian, std::shared_ptr<SolutionStorage> solutions,
        scalar_type parameter, scalar_type exclusion_distance
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Missing vector operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "Missing nonlinear problem" ) ),
          solutions_( solutions ? std::move( solutions ) : throw std::invalid_argument( "Missing solution storage" ) ),
          parameter_( parameter ), exclusion_distance_( exclusion_distance ),
          jacobian_(
              std::make_shared<jacobi_operator_type>(
                  operations_,
                  jacobian ? std::move( jacobian ) : throw std::invalid_argument( "Missing Jacobian operator" )
              )
          ),
          gradient_work_( *operations_ )
    {
        if ( !std::isfinite( exclusion_distance_ ) || exclusion_distance_ < scalar_type( 0 ) )
            throw std::invalid_argument( "Deflation exclusion distance must be finite and nonnegative" );
        gradient_work_.init();
        gradient_work_.start_use();
    }

    void set_parameter( scalar_type parameter )
    {
        parameter_ = parameter;
        prepared_  = false;
    }

    scalar_type parameter() const
    {
        return parameter_;
    }

    void apply( const vector_type &state, vector_type &residual )
    {
        physical_residual( state, residual );
        scalar_type multiplier;
        solutions_->calc_distance( state, multiplier, *gradient_work_ );
        operations_->scale( multiplier, residual );
    }

    // Reuse the strategy's work vector. Excluded roots and overflowing D are not convergence.
    scalar_type original_residual_norm( const vector_type &state, vector_type &work )
    {
        scalar_type multiplier;
        solutions_->calc_distance( state, multiplier, *gradient_work_ );
        if ( !std::isfinite( multiplier ) || solutions_->last_minimum_distance() <= exclusion_distance_ )
            return std::numeric_limits<scalar_type>::infinity();
        physical_residual( state, work );
        const scalar_type norm = operations_->norm_l2( work );
        return std::isfinite( multiplier * norm ) ? norm : std::numeric_limits<scalar_type>::infinity();
    }

    void set_linearization_point( const vector_type &state )
    {
        prepared_ = false;
        nonlinear_operators::detail::set_linearization_point( problem_.get(), state, parameter_ );
        nonlinear_operators::detail::residual_at_linearization(
            problem_.get(), state, parameter_, *jacobian_->residual
        );
        solutions_->calc_distance( state, jacobian_->multiplier, *jacobian_->negative_gradient );
        prepared_ = true;
    }

    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        if ( !prepared_ )
            throw std::logic_error( "Prepare the deflated Jacobian before requesting it" );
        return jacobian_;
    }

    struct trial_projector
    {
        std::shared_ptr<NonlinearOperator> problem;
        explicit trial_projector( std::shared_ptr<NonlinearOperator> op )
            : problem( op ? std::move( op ) : throw std::invalid_argument( "Missing nonlinear problem" ) )
        {
        }
        void apply( vector_type &trial )
        {
            nonlinear_operators::detail::project_state( problem.get(), trial );
        }
    };

private:
    void physical_residual( const vector_type &state, vector_type &residual )
    {
        if ( prepared_ )
            nonlinear_operators::detail::residual_in_frozen_chart( problem_.get(), state, parameter_, residual );
        else
            nonlinear_operators::detail::residual( problem_.get(), state, parameter_, residual );
    }

    std::shared_ptr<VectorOperations>     operations_;
    std::shared_ptr<NonlinearOperator>    problem_;
    std::shared_ptr<SolutionStorage>      solutions_;
    scalar_type                           parameter_, exclusion_distance_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    storage_type                          gradient_work_;
    bool                                  prepared_ = false;
};

}

#endif
