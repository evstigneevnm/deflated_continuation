#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_STABILITY_LINEAR_OPERATOR_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_STABILITY_LINEAR_OPERATOR_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>

namespace symmetry
{
namespace linearization
{

/**
 * Quotient-space linearization for a stationary state.
 *
 * The physical tangent block is P J P. A configurable scalar completion is
 * applied to the gauge complement so the otherwise neutral group direction
 * can be placed strictly inside the stable half-plane.
 */
template <class VectorOperations, class NonlinearOperator>
class projected_stability_linear_operator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit projected_stability_linear_operator(
        NonlinearOperator *nonlinear_operator, scalar_type gauge_completion = scalar_type( 1 )
    )
        : nonlinear_operator_(
              nonlinear_operator ? nonlinear_operator
                                 : throw std::invalid_argument( "projected stability requires a problem" )
          ),
          vector_operations_( nonlinear_operator_->get_vec_ops_ref() ), gauge_completion_( gauge_completion ),
          projected_storage_( *vector_operations_ ), gauge_storage_( *vector_operations_ )
    {
        projected_storage_.init();
        projected_storage_.start_use();
        gauge_storage_.init();
        gauge_storage_.start_use();
    }

    explicit projected_stability_linear_operator(
        std::shared_ptr<NonlinearOperator> problem, scalar_type gauge_completion = scalar_type( 1 )
    )
        : projected_stability_linear_operator( problem.get(), gauge_completion )
    {
        problem_owner_ = std::move( problem );
    }

    projected_stability_linear_operator( const projected_stability_linear_operator & )            = delete;
    projected_stability_linear_operator &operator=( const projected_stability_linear_operator & ) = delete;

    void apply( const vector_type &source, vector_type &destination ) const
    {
        nonlinear_operator_->project_current_tangent( source, projected_source_ );
        nonlinear_operator_->jacobian_u( projected_source_, destination );
        nonlinear_operator_->project_current_tangent( destination, destination );

        if ( gauge_completion_ != scalar_type( 0 ) )
        {
            vector_operations_->assign_lin_comb(
                scalar_type( 1 ), source, scalar_type( -1 ), projected_source_, gauge_component_
            );
            vector_operations_->add_mul( gauge_completion_, gauge_component_, destination );
        }
    }

    scalar_type gauge_completion() const
    {
        return gauge_completion_;
    }

private:
    std::shared_ptr<NonlinearOperator>                                problem_owner_;
    NonlinearOperator                                                *nonlinear_operator_;
    VectorOperations                                                 *vector_operations_;
    scalar_type                                                       gauge_completion_;
    mutable nmfd::detail::vector_wrap<VectorOperations, false, false> projected_storage_, gauge_storage_;
    vector_type                                                      &projected_source_ = *projected_storage_;
    vector_type                                                      &gauge_component_  = *gauge_storage_;
};

} // namespace linearization
} // namespace symmetry

#endif
