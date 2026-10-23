#ifndef __CONTINUATION_ARCLENGTH_CONSTRAINT_H__
#define __CONTINUATION_ARCLENGTH_CONSTRAINT_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>

namespace continuation
{

template <class VectorOperations>
typename VectorOperations::scalar_type evaluate_arclength_constraint(
    const VectorOperations &operations, const typename VectorOperations::vector_type &reference,
    typename VectorOperations::scalar_type reference_parameter, const typename VectorOperations::vector_type &tangent,
    typename VectorOperations::scalar_type tangent_parameter, typename VectorOperations::scalar_type step,
    const typename VectorOperations::vector_type &state, typename VectorOperations::scalar_type parameter,
    typename VectorOperations::vector_type &work
)
{
    using T = typename VectorOperations::scalar_type;
    operations.assign_mul( T( 1 ), state, T( -1 ), reference, work );
    return operations.scalar_prod( work, tangent ) + ( parameter - reference_parameter ) * tangent_parameter - step;
}

// Linear arclength in the supplied coordinates. A nonlinear chart requires
// its own constraint apply()/linearize(), including the chart derivative.
template <class VectorOperations>
class arclength_constraint
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit arclength_constraint( std::shared_ptr<VectorOperations> operations )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Missing vector operations" )
          ),
          reference_( *operations_ ), tangent_( *operations_ ), work_( *operations_ )
    {
        for ( auto *storage : { &reference_, &tangent_, &work_ } )
        {
            storage->init();
            storage->start_use();
        }
    }

    // Copy predictor/anchor data once per corrector, not on every residual trial.
    void set_tangent_space(
        const vector_type &reference, scalar_type parameter, const vector_type &tangent, scalar_type tangent_parameter,
        scalar_type step, char type = 'S'
    )
    {
        if ( type != 'S' && type != 'O' )
            throw std::invalid_argument( "Arclength type must be S or O" );
        configured_ = false;
        operations_->assign( reference, *reference_ );
        operations_->assign( tangent, *tangent_ );
        parameter_         = parameter;
        tangent_parameter_ = tangent_parameter;
        step_              = type == 'S' ? step : scalar_type( 0 );
        configured_        = true;
    }

    scalar_type apply( const vector_type &state, scalar_type parameter )
    {
        require_configured();
        return evaluate_arclength_constraint(
            *operations_, *reference_, parameter_, *tangent_, tangent_parameter_, step_, state, parameter, *work_
        );
    }

    // Write the row of the same scalar residual, in the same coordinates.
    void linearize( const vector_type &, scalar_type, vector_type &row, scalar_type &parameter_derivative )
    {
        require_configured();
        operations_->assign( *tangent_, row );
        parameter_derivative = tangent_parameter_;
    }

private:
    using storage_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations> operations_;
    storage_type                      reference_, tangent_, work_;
    scalar_type                       parameter_ = 0, tangent_parameter_ = 0, step_ = 0;
    bool                              configured_ = false;

    void require_configured() const
    {
        if ( !configured_ )
            throw std::logic_error( "Set the arclength reference and tangent before solving" );
    }
};

}

#endif
