#ifndef __CONTINUATION_TANGENT_NORMALIZATION_H__
#define __CONTINUATION_TANGENT_NORMALIZATION_H__

#include <algorithm>
#include <limits>

#include <common/scalar_math.h>

namespace continuation
{

// Use the augmented Euclidean pairing, even when residual norms are RMS.
template <class VectorOperations, class Vector, class Scalar>
Scalar rank1_tangent_line_similarity(
    VectorOperations *vec_ops, const Vector &reference, const Scalar reference_parameter, const Vector &source,
    const Scalar source_parameter
)
{
    if ( !common::scalar_math::isfinite( reference_parameter ) || !common::scalar_math::isfinite( source_parameter ) )
        return Scalar( 0 );
    const Scalar reference_norm = vec_ops->norm_rank1( reference, reference_parameter );
    const Scalar source_norm    = vec_ops->norm_rank1( source, source_parameter );
    if ( !common::scalar_math::isfinite( reference_norm ) || !common::scalar_math::isfinite( source_norm ) ||
         reference_norm <= Scalar( 0 ) || source_norm <= Scalar( 0 ) )
        return Scalar( 0 );
    const Scalar product = vec_ops->scalar_prod( reference, source ) + reference_parameter * source_parameter;
    return common::scalar_math::isfinite( product )
               ? std::min<Scalar>( Scalar( 1 ), common::scalar_math::abs( product / reference_norm / source_norm ) )
               : Scalar( 0 );
}

template <class VectorOperations, class Vector, class Scalar>
bool normalize_rank1_tangent( VectorOperations *vec_ops, Vector &vector_component, Scalar &parameter_component )
{
    if ( vec_ops == nullptr || !common::scalar_math::isfinite( parameter_component ) ||
         !vec_ops->is_valid_number( vector_component ) )
    {
        return false;
    }

    const Scalar norm         = vec_ops->norm_rank1( vector_component, parameter_component );
    const Scalar minimum_norm = Scalar( 16 ) * std::numeric_limits<Scalar>::epsilon();
    if ( !common::scalar_math::isfinite( norm ) || norm <= minimum_norm )
    {
        return false;
    }

    parameter_component /= norm;
    vec_ops->scale( Scalar( 1 ) / norm, vector_component );
    return common::scalar_math::isfinite( parameter_component ) && vec_ops->is_valid_number( vector_component );
}

} // namespace continuation

#endif
