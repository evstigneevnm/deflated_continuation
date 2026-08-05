#ifndef __CONTINUATION_TANGENT_NORMALIZATION_H__
#define __CONTINUATION_TANGENT_NORMALIZATION_H__

#include <limits>

#include <common/scalar_math.h>

namespace continuation
{

template<class VectorOperations, class Vector, class Scalar>
bool normalize_rank1_tangent(
    VectorOperations* vec_ops,
    Vector& vector_component,
    Scalar& parameter_component)
{
    if(vec_ops == nullptr ||
       !common::scalar_math::isfinite(parameter_component) ||
       !vec_ops->check_is_valid_number(vector_component))
    {
        return false;
    }

    const Scalar norm = vec_ops->norm_rank1(
        vector_component,
        parameter_component);
    const Scalar minimum_norm =
        Scalar(16)*std::numeric_limits<Scalar>::epsilon();
    if(!common::scalar_math::isfinite(norm) || norm <= minimum_norm)
    {
        return false;
    }

    parameter_component /= norm;
    vec_ops->scale(Scalar(1)/norm, vector_component);
    return common::scalar_math::isfinite(parameter_component) &&
        vec_ops->check_is_valid_number(vector_component);
}

} // namespace continuation

#endif
