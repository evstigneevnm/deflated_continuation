#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_SHIFT_BLOCK_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_SHIFT_BLOCK_OPERATOR_H__

#include <complex>

#include "complex_affine_block_operator.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class ProductVectorSpace, class RealOperator>
class complex_shift_block_operator :
    public complex_affine_block_operator<ProductVectorSpace, RealOperator>
{
public:
    using base_type =
        complex_affine_block_operator<ProductVectorSpace, RealOperator>;
    using vector_space_type = typename base_type::vector_space_type;
    using scalar_type = typename base_type::scalar_type;
    using complex_type = typename base_type::complex_type;

    complex_shift_block_operator(
        const vector_space_type& vector_space,
        const RealOperator& real_operator,
        scalar_type shift_real,
        scalar_type shift_imaginary)
        : base_type(
              vector_space,
              real_operator,
              complex_type(scalar_type(1), scalar_type{}),
              complex_type(-shift_real, -shift_imaginary))
    {
    }

    scalar_type shift_real() const
    {
        return -this->diagonal_shift().real();
    }

    scalar_type shift_imaginary() const
    {
        return -this->diagonal_shift().imag();
    }
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
