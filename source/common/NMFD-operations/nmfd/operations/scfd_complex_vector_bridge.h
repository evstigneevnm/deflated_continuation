#ifndef __NMFD_OPERATIONS_SCFD_COMPLEX_VECTOR_BRIDGE_H__
#define __NMFD_OPERATIONS_SCFD_COMPLEX_VECTOR_BRIDGE_H__

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <scfd/utils/device_tag.h>

#include <common/scfd_backend_ext/complex.h>

namespace nmfd
{
namespace operations
{

template<class ProductVectorSpace, class ComplexVectorSpace>
class scfd_complex_vector_bridge
{
public:
    using product_space_type = ProductVectorSpace;
    using complex_space_type = ComplexVectorSpace;
    using real_space_type =
        typename product_space_type::first_space_type;
    using second_real_space_type =
        typename product_space_type::second_space_type;
    using backend_type = typename real_space_type::backend_type;
    using real_scalar_type = typename real_space_type::scalar_type;
    using complex_scalar_type =
        typename complex_space_type::scalar_type;
    using ordinal_type = typename real_space_type::ordinal_type;
    using product_vector_type =
        typename product_space_type::vector_type;
    using complex_vector_type =
        typename complex_space_type::vector_type;
    using complex_traits =
        common::scfd_backend_ext::complex_value_traits<
            complex_scalar_type>;
    using for_each_type =
        typename backend_type::template for_each_type<ordinal_type>;

    static_assert(
        std::is_same<
            backend_type,
            typename second_real_space_type::backend_type>::value &&
        std::is_same<
            backend_type,
            typename complex_space_type::backend_type>::value,
        "SCFD complex bridge requires one backend");
    static_assert(
        std::is_same<
            real_scalar_type,
            typename complex_traits::real_type>::value,
        "SCFD complex bridge scalar types differ");

    scfd_complex_vector_bridge(
        const product_space_type& product_space,
        const complex_space_type& complex_space)
        : size_(product_space.first_size())
    {
        if(
            product_space.second_size() != size_ ||
            complex_space.get_default_size() != size_)
        {
            throw std::invalid_argument(
                "SCFD complex bridge vector-space size mismatch");
        }
    }

    void pack(
        const product_vector_type& source,
        complex_vector_type& destination) const
    {
        const auto real = source.first.raw_ptr();
        const auto imaginary = source.second.raw_ptr();
        auto complex = destination.raw_ptr();
        for_each_(
            [=] __DEVICE_TAG__ (ordinal_type index)
            {
                complex[index] = complex_traits::make(
                    real[index],
                    imaginary[index]);
            },
            static_cast<ordinal_type>(size_));
        for_each_.wait();
    }

    void unpack(
        const complex_vector_type& source,
        product_vector_type& destination) const
    {
        const auto complex = source.raw_ptr();
        auto real = destination.first.raw_ptr();
        auto imaginary = destination.second.raw_ptr();
        for_each_(
            [=] __DEVICE_TAG__ (ordinal_type index)
            {
                real[index] = complex_traits::real(complex[index]);
                imaginary[index] =
                    complex_traits::imag(complex[index]);
            },
            static_cast<ordinal_type>(size_));
        for_each_.wait();
    }

    std::size_t size() const
    {
        return size_;
    }

private:
    std::size_t size_;
    mutable for_each_type for_each_;
};

} // namespace operations
} // namespace nmfd

#endif
