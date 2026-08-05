#ifndef __DISCRETIZATION_FOURIER_OPERATIONS_VECTOR_ADVECTION_H__
#define __DISCRETIZATION_FOURIER_OPERATIONS_VECTOR_ADVECTION_H__

#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{
namespace operations
{

template<class Backend, class SpectralField>
void add_spectra(const SpectralField& left, const SpectralField& right, SpectralField& output)
{
    using complex_type = typename SpectralField::value_type;
    using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    if(left.size() != right.size() || left.size() != output.size())
    {
        throw std::invalid_argument("add_spectra size mismatch");
    }
    const complex_type* left_values = left.data();
    const complex_type* right_values = right.data();
    complex_type* output_values = output.data();
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
        {
            output_values[index] = traits::add(left_values[index], right_values[index]);
        },
        static_cast<std::ptrdiff_t>(left.size())
    );
    for_each.wait();
}

} // namespace operations
} // namespace fourier
} // namespace discretization

#endif
