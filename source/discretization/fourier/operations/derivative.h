#ifndef __DISCRETIZATION_FOURIER_OPERATIONS_DERIVATIVE_H__
#define __DISCRETIZATION_FOURIER_OPERATIONS_DERIVATIVE_H__

#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/wavevector_table.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{
namespace operations
{

template<class Backend, class SpectralField, class WavevectorTable>
void derivative(
    const SpectralField& input,
    const WavevectorTable& wavevectors,
    const std::size_t dimension,
    SpectralField& output
)
{
    using complex_type = typename SpectralField::value_type;
    using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using scalar_type = typename traits::real_type;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    if(dimension > 1 || input.size() != output.size() || input.size() != wavevectors.index_space().complex_size())
    {
        throw std::invalid_argument("Fourier derivative argument mismatch");
    }

    const complex_type* source = input.data();
    complex_type* destination = output.data();
    const scalar_type* modes = dimension == 0 ? wavevectors.kx().data() : wavevectors.ky().data();
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
        {
            const scalar_type k = modes[index];
            const scalar_type real = traits::real(source[index]);
            const scalar_type imag = traits::imag(source[index]);
            destination[index] = traits::make(-k*imag, k*real);
        },
        static_cast<std::ptrdiff_t>(input.size())
    );
    for_each.wait();
}

template<class Backend, class SpectralField, class WavevectorTable>
void derivative_adjoint(
    const SpectralField& input,
    const WavevectorTable& wavevectors,
    const std::size_t dimension,
    SpectralField& output)
{
    using complex_type = typename SpectralField::value_type;
    using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using scalar_type = typename traits::real_type;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    if(dimension > 1 || input.size() != output.size() ||
       input.size() != wavevectors.index_space().complex_size())
    {
        throw std::invalid_argument("Fourier derivative adjoint argument mismatch");
    }

    const complex_type* source = input.data();
    complex_type* destination = output.data();
    const scalar_type* modes =
        dimension == 0 ? wavevectors.kx().data() : wavevectors.ky().data();
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
        {
            const scalar_type k = modes[index];
            const scalar_type real = traits::real(source[index]);
            const scalar_type imag = traits::imag(source[index]);
            destination[index] = traits::make(k*imag, -k*real);
        },
        static_cast<std::ptrdiff_t>(input.size()));
    for_each.wait();
}

} // namespace operations
} // namespace fourier
} // namespace discretization

#endif
