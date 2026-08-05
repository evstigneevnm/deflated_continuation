#ifndef __SYMMETRY_FOURIER_TRANSLATION_ACTION_H__
#define __SYMMETRY_FOURIER_TRANSLATION_ACTION_H__

#include <array>
#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_backend_ext/math.h>
#include <scfd/utils/device_tag.h>

namespace symmetry
{
namespace fourier
{

namespace detail
{

template<class Real, std::size_t Dimension>
struct translation_kernel_data
{
    const Real* wavevectors[Dimension]{};
    Real shift[Dimension]{};

    __DEVICE_TAG__ Real phase(const std::ptrdiff_t index) const
    {
        Real result = Real(0);
        for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
        {
            result += wavevectors[dimension][index]*shift[dimension];
        }
        return result;
    }
};

} // namespace detail

template<class Backend, std::size_t Dimension, class SpectralField, class WavevectorTable, class Real>
void apply_translation(
    const SpectralField& source,
    const WavevectorTable& wavevectors,
    const std::array<Real, Dimension>& shift,
    SpectralField& destination
)
{
    static_assert(WavevectorTable::dimension == Dimension, "translation and wavevector dimensions differ");
    using complex_type = typename SpectralField::value_type;
    using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using math_type = ::common::scfd_backend_ext::math<Backend, Real>;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    if(source.size() != destination.size() || source.size() != wavevectors.index_space().complex_size())
    {
        throw std::invalid_argument("Fourier translation argument mismatch");
    }

    detail::translation_kernel_data<Real, Dimension> kernel_data;
    for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
    {
        kernel_data.wavevectors[dimension] = wavevectors.component(dimension).data();
        kernel_data.shift[dimension] = shift[dimension];
    }
    const complex_type* input = source.data();
    complex_type* output = destination.data();
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
        {
            const Real phase = kernel_data.phase(index);
            const Real cosine = math_type::cos(phase);
            const Real sine = math_type::sin(phase);
            const Real real = traits::real(input[index]);
            const Real imaginary = traits::imag(input[index]);
            output[index] = traits::make(
                cosine*real - sine*imaginary,
                sine*real + cosine*imaginary
            );
        },
        static_cast<std::ptrdiff_t>(source.size())
    );
    for_each.wait();
}

template<class Backend, class SpectralField, class WavevectorTable>
void translation_generator(
    const SpectralField& source,
    const WavevectorTable& wavevectors,
    const std::size_t dimension,
    SpectralField& destination
)
{
    using complex_type = typename SpectralField::value_type;
    using traits = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using real_type = typename traits::real_type;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    if(dimension >= WavevectorTable::dimension || source.size() != destination.size() ||
       source.size() != wavevectors.index_space().complex_size())
    {
        throw std::invalid_argument("Fourier translation generator argument mismatch");
    }

    const complex_type* input = source.data();
    complex_type* output = destination.data();
    const real_type* modes = wavevectors.component(dimension).data();
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
        {
            const real_type wave_number = modes[index];
            output[index] = traits::make(
                -wave_number*traits::imag(input[index]),
                wave_number*traits::real(input[index])
            );
        },
        static_cast<std::ptrdiff_t>(source.size())
    );
    for_each.wait();
}

} // namespace fourier
} // namespace symmetry

#endif
