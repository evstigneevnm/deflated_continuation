#ifndef __DISCRETIZATION_FOURIER_TESTS_LEGACY_KS2D_CODEC_ADAPTER_H__
#define __DISCRETIZATION_FOURIER_TESTS_LEGACY_KS2D_CODEC_ADAPTER_H__

#include <cstddef>
#include <vector>

#include <common/scfd_backend_ext/complex.h>

namespace discretization
{
namespace fourier
{
namespace tests
{

// Reproduces the legacy Nx*My-1 mapping for compatibility checks only.
template<class Complex>
std::vector<Complex> legacy_ks2d_unpack(const std::vector<typename ::common::scfd_backend_ext::complex_value_traits<Complex>::real_type>& state)
{
    using traits = ::common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits::real_type;
    std::vector<Complex> spectrum(state.size() + 1, traits::make(real_type(0), real_type(0)));
    for(std::size_t index = 0; index < state.size(); ++index)
    {
        spectrum[index + 1] = traits::make(real_type(0), state[index]);
    }
    return spectrum;
}

template<class Complex>
std::vector<typename ::common::scfd_backend_ext::complex_value_traits<Complex>::real_type>
legacy_ks2d_pack(const std::vector<Complex>& spectrum)
{
    using traits = ::common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits::real_type;
    std::vector<real_type> state(spectrum.size() - 1, real_type(0));
    for(std::size_t index = 1; index < spectrum.size(); ++index)
    {
        state[index - 1] = traits::imag(spectrum[index]);
    }
    return state;
}

} // namespace tests
} // namespace fourier
} // namespace discretization

#endif
