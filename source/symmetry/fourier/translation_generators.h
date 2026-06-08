#ifndef __SYMMETRY_FOURIER_TRANSLATION_GENERATORS_H__
#define __SYMMETRY_FOURIER_TRANSLATION_GENERATORS_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <scfd/utils/device_tag.h>

namespace symmetry
{
namespace fourier
{

template <class Complex>
__DEVICE_TAG__ Complex scale_complex(
    const Complex &value, const typename common::scfd_backend_ext::complex_value_traits<Complex>::real_type scale
)
{
    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    return traits_type::make( scale * traits_type::real( value ), scale * traits_type::imag( value ) );
}

template <class Complex>
__DEVICE_TAG__ Complex rotate_1d_mode(
    const std::size_t mode,
    const typename common::scfd_backend_ext::complex_value_traits<Complex>::real_type shift,
    const Complex &value
)
{
    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    const real_type phase = static_cast<real_type>( mode ) * shift;
    const real_type c = std::cos( phase );
    const real_type s = std::sin( phase );
    const real_type real_part = traits_type::real( value );
    const real_type imag_part = traits_type::imag( value );
    return traits_type::make( c * real_part - s * imag_part, s * real_part + c * imag_part );
}

template <class Complex>
__DEVICE_TAG__ Complex translation_generator_1d_mode( const std::size_t mode, const Complex &coefficient )
{
    using traits_type = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits_type::real_type;

    const real_type k = static_cast<real_type>( mode );
    const real_type real_part = traits_type::real( coefficient );
    const real_type imag_part = traits_type::imag( coefficient );
    return traits_type::make( -k * imag_part, k * real_part );
}

template <class Complex, class Real>
void apply_shift_1d( const Complex *source, Complex *destination, const std::size_t size, const Real shift )
{
    if ( source == nullptr || destination == nullptr )
        throw std::invalid_argument( "apply_shift_1d got null pointer" );
    for ( std::size_t mode = 0; mode < size; ++mode )
        destination[mode] = rotate_1d_mode( mode, shift, source[mode] );
}

template <class Complex>
void translation_generator_1d( const Complex *state_on_slice, Complex *generator, const std::size_t size )
{
    if ( state_on_slice == nullptr || generator == nullptr )
        throw std::invalid_argument( "translation_generator_1d got null pointer" );
    for ( std::size_t mode = 0; mode < size; ++mode )
        generator[mode] = translation_generator_1d_mode( mode, state_on_slice[mode] );
}

} // namespace fourier
} // namespace symmetry

#endif
