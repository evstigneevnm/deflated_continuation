#ifndef __SYMMETRY_FOURIER_PHASE_CONDITIONS_H__
#define __SYMMETRY_FOURIER_PHASE_CONDITIONS_H__

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

#include <common/scfd_backend_ext/complex.h>
#include <scfd/utils/device_tag.h>

namespace symmetry
{
namespace fourier
{

template <class Complex, std::size_t MaxComponents>
class vector_valued_phase_functional
{
    static_assert( MaxComponents > 0, "vector_valued_phase_functional requires MaxComponents > 0" );

public:
    using complex_type = Complex;
    using traits_type = common::scfd_backend_ext::complex_value_traits<complex_type>;
    using real_type = typename traits_type::real_type;

    vector_valued_phase_functional() = default;

    vector_valued_phase_functional( std::initializer_list<complex_type> eta )
    {
        resize( eta.size() );
        std::size_t i = 0;
        for ( const auto value : eta )
        {
            eta_[i] = value;
            ++i;
        }
    }

    void resize( const std::size_t components )
    {
        if ( components > MaxComponents )
            throw std::out_of_range( "vector_valued_phase_functional::resize exceeds MaxComponents" );
        components_ = components;
    }

    std::size_t size() const
    {
        return components_;
    }

    void set_eta( const std::size_t component, const complex_type value )
    {
        if ( component >= components_ )
            throw std::out_of_range( "vector_valued_phase_functional::set_eta" );
        eta_[component] = value;
    }

    const complex_type &eta( const std::size_t component ) const
    {
        if ( component >= components_ )
            throw std::out_of_range( "vector_valued_phase_functional::eta" );
        return eta_[component];
    }

    __DEVICE_TAG__ complex_type evaluate( const complex_type *coefficients ) const
    {
        complex_type result = traits_type::make( real_type( 0 ), real_type( 0 ) );
        for ( std::size_t i = 0; i < components_; ++i )
        {
            result = traits_type::add( result, traits_type::mul( traits_type::conj( eta_[i] ), coefficients[i] ) );
        }
        return result;
    }

    __DEVICE_TAG__ real_type real_value( const complex_type *coefficients ) const
    {
        return traits_type::real( evaluate( coefficients ) );
    }

    __DEVICE_TAG__ real_type phase_value( const complex_type *coefficients ) const
    {
        return traits_type::imag( evaluate( coefficients ) );
    }

private:
    std::size_t components_ = 0;
    std::array<complex_type, MaxComponents> eta_{};
};

template <class Complex>
__DEVICE_TAG__ typename common::scfd_backend_ext::complex_value_traits<Complex>::real_type scalar_phase_value(
    const Complex &coefficient
)
{
    return common::scfd_backend_ext::complex_value_traits<Complex>::imag( coefficient );
}

template <class Complex>
__DEVICE_TAG__ typename common::scfd_backend_ext::complex_value_traits<Complex>::real_type scalar_phase_real_part(
    const Complex &coefficient
)
{
    return common::scfd_backend_ext::complex_value_traits<Complex>::real( coefficient );
}

} // namespace fourier
} // namespace symmetry

#endif
