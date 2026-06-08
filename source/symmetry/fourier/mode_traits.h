#ifndef __SYMMETRY_FOURIER_MODE_TRAITS_H__
#define __SYMMETRY_FOURIER_MODE_TRAITS_H__

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

template <std::size_t Dim>
class mode_index
{
    static_assert( Dim > 0, "mode_index requires Dim > 0" );

public:
    using storage_type = std::array<int, Dim>;

    mode_index() = default;

    explicit mode_index( const storage_type &values ) : values_( values )
    {
    }

    mode_index( std::initializer_list<int> values )
    {
        if ( values.size() != Dim )
            throw std::invalid_argument( "mode_index initializer has incorrect dimension" );
        std::size_t i = 0;
        for ( const auto value : values )
        {
            values_[i] = value;
            ++i;
        }
    }

    __DEVICE_TAG__ int operator[]( const std::size_t i ) const
    {
        return values_[i];
    }

    const storage_type &values() const
    {
        return values_;
    }

private:
    storage_type values_{};
};

template <class Real, std::size_t Dim>
class translation_direction
{
    static_assert( Dim > 0, "translation_direction requires Dim > 0" );

public:
    using real_type = Real;
    using storage_type = std::array<Real, Dim>;

    translation_direction() = default;

    explicit translation_direction( const storage_type &values ) : values_( values )
    {
    }

    translation_direction( std::initializer_list<Real> values )
    {
        if ( values.size() != Dim )
            throw std::invalid_argument( "translation_direction initializer has incorrect dimension" );
        std::size_t i = 0;
        for ( const auto value : values )
        {
            values_[i] = value;
            ++i;
        }
    }

    __DEVICE_TAG__ Real operator[]( const std::size_t i ) const
    {
        return values_[i];
    }

    const storage_type &values() const
    {
        return values_;
    }

private:
    storage_type values_{};
};

template <class Real, std::size_t Dim>
__DEVICE_TAG__ Real wave_dot( const mode_index<Dim> &mode, const translation_direction<Real, Dim> &direction )
{
    Real result = Real{};
    for ( std::size_t i = 0; i < Dim; ++i )
        result += static_cast<Real>( mode[i] ) * direction[i];
    return result;
}

template <class Real>
__DEVICE_TAG__ Real slice_matrix_entry( const Real wave_direction_dot, const Real observable_real_part )
{
    return wave_direction_dot * observable_real_part;
}

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
            result = traits_type::add(
                result,
                traits_type::mul( traits_type::conj( eta_[i] ), coefficients[i] )
            );
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

} // namespace fourier
} // namespace symmetry

#endif
