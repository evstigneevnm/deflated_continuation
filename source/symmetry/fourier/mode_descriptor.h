#ifndef __SYMMETRY_FOURIER_MODE_DESCRIPTOR_H__
#define __SYMMETRY_FOURIER_MODE_DESCRIPTOR_H__

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

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

} // namespace fourier
} // namespace symmetry

#endif
