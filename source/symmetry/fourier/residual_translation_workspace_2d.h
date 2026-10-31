#ifndef __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_WORKSPACE_2D_H__
#define __SYMMETRY_FOURIER_RESIDUAL_TRANSLATION_WORKSPACE_2D_H__

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <scfd/arrays/array.h>
#include <scfd/utils/scalar_traits.h>
#include <symmetry/fourier/residual_translation_group_2d.h>

namespace symmetry
{
namespace fourier
{
namespace detail
{

__DEVICE_TAG__ inline int
retained_translation_sign( const residual_translation_character_2d &character, const mode_index<2> &mode )
{
    const std::int64_t dot = character.numerator_x * static_cast<std::int64_t>( mode[0] ) +
                             character.numerator_y * static_cast<std::int64_t>( mode[1] );
    return dot % character.denominator == 0 && ( dot / character.denominator ) % 2 != 0 ? -1 : 1;
}

template <class VectorOperations>
class residual_translation_workspace_2d
{
public:
    using scalar_type    = typename VectorOperations::scalar_type;
    using norm_type      = typename VectorOperations::norm_type;
    using vector_type    = typename VectorOperations::vector_type;
    using memory_type    = typename VectorOperations::memory_type;
    using ordinal_type   = typename VectorOperations::ordinal_type;
    using for_each_type  = typename VectorOperations::for_each_type;
    using traits         = scfd::utils::scalar_traits<scalar_type>;
    using character_type = residual_translation_character_2d;

    residual_translation_workspace_2d( VectorOperations &operations, const std::vector<mode_index<2>> &modes )
        : operations_( operations ), size_( modes.size() ), mask_host_( ( size_ + 7 ) / 8 ), scratch_( operations )
    {
        if ( size_ != 0 )
        {
            modes_.init( size_ );
            mask_.init( mask_host_.size() );
            memory_type::copy_from_host( size_ * sizeof( mode_index<2> ), modes.data(), modes_.raw_ptr() );
        }
        scratch_.init();
        scratch_.start_use();
    }

    norm_type maximum_amplitude( const vector_type &source ) const
    {
        if ( size_ == 0 )
            return norm_type( 0 );
        const auto maximum = operations_.transform_reduce_max(
            [] __DEVICE_TAG__( scalar_type value ) {
                return traits::is_finite( value ) ? traits::abs( value ) : std::numeric_limits<norm_type>::quiet_NaN();
            },
            source
        );
        if ( !std::isfinite( maximum ) )
            throw std::invalid_argument( "Residual translation requires finite vectors" );
        return maximum;
    }

    const std::vector<std::uint8_t> &active_mask( const vector_type &source, norm_type tolerance )
    {
        if ( size_ == 0 )
            return mask_host_;
        const auto input  = source;
        const auto output = mask_;
        const auto size   = size_;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type byte ) {
                std::uint8_t bits = 0;
                for ( unsigned bit = 0; bit < 8; ++bit )
                {
                    const auto index = static_cast<std::size_t>( byte ) * 8 + bit;
                    if ( index < size && traits::abs( input( index ) ) > tolerance )
                        bits |= static_cast<std::uint8_t>( 1u << bit );
                }
                output( byte ) = bits;
            },
            static_cast<ordinal_type>( mask_host_.size() )
        );
        for_each_.wait();
        if ( !mask_host_.empty() )
            memory_type::copy_to_host( mask_host_.size(), mask_.raw_ptr(), mask_host_.data() );
        return mask_host_;
    }

    norm_type incompatible_norm_sq( const character_type &character, const vector_type &source )
    {
        if ( character.is_identity() )
            return norm_type( 0 );
        const auto input  = source;
        const auto output = *scratch_;
        const auto modes  = modes_;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                output( i ) = character.compatible( modes( i ) ) ? scalar_type( 0 ) : input( i );
            },
            static_cast<ordinal_type>( size_ )
        );
        for_each_.wait();
        return operations_.norm_sq( *scratch_ );
    }

    bool preserves_subspace( const character_type &character, const vector_type &source, norm_type tolerance )
    {
        // Retained incompatible coefficients differ from exact translation by at most 2|u_k|.
        return norm_type( 4 ) * incompatible_norm_sq( character, source ) <=
               tolerance * tolerance * static_cast<norm_type>( size_ );
    }

    norm_type discarded_error_bound( const vector_type &source, norm_type threshold )
    {
        if ( size_ == 0 )
            return norm_type( 0 );
        const auto input  = source;
        const auto output = *scratch_;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                output( i ) = traits::abs( input( i ) ) <= threshold ? input( i ) : scalar_type( 0 );
            },
            static_cast<ordinal_type>( size_ )
        );
        for_each_.wait();
        return norm_type( 2 ) * std::sqrt( operations_.norm_sq( *scratch_ ) / static_cast<norm_type>( size_ ) );
    }

    norm_type
    distance_squared( const character_type &character, const vector_type &reference, const vector_type &source )
    {
        if ( size_ == 0 )
            return norm_type( 0 );
        const auto input  = source;
        const auto target = reference;
        const auto output = *scratch_;
        const auto modes  = modes_;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                output( i ) =
                    static_cast<scalar_type>( retained_translation_sign( character, modes( i ) ) ) * input( i ) -
                    target( i );
            },
            static_cast<ordinal_type>( size_ )
        );
        for_each_.wait();
        return operations_.norm_sq( *scratch_ );
    }

    void apply( const character_type &character, const vector_type &source, vector_type &destination )
    {
        if ( character.is_identity() )
        {
            operations_.assign( source, destination );
            return;
        }
        const auto input  = source;
        const auto output = destination;
        const auto modes  = modes_;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                output( i ) =
                    static_cast<scalar_type>( retained_translation_sign( character, modes( i ) ) ) * input( i );
            },
            static_cast<ordinal_type>( size_ )
        );
        for_each_.wait();
    }

private:
    VectorOperations                                         &operations_;
    std::size_t                                               size_;
    scfd::arrays::array<mode_index<2>, memory_type>           modes_;
    scfd::arrays::array<std::uint8_t, memory_type>            mask_;
    std::vector<std::uint8_t>                                 mask_host_;
    nmfd::detail::vector_wrap<VectorOperations, false, false> scratch_;
    for_each_type                                             for_each_;
};

} // namespace detail
} // namespace fourier
} // namespace symmetry

#endif
