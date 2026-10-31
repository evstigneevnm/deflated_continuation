#ifndef __SYMMETRY_FOURIER_TRANSLATION_ISOTROPY_GEOMETRY_H__
#define __SYMMETRY_FOURIER_TRANSLATION_ISOTROPY_GEOMETRY_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <scfd/arrays/array.h>
#include <scfd/utils/scalar_traits.h>
#include <symmetry/fourier/mode_descriptor.h>
#include <symmetry/stabilizer_description.h>

namespace symmetry
{
namespace fourier
{

enum class isotropy_support_scale
{
    maximum_amplitude,
    state_norm
};

template <std::size_t Dimension>
struct isotropy_coefficient
{
    mode_index<Dimension> mode;
    std::size_t           first  = 0;
    std::size_t           second = std::numeric_limits<std::size_t>::max();
};

namespace detail
{

template <std::size_t Dimension>
struct isotropy_backend_coefficient
{
    int         mode[Dimension]{};
    std::size_t first = 0, second = 0;
};

template <class T>
struct isotropy_amplitude
{
    T    maximum = T( 0 );
    bool finite  = true;
};

template <class T>
struct isotropy_amplitude_maximum
{
    __DEVICE_TAG__ isotropy_amplitude<T>
                   operator()( const isotropy_amplitude<T> &left, const isotropy_amplitude<T> &right ) const
    {
        return { left.maximum > right.maximum ? left.maximum : right.maximum, left.finite && right.finite };
    }
};

}

// The modes are characters of the problem-declared admissible translation torus.
// Real/imaginary pairs and repeated modes across fields share the same contract.
// This workspace borrows operations, which must outlive it (as with vector_wrap).
template <class VectorOperations, std::size_t Dimension>
class translation_isotropy_geometry
{
public:
    using scalar_type              = typename VectorOperations::scalar_type;
    using norm_type                = typename VectorOperations::norm_type;
    using vector_type              = typename VectorOperations::vector_type;
    using backend_type             = typename VectorOperations::backend_type;
    using memory_type              = typename VectorOperations::memory_type;
    using ordinal_type             = typename VectorOperations::ordinal_type;
    using for_each_type            = typename VectorOperations::for_each_type;
    using reduce_type              = typename backend_type::reduce_type;
    using scalar_traits_type       = scfd::utils::scalar_traits<scalar_type>;
    using norm_traits_type         = scfd::utils::scalar_traits<norm_type>;
    using coefficient_type         = isotropy_coefficient<Dimension>;
    using backend_coefficient_type = detail::isotropy_backend_coefficient<Dimension>;
    using lattice_type             = translation::integer_lattice<Dimension>;
    using amplitude_type           = detail::isotropy_amplitude<norm_type>;

    struct classification
    {
        lattice_type lattice;
        norm_type    maximum = norm_type( 0 );
    };

    translation_isotropy_geometry(
        VectorOperations &operations, const std::vector<coefficient_type> &coefficients, std::string fingerprint,
        isotropy_support_scale support_scale = isotropy_support_scale::maximum_amplitude
    )
        : operations_( operations ), size_( operations.get_default_size() ), count_( coefficients.size() ),
          batches_( std::min<std::size_t>( count_, 256 ) ), fingerprint_( std::move( fingerprint ) ),
          support_scale_( support_scale ), scratch_( operations )
    {
        if ( fingerprint_.empty() )
            throw std::invalid_argument( "Translation isotropy requires an admissible-group fingerprint" );
        std::vector<bool> covered( size_, false );
        for ( const auto &coefficient : coefficients )
        {
            if ( coefficient.first == missing_component )
                throw std::invalid_argument( "Isotropy coefficient requires its first component" );
            for ( unsigned component = 0; component < 2; ++component )
            {
                const auto index = component == 0 ? coefficient.first : coefficient.second;
                if ( index == missing_component )
                    continue;
                if ( index >= size_ || covered[index] )
                    throw std::invalid_argument( "Isotropy coefficient layout has an invalid or repeated index" );
                covered[index] = true;
            }
        }
        if ( std::find( covered.begin(), covered.end(), false ) != covered.end() )
            throw std::invalid_argument( "Isotropy coefficient layout must cover the state" );
        if ( count_ != 0 )
        {
            // Upload plain device metadata, without host-only standard-container accessors.
            std::vector<backend_coefficient_type> metadata( count_ );
            for ( std::size_t i = 0; i < count_; ++i )
            {
                for ( std::size_t d = 0; d < Dimension; ++d )
                    metadata[i].mode[d] = coefficients[i].mode.values()[d];
                metadata[i].first  = coefficients[i].first;
                metadata[i].second = coefficients[i].second;
            }
            coefficients_.init( count_ );
            memory_type::copy_from_host(
                count_ * sizeof( backend_coefficient_type ), metadata.data(), coefficients_.raw_ptr()
            );
            amplitudes_.init( batches_ );
            lattices_.init( batches_ );
        }
    }

    translation_isotropy_geometry(
        VectorOperations &operations, const std::vector<mode_index<Dimension>> &modes, std::string fingerprint
    )
        : translation_isotropy_geometry( operations, scalar_coefficients( modes ), std::move( fingerprint ) )
    {
    }

    classification classify( const vector_type &state, norm_type relative_tolerance )
    {
        require_size( state );
        if ( !( relative_tolerance > norm_type( 0 ) ) || !std::isfinite( relative_tolerance ) )
            throw std::invalid_argument( "Isotropy mode tolerance must be finite and positive" );
        classification result;
        if ( count_ == 0 )
            return result;
        const auto input        = state;
        const auto coefficients = coefficients_;
        const auto amplitudes   = amplitudes_;
        const auto lattices     = lattices_;
        const auto count        = count_;
        const auto batches      = batches_;
        const bool device       = backend_type::is_device_backend();
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type batch ) {
                amplitude_type    summary;
                const std::size_t begin  = device ? batch : count * batch / batches;
                const std::size_t end    = device ? count : count * ( batch + 1 ) / batches;
                const std::size_t stride = device ? batches : 1;
                for ( auto i = begin; i < end; i += stride )
                {
                    const auto coefficient = coefficients( i );
                    auto       a           = scalar_traits_type::abs( input( coefficient.first ) );
                    auto       b           = coefficient.second == missing_component
                                                 ? norm_type( 0 )
                                                 : scalar_traits_type::abs( input( coefficient.second ) );
                    if ( !scalar_traits_type::is_finite( input( coefficient.first ) ) ||
                         ( coefficient.second != missing_component &&
                           !scalar_traits_type::is_finite( input( coefficient.second ) ) ) )
                    {
                        summary.finite = false;
                        continue;
                    }
                    if ( a < b )
                    {
                        const auto temporary = a;
                        a                    = b;
                        b                    = temporary;
                    }
                    const auto ratio     = a == norm_type( 0 ) ? norm_type( 0 ) : b / a;
                    const auto amplitude = a * norm_traits_type::sqrt( norm_type( 1 ) + ratio * ratio );
                    if ( amplitude > summary.maximum )
                        summary.maximum = amplitude;
                }
                amplitudes( batch ) = summary;
            },
            static_cast<ordinal_type>( batches_ )
        );
        for_each_.wait();
        const auto maximum = reduce_(
            static_cast<ordinal_type>( batches_ ), amplitudes_.raw_ptr(), amplitude_type{},
            detail::isotropy_amplitude_maximum<norm_type>{}
        );
        if ( !maximum.finite || !std::isfinite( maximum.maximum ) )
            throw std::invalid_argument( "Translation isotropy requires finite coefficients" );
        result.maximum       = maximum.maximum;
        const auto threshold = relative_tolerance * ( support_scale_ == isotropy_support_scale::state_norm
                                                          ? std::sqrt( operations_.norm_sq( state ) )
                                                          : maximum.maximum );
        if ( !std::isfinite( threshold ) )
            throw std::overflow_error( "Translation isotropy support threshold is not finite" );
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type batch ) {
                lattice_type      lattice;
                const std::size_t begin  = device ? batch : count * batch / batches;
                const std::size_t end    = device ? count : count * ( batch + 1 ) / batches;
                const std::size_t stride = device ? batches : 1;
                for ( auto i = begin; i < end; i += stride )
                {
                    const auto coefficient = coefficients( i );
                    const auto a           = scalar_traits_type::abs( input( coefficient.first ) );
                    const auto b           = coefficient.second == missing_component
                                                 ? norm_type( 0 )
                                                 : scalar_traits_type::abs( input( coefficient.second ) );
                    const auto scale       = a > b ? a : b;
                    const auto ratio       = scale == norm_type( 0 ) ? norm_type( 0 ) : ( a < b ? a : b ) / scale;
                    if ( scale * norm_traits_type::sqrt( norm_type( 1 ) + ratio * ratio ) > threshold )
                    {
                        std::int64_t mode[Dimension];
                        for ( std::size_t d = 0; d < Dimension; ++d )
                            mode[d] = coefficient.mode[d];
                        lattice.insert( mode );
                    }
                }
                lattices( batch ) = lattice;
            },
            static_cast<ordinal_type>( batches_ )
        );
        for_each_.wait();
        result.lattice = reduce_(
            static_cast<ordinal_type>( batches_ ), lattices_.raw_ptr(), lattice_type{},
            translation::lattice_sum<Dimension>{}
        );
        if ( result.lattice.overflow )
            throw std::overflow_error( "Translation isotropy lattice reduction overflow" );
        return result;
    }

    bool is_zero( const classification &value ) const
    {
        return value.maximum == norm_type( 0 );
    }

    translation::orbit_type orbit_type( const classification &value ) const
    {
        return describe_orbit( value.lattice );
    }

    stabilizer_description describe( const classification &value ) const
    {
        return describe_lattice( value.lattice, fingerprint_ );
    }

    bool proper_stabilizer_increase( const classification &before, const classification &after ) const
    {
        return before.lattice.contains( after.lattice ) && !after.lattice.contains( before.lattice );
    }

    norm_type transverse_ratio( const vector_type &state, const classification &target )
    {
        require_size( state );
        const auto total = operations_.norm_sq( state );
        if ( !std::isfinite( total ) || total < norm_type( 0 ) )
            throw std::overflow_error( "Translation isotropy state norm is not finite" );
        if ( total == norm_type( 0 ) )
            return norm_type( 0 );
        scratch_.init();
        scratch_.start_use();
        const auto input        = state;
        const auto output       = *scratch_;
        const auto coefficients = coefficients_;
        const auto lattice      = target.lattice;
        for_each_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                const auto   coefficient = coefficients( i );
                std::int64_t mode[Dimension];
                for ( std::size_t d = 0; d < Dimension; ++d )
                    mode[d] = coefficient.mode[d];
                const bool retained         = lattice.contains( mode );
                output( coefficient.first ) = retained ? scalar_type( 0 ) : input( coefficient.first );
                if ( coefficient.second != missing_component )
                    output( coefficient.second ) = retained ? scalar_type( 0 ) : input( coefficient.second );
            },
            static_cast<ordinal_type>( count_ )
        );
        for_each_.wait();
        return std::sqrt( operations_.norm_sq( *scratch_ ) / total );
    }

private:
    static constexpr std::size_t missing_component = std::numeric_limits<std::size_t>::max();

    static std::vector<coefficient_type> scalar_coefficients( const std::vector<mode_index<Dimension>> &modes )
    {
        std::vector<coefficient_type> result;
        result.reserve( modes.size() );
        for ( std::size_t i = 0; i < modes.size(); ++i )
            result.push_back( { modes[i], i, missing_component } );
        return result;
    }

    void require_size( const vector_type &state ) const
    {
        if ( operations_.get_size( state ) != size_ )
            throw std::invalid_argument( "Isotropy state does not match its coefficient layout" );
    }

    VectorOperations                                          &operations_;
    std::size_t                                                size_, count_, batches_;
    std::string                                                fingerprint_;
    isotropy_support_scale                                     support_scale_;
    scfd::arrays::array<backend_coefficient_type, memory_type> coefficients_;
    scfd::arrays::array<amplitude_type, memory_type>           amplitudes_;
    scfd::arrays::array<lattice_type, memory_type>             lattices_;
    nmfd::detail::vector_wrap<VectorOperations, false, false>  scratch_;
    for_each_type                                              for_each_;
    reduce_type                                                reduce_;
};

}
}

#endif
