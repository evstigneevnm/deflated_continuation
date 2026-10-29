#ifndef __SYMMETRY_BACKEND_VECTOR_COMPARISON_H__
#define __SYMMETRY_BACKEND_VECTOR_COMPARISON_H__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <scfd/arrays/array.h>
#include <scfd/functional/basic_ops.h>
#include <scfd/utils/scalar_traits.h>

namespace symmetry
{
namespace detail
{

template <class VectorOperations>
class backend_vector_comparison
{
public:
    using vector_type  = typename VectorOperations::vector_type;
    using scalar_type  = typename VectorOperations::scalar_type;
    using ordinal_type = typename VectorOperations::ordinal_type;
    using memory_type  = typename VectorOperations::memory_type;
    using traits       = scfd::utils::scalar_traits<scalar_type>;

    explicit backend_vector_comparison( VectorOperations &operations ) : operations_( operations )
    {
    }

    scalar_type canonical_tolerance( const vector_type &source ) const
    {
        const auto maximum = operations_.transform_reduce_max(
            [] __DEVICE_TAG__( scalar_type value ) { return traits::abs( value ); }, source
        );
        return scalar_type( 4096 ) * std::numeric_limits<scalar_type>::epsilon() *
               std::max( scalar_type( 1 ), maximum );
    }

    bool lexicographically_greater( const vector_type &left, const vector_type &right, scalar_type tolerance )
    {
        const auto size = operations_.get_size( left );
        if ( size != operations_.get_size( right ) )
            throw std::invalid_argument( "canonical comparison vector sizes differ" );
        if ( size == 0 )
            return false;
        if ( size > static_cast<std::size_t>( std::numeric_limits<ordinal_type>::max() / 2 ) )
            throw std::overflow_error( "canonical comparison index exceeds ordinal range" );
        if ( indices_.is_free() || indices_.size() != size )
        {
            if ( !indices_.is_free() )
                indices_.free();
            indices_.init( size );
        }
        const auto                               lhs     = left;
        const auto                               rhs     = right;
        const auto                               indices = indices_;
        const auto                               end     = static_cast<ordinal_type>( 2 * size );
        typename VectorOperations::for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                const auto a     = traits::abs( lhs( i ) ) <= tolerance ? scalar_type( 0 ) : lhs( i );
                const auto b     = traits::abs( rhs( i ) ) <= tolerance ? scalar_type( 0 ) : rhs( i );
                const auto delta = a - b;
                // The lowest code identifies the first significant difference and its direction.
                indices( i ) = delta > tolerance ? 2 * i : ( delta < -tolerance ? 2 * i + 1 : end );
            },
            static_cast<ordinal_type>( size )
        );
        for_each.wait();
        typename VectorOperations::reduce_type reduce;
        const auto                             first = reduce(
            static_cast<ordinal_type>( size ), indices.raw_ptr(), end, scfd::functional::minimum<ordinal_type>{}
        );
        return first != end && first % 2 == 0;
    }

private:
    VectorOperations                              &operations_;
    scfd::arrays::array<ordinal_type, memory_type> indices_;
};

} // namespace detail
} // namespace symmetry

#endif
