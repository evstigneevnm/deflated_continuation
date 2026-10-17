#ifndef __NMFD_DEFAULT_MULTIVECTOR_OPERATIONS_BASE_H__
#define __NMFD_DEFAULT_MULTIVECTOR_OPERATIONS_BASE_H__

#include <vector>
#include <stdexcept>
#include "vector_space_base.h"

namespace nmfd
{
namespace operations
{

template <
    class DerivedSpace, class Type, class VectorType, class Ordinal = std::ptrdiff_t,
    /************* Internal usage only ********************************************/
    class MultiVectorType = std::vector<VectorType>,
    class ParentType      = vector_operations_base<Type, VectorType, MultiVectorType, Ordinal>>
class default_multivector_operations_base : public ParentType
{

public:
    using vector_type      = typename ParentType::vector_type;
    using multivector_type = typename ParentType::multivector_type;
    using scalar_type      = typename ParentType::scalar_type;
    using Ord              = typename ParentType::Ord;
    using ordinal_type     = typename ParentType::ordinal_type;

public:
    using ParentType::add_lin_comb;
    using ParentType::assign;
    using ParentType::scalar_prod;
    using ParentType::scalar_prod_l2;

    default_multivector_operations_base( bool use_high_precision = false ) : ParentType( use_high_precision )
    {
    }

    /// multivector interface
    void assign( const multivector_type &mx, Ord m, Ord k_, vector_type &x ) const
    {
        static_cast<const DerivedSpace *>( this )->assign( mx[checked_index( mx, m, k_ )], x );
    }
    void assign( const vector_type &x, multivector_type &mx, Ord m, Ord k_ ) const
    {
        static_cast<const DerivedSpace *>( this )->assign( x, mx[checked_index( mx, m, k_ )] );
    }
    [[nodiscard]] scalar_type scalar_prod( const multivector_type &mx, Ord m, Ord k_, const vector_type &y ) const
    {
        return static_cast<const DerivedSpace *>( this )->scalar_prod( mx[checked_index( mx, m, k_ )], y );
    }
    [[nodiscard]] scalar_type scalar_prod_l2( const multivector_type &mx, Ord m, Ord k_, const vector_type &y ) const
    {
        return static_cast<const DerivedSpace *>( this )->scalar_prod_l2( mx[checked_index( mx, m, k_ )], y );
    }
    void add_lin_comb(
        const scalar_type mul_x, const multivector_type &mx, Ord m, Ord k_, const scalar_type mul_y, vector_type &y
    ) const
    {
        static_cast<const DerivedSpace *>( this )->add_lin_comb( mul_x, mx[checked_index( mx, m, k_ )], mul_y, y );
    }

private:
    static std::size_t checked_index( const multivector_type &vectors, Ord count, Ord index )
    {
        if ( count < Ord( 0 ) || index < Ord( 0 ) || index >= count ||
             static_cast<std::size_t>( count ) > vectors.size() )
            throw std::out_of_range( "Multivector index or count is out of range" );
        return static_cast<std::size_t>( index );
    }
};

}
}

#endif
