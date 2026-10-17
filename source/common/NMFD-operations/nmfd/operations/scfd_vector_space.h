#ifndef __NMFD_SCFD_VECTOR_SPACE_H__
#define __NMFD_SCFD_VECTOR_SPACE_H__

#include <nmfd/operations/detail/scfd_array_traits.h>
#include <nmfd/operations/dense_vector_space.h>

namespace nmfd
{
namespace operations
{

// The project SCFD space keeps its historical activation and RMS norm conventions.
template <class Backend, class T, class Ordinal = typename Backend::ordinal_type>
class scfd_vector_space
    : public dense_vector_space<detail::scfd_array_traits<T, typename Backend::memory_type>, Backend, Ordinal>
{
public:
    using parent_t = dense_vector_space<detail::scfd_array_traits<T, typename Backend::memory_type>, Backend, Ordinal>;
    using vector_type           = typename parent_t::vector_type;
    using norm_type             = typename parent_t::norm_type;
    using nmfd_parent_type      = parent_t;
    using vector_host_view_type = typename parent_t::host_view_traits::scoped_view_type;

    explicit scfd_vector_space( std::size_t size ) : parent_t( size )
    {
    }

    void init_vector( vector_type &vector ) const override
    {
        vector = vector_type();
    }
    void init_vector( vector_type &vector, std::size_t ) const
    {
        init_vector( vector );
    }
    void start_use_vector( vector_type &vector ) const override
    {
        start_use_vector( vector, 0 );
    }
    void start_use_vector( vector_type &vector, std::size_t size ) const
    {
        if ( vector.is_free() )
            parent_t::vt_.alloc( parent_t::checked_size( size > 0 ? size : this->size() ), vector );
    }
    norm_type norm_l2( const vector_type &vector ) const override
    {
        return this->size() == 0 ? norm_type( 0 ) : this->norm( vector ) / this->get_l2_size();
    }
    norm_type norm2( const vector_type &vector ) const override
    {
        return norm_l2( vector );
    }
    norm_type norm2_sq( const vector_type &vector ) const override
    {
        return this->size() == 0 ? norm_type( 0 ) : this->norm_sq( vector ) / static_cast<norm_type>( this->size() );
    }
    norm_type norm_l2_sq( const vector_type &vector ) const override
    {
        return norm2_sq( vector );
    }
    norm_type norm_rank1_l2( const vector_type &vector, const T value ) const
    {
        return this->size() == 0 ? norm_type( 0 ) : this->norm_rank1( vector, value ) / this->get_l2_size();
    }
};

}
}

#endif
