#ifndef __NMFD_DENSE_VECTOR_SPACE_H__
#define __NMFD_DENSE_VECTOR_SPACE_H__

#include <initializer_list>
#include <utility>
#include <nmfd/operations/dense_vector_operations.h>
#include <nmfd/operations/default_multivector_factory_base.h>

namespace nmfd
{
namespace operations
{

template <class VectorTraits, class Backend, class Ordinal = typename Backend::ordinal_type>
class dense_vector_space : public dense_vector_operations<VectorTraits, Backend, Ordinal>,
                           public default_multivector_factory_base<
                               dense_vector_space<VectorTraits, Backend, Ordinal>, typename VectorTraits::scalar_type,
                               typename VectorTraits::vector_type, Ordinal>
{
public:
    using parent_t          = dense_vector_operations<VectorTraits, Backend, Ordinal>;
    using vector_type       = typename parent_t::vector_type;
    using scalar_type       = typename parent_t::scalar_type;
    using norm_type         = typename parent_t::norm_type;
    using multivector_type  = typename parent_t::multivector_type;
    using ordinal_type      = Ordinal;
    using Ord               = Ordinal;
    using factory_base_type = vector_factory_base<scalar_type, vector_type, multivector_type, Ordinal>;

    dense_vector_space() = default;
    template <class... Args>
    explicit dense_vector_space( Args &&...args ) : parent_t( std::forward<Args>( args )... )
    {
    }

    void init_vector( vector_type &vector ) const override
    {
        parent_t::vt_.alloc( size(), vector );
    }
    template <class... Args>
    void init_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)init_vector( std::forward<Args>( args ) ), 0 )... };
    }
    void free_vector( vector_type &vector ) const override
    {
        parent_t::vt_.dealloc( vector );
    }
    template <class... Args>
    void free_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_vector( std::forward<Args>( args ) ), 0 )... };
    }
    void start_use_vector( vector_type & ) const override
    {
    }
    template <class... Args>
    void start_use_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)start_use_vector( std::forward<Args>( args ) ), 0 )... };
    }
    void stop_use_vector( vector_type & ) const override
    {
    }
    template <class... Args>
    void stop_use_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)stop_use_vector( std::forward<Args>( args ) ), 0 )... };
    }
    std::size_t size() const
    {
        return parent_t::checked_size( parent_t::vt_.size() );
    }
    std::size_t get_default_size() const
    {
        return size();
    }
    std::size_t get_vector_size() const
    {
        return size();
    }
    norm_type get_l2_size() const
    {
        return common::scfd_backend_ext::math<Backend, norm_type>::sqrt( static_cast<norm_type>( size() ) );
    }
};

}
}

#endif
