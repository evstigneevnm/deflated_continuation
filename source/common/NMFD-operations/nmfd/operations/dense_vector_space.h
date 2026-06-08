#ifndef __NMFD_DENSE_VECTOR_SPACE_H__
#define __NMFD_DENSE_VECTOR_SPACE_H__

#include <scfd/utils/todo.h>

#include <nmfd/operations/kernels/dense_vector_space.h>
#include <nmfd/operations/dense_vector_operations.h>
#include <nmfd/operations/default_multivector_space_base.h>

namespace nmfd
{
namespace operations
{


template <class VectorTraits, class Backend, class Ordinal = std::ptrdiff_t>
class dense_vector_space : public dense_vector_operations<VectorTraits, Backend, Ordinal>,
                           public default_multivector_factory_base<
                               dense_vector_space<VectorTraits, Backend, Ordinal>, typename VectorTraits::scalar_type,
                               typename VectorTraits::vector_type, Ordinal>
{
public:
    using vector_type      = typename VectorTraits::vector_type;
    using parent_t         = dense_vector_operations<VectorTraits, Backend, Ordinal>;
    using scalar_type      = typename VectorTraits::scalar_type;
    using Ord              = Ordinal;
    using ordinal_type     = Ordinal;
    using multivector_type = typename parent_t::multivector_type;

public:
    using parent_t::add_lin_comb;
    using parent_t::assign;
    using parent_t::scalar_prod;
    using parent_t::scalar_prod_l2;

    dense_vector_space() = default;

    template <typename... Args>
    dense_vector_space( Args &&...args )
        : dense_vector_operations<VectorTraits, Backend, Ordinal>( std::forward<Args>( args )... )
    {
    }

    void init_vector( vector_type &vec ) const
    {
        const VectorTraits &vt = parent_t::vt_;
        vt.alloc( vt.loc_size(), vec );
    }
    template <class... Args>
    void init_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)init_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void free_vector( vector_type &vec ) const
    {
        parent_t::vt_.dealloc( vec );
    }
    template <class... Args>
    void free_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void start_use_vector( vector_type &x ) const
    {
    }
    template <class... Args>
    void start_use_vectors( Args &&...args ) const
    {
    }

    void stop_use_vector( vector_type &x ) const
    {
    }
    template <class... Args>
    void stop_use_vectors( Args &&...args ) const
    {
    }

    [[nodiscard]] size_t size() const
    {
        return parent_t::vt_.size();
    }
};

} // namespace operations

} // namespace nmfd

#endif
