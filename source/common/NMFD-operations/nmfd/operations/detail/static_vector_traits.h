#ifndef __NMFD_OPERATIONS_DETAIL_STATIC_VECTOR_TRAITS_H__
#define __NMFD_OPERATIONS_DETAIL_STATIC_VECTOR_TRAITS_H__

#include <array>
#include <cstddef>

namespace nmfd
{
namespace operations
{
namespace detail
{

template <class T, size_t Dim>
class static_vector_traits
{
public:
    using scalar_type = T;
    using vector_type = std::array<T, Dim>;

public:
    void alloc( size_t loc_sz, vector_type &v ) const
    {
    }

    void dealloc( vector_type &v ) const
    {
    }

    scalar_type *get_raw_ptr( vector_type &v ) const
    {
        return v.data();
    }
    const scalar_type *get_raw_ptr( const vector_type &v ) const
    {
        return v.data();
    }
    [[nodiscard]] size_t get_loc_size( const vector_type &v ) const
    {
        return size_;
    }
    [[nodiscard]] size_t size() const
    {
        return size_;
    }
    [[nodiscard]] size_t loc_size() const
    {
        return size_;
    }

private:
    size_t size_{ Dim };
};

} // namespace detail
} // namespace operations
} // namespace nmfd

#endif
