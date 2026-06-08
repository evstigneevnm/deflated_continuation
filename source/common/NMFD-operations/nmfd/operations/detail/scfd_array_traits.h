#ifndef __NMFD_OPERATIONS_DETAIL_SCFD_ARRAY_TRAITS_H__
#define __NMFD_OPERATIONS_DETAIL_SCFD_ARRAY_TRAITS_H__

#include <cstddef>
#include <scfd/arrays/array.h>

namespace nmfd
{
namespace operations
{
namespace detail
{

template <class T, class Memory>
class scfd_array_traits
{
public:
    using scalar_type = T;
    using memory_type = Memory;
    using vector_type = scfd::arrays::array<scalar_type, memory_type>;

public:
    scfd_array_traits() = default;
    explicit scfd_array_traits( size_t size ) : size_( size )
    {
    }

    void alloc( size_t loc_sz, vector_type &v ) const
    {
        v.init( loc_sz );
    }

    void dealloc( vector_type &v ) const
    {
        v.free();
    }

    scalar_type *get_raw_ptr( vector_type &v ) const
    {
        return v.raw_ptr();
    }
    const scalar_type *get_raw_ptr( const vector_type &v ) const
    {
        return v.raw_ptr();
    }

    [[nodiscard]] size_t get_loc_size( const vector_type &v ) const
    {
        return v.size();
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
    size_t size_{ 0 };
};

} // namespace detail
} // namespace operations
} // namespace nmfd

#endif
