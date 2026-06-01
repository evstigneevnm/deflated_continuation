#ifndef __COMMON_SCFD_BACKEND_EXT_HOST_VIEW_H__
#define __COMMON_SCFD_BACKEND_EXT_HOST_VIEW_H__

#include <cstddef>

#include <scfd/arrays/array.h>
#include <scfd/arrays/tensor_array_nd_view.h>

namespace common
{
namespace scfd_backend_ext
{

template<class Array>
struct host_view_traits
{
    using array_type = Array;
    using view_type = typename array_type::view_type;
    using memory_type = typename array_type::memory_type;
    using host_memory_type = typename memory_type::host_memory_type;
    using value_type = typename array_type::value_type;
    using host_array_type = scfd::arrays::array<value_type, host_memory_type>;
};

template<class HostArray>
void ensure_host_array(HostArray& array, std::size_t size)
{
    if(!array.is_free() && static_cast<std::size_t>(array.size()) >= size)
    {
        return;
    }
    if(!array.is_free())
    {
        array.free();
    }
    array.init(static_cast<typename HostArray::ordinal_type>(size));
}

} // namespace scfd_backend_ext
} // namespace common

#endif
