#ifndef __SCFD_VECTOR_OPERATIONS_LOCAL_H__
#define __SCFD_VECTOR_OPERATIONS_LOCAL_H__

#include <nmfd/operations/scfd_vector_space.h>

namespace nmfd
{
namespace operations
{

template <class Backend, class T, class Ordinal = typename Backend::ordinal_type>
using scfd_vector_operations =
    dense_vector_operations<detail::scfd_array_traits<T, typename Backend::memory_type>, Backend, Ordinal>;

}
}

// Compatibility name: historical callers use this as a complete vector space.
template <class Backend, class T, class Ordinal = typename Backend::ordinal_type>
using scfd_vector_operations = nmfd::operations::scfd_vector_space<Backend, T, Ordinal>;

#endif
