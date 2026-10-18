#ifndef __SCFD_GLUED_VECTOR_OPERATIONS_H__
#define __SCFD_GLUED_VECTOR_OPERATIONS_H__

// Frozen timestepper compatibility only. New callers use nmfd::operations directly.

#include <nmfd/operations/glued_vector_operations.h>

namespace scfd
{
namespace linspace
{

template <class Vector, std::size_t n>
using glued_vector = nmfd::operations::glued_vector<Vector, n>;

template <class VectorOperations, std::size_t n>
using glued_vector_operations = nmfd::operations::glued_vector_operations<VectorOperations, n>;

}
}

#endif
