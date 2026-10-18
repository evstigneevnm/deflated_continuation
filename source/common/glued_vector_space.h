#ifndef __SCFD_GLUED_VECTOR_SPACE_H__
#define __SCFD_GLUED_VECTOR_SPACE_H__

#include <common/glued_vector_operations.h>
#include <nmfd/operations/glued_vector_space.h>

namespace scfd
{
namespace linspace
{

template <class VectorSpace, std::size_t n>
using glued_vector_space = nmfd::operations::glued_vector_space<VectorSpace, n>;

}
}

#endif
