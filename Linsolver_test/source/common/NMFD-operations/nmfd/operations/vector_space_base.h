#ifndef __NMFD_VECTOR_SPACE_BASE_H__
#define __NMFD_VECTOR_SPACE_BASE_H__

#include "vector_factory_base.h"
#include "vector_operations_base.h"

namespace nmfd
{
namespace operations
{

template<class Type, class VectorType, class MultiVectorType, class Ordinal = std::ptrdiff_t, class NormType = Type>
class vector_space_base :
    public vector_factory_base<Type,VectorType,MultiVectorType,Ordinal>,
    public vector_operations_base<Type,VectorType,MultiVectorType,Ordinal,NormType>
{
public:
    /// To overshadow ambiguity
    using vector_type = VectorType;
    using multivector_type = MultiVectorType;
    using scalar_type = Type;
    using norm_type = NormType;
    using Ord = Ordinal;
    using ordinal_type = Ord;

    vector_space_base(bool use_high_precision = false) :
      vector_operations_base<Type,VectorType,MultiVectorType,Ordinal,NormType>(use_high_precision)
    {
    }
    virtual ~vector_space_base() = default;
};

} // namespace operations
} // namespace nmfd

#endif
