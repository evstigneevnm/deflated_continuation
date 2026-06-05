#ifndef __NMFD_DEFAULT_MULTIVECTOR_SPACE_BASE_H__
#define __NMFD_DEFAULT_MULTIVECTOR_SPACE_BASE_H__

#include "default_multivector_factory_base.h"
#include "default_multivector_operations_base.h"

namespace nmfd
{
namespace operations
{

template<class DerivedSpace, class Type, class VectorType, class Ordinal = std::ptrdiff_t>
class default_multivector_space_base :
    public default_multivector_factory_base<DerivedSpace,Type,VectorType,Ordinal>,
    public default_multivector_operations_base<DerivedSpace,Type,VectorType,Ordinal>
{
public:
    /// To overshadow ambiguity
    using vector_type = VectorType;
    using multivector_type = typename default_multivector_operations_base<DerivedSpace,Type,VectorType,Ordinal>::multivector_type;
    using scalar_type = Type;
    using Ord = Ordinal;
    using ordinal_type = Ord;

    default_multivector_space_base(bool use_high_precision = false) :
      default_multivector_operations_base<DerivedSpace,Type,VectorType,Ordinal>(use_high_precision)
    {
    }
    virtual ~default_multivector_space_base() = default;
};

} // namespace operations
} // namespace nmfd

#endif
