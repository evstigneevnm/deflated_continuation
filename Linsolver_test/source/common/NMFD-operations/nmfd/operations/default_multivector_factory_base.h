#ifndef __NMFD_DEFAULT_MULTIVECTOR_FACTORY_BASE_H__
#define __NMFD_DEFAULT_MULTIVECTOR_FACTORY_BASE_H__

#include <vector>
#include "vector_factory_base.h"

namespace nmfd
{
namespace operations
{

template
<
    class DerivedSpace, class Type, class VectorType, class Ordinal = std::ptrdiff_t,
    /************* Internal usage only ********************************************/
    class MultiVectorType = std::vector<VectorType>,
    class ParentType = vector_factory_base<Type,VectorType,MultiVectorType,Ordinal>
>
class default_multivector_factory_base : public ParentType
{

public:
    using vector_type = typename ParentType::vector_type;
    using multivector_type = typename ParentType::multivector_type;
    using scalar_type = typename ParentType::scalar_type;
    using Ord = typename ParentType::Ord;    
    using ordinal_type = typename ParentType::ordinal_type;

public:
    default_multivector_factory_base() = default;

    /// multivector interface
    void init_multivector(multivector_type& x, Ord m) const
    {
        x.resize(m);
        for (Ord i = 0;i < m;++i)
        {
            static_cast<const DerivedSpace*>(this)->init_vector(x[i]);
        }
    }
    void free_multivector(multivector_type& x, Ord m) const
    {
        for (Ord i = 0;i < m;++i)
        {
            static_cast<const DerivedSpace*>(this)->free_vector(x[i]);
        }
        x.clear();
    }
    void start_use_multivector(multivector_type& x, Ord m) const
    {
        for (Ord i = 0;i < m;++i)
        {
            static_cast<const DerivedSpace*>(this)->start_use_vector(x[i]);
        }
    }
    void stop_use_multivector(multivector_type& x, Ord m) const
    {
        for (Ord i = 0;i < m;++i)
        {
            static_cast<const DerivedSpace*>(this)->stop_use_vector(x[i]);
        }
    }

};

}
}

#endif
