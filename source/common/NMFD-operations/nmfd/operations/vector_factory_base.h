#ifndef __NMFD_VECTOR_FACTORY_BASE_H__
#define __NMFD_VECTOR_FACTORY_BASE_H__

#include <cstddef>
#include <utility>


namespace nmfd
{
namespace operations
{

template<class Type, class VectorType, class MultiVectorType, class Ordinal = std::ptrdiff_t>
class vector_factory_base 
{

public:
    using vector_type = VectorType;
    using multivector_type = MultiVectorType;
    using scalar_type = Type;
    using Ord = Ordinal; 
    using ordinal_type = Ord;   

private:
    using T = scalar_type;

protected:

public:
    vector_factory_base() = default;
    virtual ~vector_factory_base() = default;

    virtual void init_vector(vector_type& x) const = 0;
    virtual void free_vector(vector_type& x) const = 0;
    virtual void start_use_vector(vector_type& x) const = 0;
    virtual void stop_use_vector(vector_type& x) const = 0; 

    virtual void init_multivector(multivector_type& x, Ord m) const = 0;
    virtual void free_multivector(multivector_type& x, Ord m) const = 0;
    virtual void start_use_multivector(multivector_type& x, Ord m) const = 0;
    virtual void stop_use_multivector(multivector_type& x, Ord m) const = 0; 
};

}
}

#endif
