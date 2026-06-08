#ifndef __NMFD_ZERO_FUNCTIONAL_H__
#define __NMFD_ZERO_FUNCTIONAL_H__

namespace nmfd
{
namespace operations
{

template<class VectorSpace> 
class zero_functional
{
public:    
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
public:
    scalar_type calc(const vector_type& x)const
    { 
        return scalar_type(0);
    }

};

}
}

#endif