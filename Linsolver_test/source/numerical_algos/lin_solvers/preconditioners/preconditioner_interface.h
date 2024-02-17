#ifndef __SCFD_PRECONDITIONER_INTERFACE_H__
#define __SCFD_PRECONDITIONER_INTERFACE_H__

#include <memory>
#include "../detail/algo_utils_hierarchy.h"
#include "../detail/algo_params_hierarchy.h"
#include "../detail/algo_hierarchy_creator.h"

namespace numerical_algos
{
namespace preconditioners 
{

using numerical_algos::detail::algo_params_hierarchy;
using numerical_algos::detail::algo_utils_hierarchy;
using numerical_algos::detail::algo_hierarchy_creator;

template <class VectorSpace, class LinearOperator>
class preconditioner_interface
{
public:
    using vector_space_type = VectorSpace;
    using vector_type = typename VectorSpace::vector_type;
    using operator_type = LinearOperator;
    
    virtual ~preconditioner_interface() 
    {
    }
    virtual void set_operator(std::shared_ptr<const operator_type> op) = 0;
    virtual void apply(const vector_type &rhs, vector_type &x) const = 0;
    /// inplace version for preconditioner interface
    virtual void apply(vector_type &x) const = 0;
};


}  // preconditioners
}  // numerical_algos

#endif