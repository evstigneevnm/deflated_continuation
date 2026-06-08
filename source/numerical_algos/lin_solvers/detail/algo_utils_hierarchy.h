#ifndef __SCFD_ALGO_UTILS_HIERARCHY_H__
#define __SCFD_ALGO_UTILS_HIERARCHY_H__

#include "utils_hierarchy_dummy.h"

namespace numerical_algos
{
namespace detail 
{

template<class Algo, class = int>
struct algo_utils_hierarchy
{
    using type = utils_hierarchy_dummy;
};

template<class Algo>
struct algo_utils_hierarchy<Algo,decltype((void)(typename Algo::utils_hierarchy()),int(0))>
{
    using type = typename Algo::utils_hierarchy;
};

} // namespace detail
} // namespace numerical_algos

#endif
