#ifndef __SCFD_ALGO_PARAMS_HIERARCHY_H__
#define __SCFD_ALGO_PARAMS_HIERARCHY_H__

#include "params_hierarchy_dummy.h"

namespace numerical_algos
{
namespace detail 
{

template<class Algo, class = int>
struct algo_params_hierarchy
{
    using type = params_hierarchy_dummy;
};

template<class Algo>
struct algo_params_hierarchy<Algo,decltype((void)(typename Algo::params_hierarchy()),int(0))>
{
    using type = typename Algo::params_hierarchy;
};

} // namespace detail
} // namespace numerical_algos

#endif
