#ifndef __SCFD_ALGO_HIERARCHY_CREATOR_H__
#define __SCFD_ALGO_HIERARCHY_CREATOR_H__

#include <memory>
#include "utils_hierarchy_dummy.h"
#include "params_hierarchy_dummy.h"

namespace numerical_algos
{
namespace detail 
{

template<class Algo, class = int>
struct algo_hierarchy_creator
{
    static std::shared_ptr<Algo> get(
        const utils_hierarchy_dummy &utils,
        const params_hierarchy_dummy &params
    )
    {
        throw std::logic_error("algo_hierarchy_creator::hierarchy constructor is not implemented");
        return std::shared_ptr<Algo>(nullptr);
    }
};

template<class Algo>
struct algo_hierarchy_creator<Algo,decltype((void)(Algo(typename Algo::utils_hierarchy(),typename Algo::params_hierarchy())),int(0))>
{
    static std::shared_ptr<Algo> get(
        const typename Algo::utils_hierarchy &utils,
        const typename Algo::params_hierarchy &params
    )
    {
        return std::make_shared<Algo>(utils,params);
    }
};

} // namespace detail
} // namespace numerical_algos

#endif
