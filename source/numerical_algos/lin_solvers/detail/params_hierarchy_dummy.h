#ifndef __SCFD_PARAMS_HIERARCHY_DUMMY_H__
#define __SCFD_PARAMS_HIERARCHY_DUMMY_H__

#include <string>

namespace numerical_algos
{
namespace detail 
{

struct params_hierarchy_dummy
{
    params_hierarchy_dummy(const std::string &log_prefix = "", const std::string &log_name = "")
    {
    }
    #ifdef SCFD_ENABLE_NLOHMANN
    void from_json(const nlohmann::json& j)
    {
    }
    nlohmann::json to_json() const
    {
        return nlohmann::json();
    }
    #endif
};

} // namespace detail
} // namespace numerical_algos

#endif
