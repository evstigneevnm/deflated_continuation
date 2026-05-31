#ifndef __COMMON_CUDA_INIT_SCFD_H__
#define __COMMON_CUDA_INIT_SCFD_H__

#include <algorithm>
#include <cctype>
#include <string>

#include <scfd/utils/init_cuda.h>

namespace common
{

inline int init_cuda_from_scfd_selector(const std::string& device_selector)
{
    if((device_selector.empty())||(device_selector == "auto"))
    {
        return scfd::utils::init_cuda(-2, 0);
    }
    if(device_selector == "best_mem")
    {
        return scfd::utils::init_cuda_persistent();
    }

    const bool is_integer = std::all_of(device_selector.begin(), device_selector.end(), [](unsigned char c)
    {
        return std::isdigit(c) != 0;
    });
    if(is_integer)
    {
        return scfd::utils::init_cuda(-2, std::stoi(device_selector));
    }

    return scfd::utils::init_cuda_str(device_selector);
}

}

#endif
