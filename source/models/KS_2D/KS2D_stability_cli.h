#ifndef __KS2D_STABILITY_CLI_H__
#define __KS2D_STABILITY_CLI_H__

#include <stdexcept>
#include <string>

#if defined(KS2D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

namespace ks2d_stability_model
{

struct command_line_options
{
    std::string config_file;
    std::string device_selector = "auto";
    bool quiet = false;
    bool edit = false;
};

inline bool backend_needs_device_init()
{
#if defined(KS2D_VECTOR_BACKEND_CUDA)
    return true;
#else
    return false;
#endif
}

inline bool is_device_selector(const std::string& value)
{
    return value == "auto" || value == "best_mem" ||
        value.rfind("dev_num:", 0) == 0 || value.rfind("pci_id:", 0) == 0;
}

inline command_line_options parse_command_line(int argc, char** argv, std::string default_config)
{
    command_line_options result;
    result.config_file = std::move(default_config);
    bool config_was_set = false;
    for(int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if(argument == "--quiet")
        {
            result.quiet = true;
        }
        else if(argument == "--edit")
        {
            result.edit = true;
        }
        else if(backend_needs_device_init() && is_device_selector(argument))
        {
            result.device_selector = argument;
        }
        else if(!config_was_set)
        {
            result.config_file = argument;
            config_was_set = true;
        }
        else if(backend_needs_device_init())
        {
            result.device_selector = argument;
        }
        else
        {
            throw std::invalid_argument("unexpected KS2D stability argument: " + argument);
        }
    }
    return result;
}

inline int initialize_device(const std::string& selector)
{
#if defined(KS2D_VECTOR_BACKEND_CUDA)
    return common::init_cuda_from_scfd_selector(selector);
#else
    (void)selector;
    return -1;
#endif
}

} // namespace ks2d_stability_model

#endif
