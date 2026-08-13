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
    std::string state_file;
    double state_parameter = 0.0;
    bool state_parameter_set = false;
    std::string second_state_file;
    double second_state_parameter = 0.0;
    bool second_state_parameter_set = false;
    bool quiet = false;
    bool edit = false;
    bool confirm = false;
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
        else if(argument == "--confirm")
        {
            result.confirm = true;
        }
        else if(argument == "--edit")
        {
            result.edit = true;
        }
        else if(argument == "--state-file")
        {
            if(index + 1 >= argc)
                throw std::invalid_argument(
                    "--state-file requires a file name");
            result.state_file = argv[++index];
        }
        else if(argument == "--parameter")
        {
            if(index + 1 >= argc)
                throw std::invalid_argument(
                    "--parameter requires a value");
            result.state_parameter = std::stod(argv[++index]);
            result.state_parameter_set = true;
        }
        else if(argument == "--state-file-2")
        {
            if(index + 1 >= argc)
                throw std::invalid_argument(
                    "--state-file-2 requires a file name");
            result.second_state_file = argv[++index];
        }
        else if(argument == "--parameter-2")
        {
            if(index + 1 >= argc)
                throw std::invalid_argument(
                    "--parameter-2 requires a value");
            result.second_state_parameter = std::stod(argv[++index]);
            result.second_state_parameter_set = true;
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
    if(result.state_file.empty() != !result.state_parameter_set)
    {
        throw std::invalid_argument(
            "--state-file and --parameter must be provided together");
    }
    if(result.second_state_file.empty() != !result.second_state_parameter_set)
    {
        throw std::invalid_argument(
            "--state-file-2 and --parameter-2 must be provided together");
    }
    if(!result.second_state_file.empty() && result.state_file.empty())
    {
        throw std::invalid_argument(
            "two-state transition replay requires both state pairs");
    }
    if(result.edit && !result.state_file.empty())
    {
        throw std::invalid_argument(
            "--edit cannot be combined with --state-file");
    }
    if(result.confirm && result.state_file.empty())
    {
        throw std::invalid_argument(
            "--confirm requires --state-file");
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
