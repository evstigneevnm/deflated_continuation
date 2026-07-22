#ifndef __MAIN_READ_PARAMETERS_JSON_H__
#define __MAIN_READ_PARAMETERS_JSON_H__

#include <fstream>
#include <stdexcept>
#include <string>

#include <main/parameters/parameters_json.h>

namespace main_classes
{

inline nlohmann::json read_json(const std::string& project_file_name)
{
    try
    {
        std::ifstream file(project_file_name);
        if(!file)
        {
            throw std::runtime_error("Failed to open file " + project_file_name + " for reading");
        }
        nlohmann::json json;
        file >> json;
        return json;
    }
    catch(const nlohmann::json::exception& exception)
    {
        std::throw_with_nested(
            std::runtime_error{"json path: " + project_file_name + "\n" + exception.what()});
    }
}

template<class T>
parameters<T> read_parameters_json(const std::string& project_file_name)
{
    try
    {
        return read_json(project_file_name).template get<parameters<T>>();
    }
    catch(const std::exception& exception)
    {
        std::throw_with_nested(
            std::runtime_error{
                "failed to read parameters JSON file: " + project_file_name + "\n" + exception.what()});
    }
}

} // namespace main_classes

#endif // __MAIN_READ_PARAMETERS_JSON_H__
