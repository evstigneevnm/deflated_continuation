#ifndef __MODELS_BRATU_MODEL_CONFIG_H__
#define __MODELS_BRATU_MODEL_CONFIG_H__

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <main/parameters.hpp>

namespace bratu_model
{

inline std::string normalize_spatial_discretization_name(
    std::string name)
{
    std::transform(
        name.begin(),
        name.end(),
        name.begin(),
        [](unsigned char value)
        {
            return static_cast<char>(std::tolower(value));
        });
    if(
        name == "finite_difference" ||
        name == "finite-difference" ||
        name == "fd")
    {
        return "fd3";
    }
    if(
        name == "cheb" ||
        name == "chebyshev_lobatto" ||
        name == "chebyshev-lobatto")
    {
        return "chebyshev";
    }
    return name;
}

template<class Bratu, class Parameters>
typename Bratu::spatial_discretization
spatial_discretization_from_parameters(
    const Parameters& parameters,
    const std::string& config_file)
{
    int discretization_id = 0;
    if(!parameters.nonlinear_operator.
           problem_int_parameters_vector.empty())
    {
        discretization_id =
            parameters.nonlinear_operator.
                problem_int_parameters_vector.front();
    }

    const auto json_config =
        main_classes::read_json(config_file);
    if(json_config.contains("nonlinear_operator"))
    {
        const auto& nonlinear_operator =
            json_config.at("nonlinear_operator");
        if(nonlinear_operator.contains("spatial_discretization"))
        {
            const auto& value =
                nonlinear_operator.at("spatial_discretization");
            if(value.is_string())
            {
                const std::string name =
                    normalize_spatial_discretization_name(
                        value.template get<std::string>());
                if(name == "chebyshev")
                    discretization_id = 0;
                else if(name == "fd3")
                    discretization_id = 1;
                else
                {
                    throw std::runtime_error(
                        "unknown Bratu spatial_discretization: " +
                        value.template get<std::string>());
                }
            }
            else
            {
                discretization_id =
                    value.template get<int>();
            }
        }
    }
    return Bratu::discretization_from_int(discretization_id);
}

} // namespace bratu_model

#endif
