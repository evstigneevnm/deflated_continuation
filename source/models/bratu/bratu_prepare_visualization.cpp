#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <main/parameters.hpp>
#include <nonlinear_operators/bratu/bratu.h>
#include <visualization/bd_prepare_visualization.hpp>

#include "bratu_backend_typedefs.h"

namespace
{

std::string normalize_spatial_discretization_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c)
    {
        return static_cast<char>(std::tolower(c));
    });
    if(name == "finite_difference" || name == "finite-difference" || name == "fd")
    {
        return "fd3";
    }
    if(name == "cheb" || name == "chebyshev_lobatto" || name == "chebyshev-lobatto")
    {
        return "chebyshev";
    }
    return name;
}

template<class Bratu, class Parameters>
typename Bratu::spatial_discretization spatial_discretization_from_parameters(
    const Parameters& parameters,
    const std::string& path_to_config_file)
{
    int discretization_id = 0;
    if(!parameters.nonlinear_operator.problem_int_parameters_vector.empty())
    {
        discretization_id = parameters.nonlinear_operator.problem_int_parameters_vector.front();
    }

    const auto json_config = main_classes::read_json(path_to_config_file);
    if(json_config.contains("nonlinear_operator"))
    {
        const auto& nonlinear_operator = json_config.at("nonlinear_operator");
        if(nonlinear_operator.contains("spatial_discretization"))
        {
            const auto& value = nonlinear_operator.at("spatial_discretization");
            if(value.is_string())
            {
                const std::string name = normalize_spatial_discretization_name(value.get<std::string>());
                if(name == "chebyshev")
                {
                    discretization_id = 0;
                }
                else if(name == "fd3")
                {
                    discretization_id = 1;
                }
                else
                {
                    throw std::runtime_error("unknown Bratu spatial_discretization: " + value.get<std::string>());
                }
            }
            else
            {
                discretization_id = value.get<int>();
            }
        }
    }

    return Bratu::discretization_from_int(discretization_id);
}

} // namespace

int main(int argc, char const* argv[])
{
    using files_ops_t = gpu_file_operations<vec_ops_real>;
    using bratu_t = nonlinear_operators::bratu<vec_ops_real, Blocks_x_>;

#if defined(BRATU_VECTOR_BACKEND_VAR_PREC)
    using parameters_real = double;
    using parameters_t = main_classes::parameters<double>;
#else
    using parameters_real = real;
    using parameters_t = main_classes::parameters<real>;
#endif

    std::string path_to_config_file = "json_project_files/bratu_test.json";
    visualization::bd_prepare_options options;

    try
    {
        for(int argi = 1; argi < argc; ++argi)
        {
            const std::string arg = argv[argi];
            if(visualization::parse_common_prepare_argument(arg, argi, argc, argv, options))
            {
                continue;
            }
            if(path_to_config_file == "json_project_files/bratu_test.json")
            {
                path_to_config_file = arg;
            }
            else
            {
                visualization::print_prepare_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        parameters_t parameters = main_classes::read_parameters_json<parameters_real>(path_to_config_file);
        if(parameters.nonlinear_operator.N_size.empty())
        {
            throw std::runtime_error("Bratu visualization expects nonlinear_operator.discrete_problem_dimensions[0]");
        }
        const std::size_t size = parameters.nonlinear_operator.N_size.at(0);
        if(options.project_dir.empty())
        {
            options.project_dir = parameters.path_to_project;
        }

        vec_ops_real vec_ops(size);
        files_ops_t file_ops(&vec_ops);
        const auto discretization = spatial_discretization_from_parameters<bratu_t>(parameters, path_to_config_file);
        bratu_t bratu(size, &vec_ops, discretization);
        visualization::state_vector_writer<vec_ops_real> writer(&vec_ops, "state_scalar_1d");
        visualization::bd_visualization_preparer<vec_ops_real, files_ops_t, decltype(writer)> preparer(&vec_ops, &file_ops);
        preparer.prepare(options, writer);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
