#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <main/parameters.hpp>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <visualization/bd_prepare_visualization.hpp>

#include "KS1D_backend_typedefs.h"

int main(int argc, char const* argv[])
{
    using state_files_ops_t = gpu_file_operations<vec_ops_real>;
    using physical_vec_ops_t = vec_ops_real;
    using ks1d_t = nonlinear_operators::kuramoto_sivashinskiy_1d<vec_ops_real, fft_backend_t, Blocks_x_>;
    using parameters_t = main_classes::parameters<real>;

    std::string path_to_config_file = "json_project_files/KS1D_test_sym.json";
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
            if(path_to_config_file == "json_project_files/KS1D_test_sym.json")
            {
                path_to_config_file = arg;
            }
            else
            {
                visualization::print_prepare_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file);
        if(parameters.nonlinear_operator.N_size.empty())
        {
            throw std::runtime_error("KS1D visualization expects nonlinear_operator.discrete_problem_dimensions[0]");
        }
        const std::size_t physical_size = parameters.nonlinear_operator.N_size.at(0);
        if(physical_size < 4 || physical_size%2 != 0)
        {
            throw std::runtime_error("KS1D visualization expects an even physical size >= 4");
        }
        const std::size_t reduced_size = physical_size/2 - 1;
        if(options.project_dir.empty())
        {
            options.project_dir = parameters.path_to_project;
        }

        real a_val = real(2);
        real b_val = real(4);
        if(!parameters.nonlinear_operator.problem_real_parameters_vector.empty())
        {
            a_val = parameters.nonlinear_operator.problem_real_parameters_vector.at(0);
        }
        if(parameters.nonlinear_operator.problem_real_parameters_vector.size() > 1)
        {
            b_val = parameters.nonlinear_operator.problem_real_parameters_vector.at(1);
        }

        vec_ops_real state_vec_ops(reduced_size);
        physical_vec_ops_t physical_vec_ops(physical_size);
        state_files_ops_t file_ops(&state_vec_ops);
        ks1d_t ks1d(a_val, b_val, physical_size, &state_vec_ops);
        visualization::physical_solution_writer<vec_ops_real, physical_vec_ops_t, ks1d_t> writer(
            &physical_vec_ops,
            &ks1d,
            physical_size,
            2.0*std::acos(-1.0));
        visualization::bd_visualization_preparer<vec_ops_real, state_files_ops_t, decltype(writer)> preparer(
            &state_vec_ops,
            &file_ops);
        preparer.prepare(options, writer);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
