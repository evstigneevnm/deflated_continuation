#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <main/parameters.hpp>
#include <nmfd/operations/io/vector_file_operations.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <visualization/bd_prepare_visualization.hpp>
#include <visualization/structured_physical_solution_writer.h>

#include "KS2D_backend_typedefs.h"

#if defined(KS2D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

namespace
{

bool backend_needs_device_init()
{
#if defined(KS2D_VECTOR_BACKEND_CUDA)
    return true;
#else
    return false;
#endif
}

bool is_device_selector(const std::string& value)
{
    return value == "auto" || value == "best_mem" ||
        value.rfind("dev_num:", 0) == 0 || value.rfind("pci_id:", 0) == 0;
}

int initialize_device(const std::string& selector)
{
#if defined(KS2D_VECTOR_BACKEND_CUDA)
    return common::init_cuda_from_scfd_selector(selector);
#else
    (void)selector;
    return -1;
#endif
}

void print_usage(const char* executable)
{
    visualization::print_prepare_usage(executable);
    std::cerr << "       CUDA builds additionally accept --device selector or a positional selector.\n";
}

} // namespace

int main(int argc, char const* argv[])
{
    using parameters_type = main_classes::parameters<real>;
    using state_file_operations_type =
        nmfd::operations::io::vector_file_operations<vec_ops_real>;
    using nonlinear_operator_type = nonlinear_operators::kuramoto_sivashinskiy_2d<
        vec_ops_real,
        fft_backend_t>;
    using extent_type = typename nonlinear_operator_type::extent_type;
    using index_space_type = typename nonlinear_operator_type::index_space_type;
    using writer_type = visualization::structured_physical_solution_writer<
        vec_ops_real,
        vec_ops_real,
        nonlinear_operator_type,
        2>;

    std::string config_file = "json_project_files/KS2D_test_sym.json";
    std::string device_selector = "auto";
    bool config_was_set = false;
    visualization::bd_prepare_options options;

    try
    {
        for(int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if(argument == "--device")
            {
                if(index + 1 >= argc || !backend_needs_device_init())
                {
                    throw std::invalid_argument("--device requires a CUDA visualization build and a selector");
                }
                device_selector = argv[++index];
            }
            else if(visualization::parse_common_prepare_argument(argument, index, argc, argv, options))
            {
                continue;
            }
            else if(backend_needs_device_init() && is_device_selector(argument))
            {
                device_selector = argument;
            }
            else if(!config_was_set)
            {
                config_file = argument;
                config_was_set = true;
            }
            else
            {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }

        parameters_type parameters = main_classes::read_parameters_json<real>(config_file);
        if(parameters.nonlinear_operator.N_size.size() != 2)
        {
            throw std::invalid_argument(
                "KS2D visualization requires nonlinear_operator.discrete_problem_dimensions=[Nx, Ny]");
        }
        const std::size_t nx = parameters.nonlinear_operator.N_size.at(0);
        const std::size_t ny = parameters.nonlinear_operator.N_size.at(1);
        const extent_type physical_extent(nx, ny);
        const index_space_type index_space(physical_extent);
        const std::size_t state_size = index_space.inversion_odd_state_size();
        const std::size_t physical_size = index_space.physical_size();
        if(options.project_dir.empty())
        {
            options.project_dir = parameters.path_to_project;
        }

        if(backend_needs_device_init())
        {
            const int device = initialize_device(device_selector);
            if(!options.quiet)
            {
                std::cout << "Using device " << device << '\n';
            }
        }

        real a = real(2);
        real b = real(4);
        if(!parameters.nonlinear_operator.problem_real_parameters_vector.empty())
        {
            a = parameters.nonlinear_operator.problem_real_parameters_vector.at(0);
        }
        if(parameters.nonlinear_operator.problem_real_parameters_vector.size() > 1)
        {
            b = parameters.nonlinear_operator.problem_real_parameters_vector.at(1);
        }

        vec_ops_real state_vector_operations(state_size);
        vec_ops_real physical_vector_operations(physical_size);
        state_file_operations_type file_operations(&state_vector_operations);
        nonlinear_operator_type nonlinear_operator(
            a,
            b,
            nx,
            ny,
            &state_vector_operations);

        const auto& domain_lengths = nonlinear_operator.domain_lengths();
        writer_type writer(
            &physical_vector_operations,
            &nonlinear_operator,
            typename writer_type::shape_type{nx, ny},
            typename writer_type::coordinates_type{0.0, 0.0},
            typename writer_type::coordinates_type{
                static_cast<double>(domain_lengths[0]),
                static_cast<double>(domain_lengths[1])},
            typename writer_type::periodicity_type{true, true},
            typename writer_type::axis_names_type{"x", "y"},
            "u",
            KS2D_BACKEND_NAME);
        visualization::bd_visualization_preparer<
            vec_ops_real,
            state_file_operations_type,
            writer_type> preparer(
                &state_vector_operations,
                &file_operations);

        if(!options.quiet)
        {
            std::cout << "Using KS2D visualization backend: " << KS2D_BACKEND_NAME << '\n'
                      << "KS2D visualization grid: " << nx << " x " << ny << '\n';
        }
        preparer.prepare(options, writer);
    }
    catch(const std::exception& error)
    {
        std::cerr << "KS2D visualization failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
