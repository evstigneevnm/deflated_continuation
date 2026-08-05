#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <deflation/symmetry_solution_storage.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

#include "KS2D_backend_typedefs.h"

#if defined(KS2D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

namespace
{

void print_usage(const char* executable)
{
    std::cerr << "Usage: " << executable
              << " [config.json] [--seed-zero lambda]"
              << " [--replay-seed-file file lambda]"
              << " [--continue-seed-only] [--quiet] [device]\n";
}

template<class T>
T parse_scalar(const std::string& value)
{
    std::istringstream stream(value);
    T result{};
    stream >> result;
    if(!stream)
    {
        throw std::invalid_argument("failed to parse seed parameter: " + value);
    }
    return result;
}

bool backend_needs_device_init()
{
#if defined(KS2D_VECTOR_BACKEND_CUDA)
    return true;
#else
    return false;
#endif
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

} // namespace

int main(int argc, char** argv)
{
    using file_operations_type = gpu_file_operations<vec_ops_real>;
    using monitor_type = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
    using vector_type = typename vec_ops_real::vector_type;
    using nonlinear_operator_type = nonlinear_operators::kuramoto_sivashinskiy_2d<
        vec_ops_real,
        fft_backend_t
    >;
    using linear_operator_type = nonlinear_operators::linear_operator_KS_2D<
        vec_ops_real,
        nonlinear_operator_type
    >;
    using preconditioner_type = nonlinear_operators::preconditioner_KS_2D<
        vec_ops_real,
        nonlinear_operator_type,
        linear_operator_type
    >;
    using finite_actions_type = symmetry::finite_action_registry<vec_ops_real>;
    using identity_adapter_type = deflation::identity_symmetry_adapter<vec_ops_real>;
    using residual_translation_aligner_type =
        symmetry::fourier::residual_translation_orbit_aligner_2d<
            vec_ops_real>;
    using quotient_adapter_type = symmetry::finite_quotient_adapter<
        vec_ops_real,
        identity_adapter_type,
        residual_translation_aligner_type>;
    using solution_storage_type = deflation::symmetry_solution_storage<
        vec_ops_real,
        quotient_adapter_type,
        log_t>;
    using parameters_type = main_classes::parameters<real>;

    std::string config_file = "json_project_files/KS2D_test_sym.json";
    std::string device_selector = "auto";
    bool config_was_set = false;
    bool quiet = false;
    bool seed_zero = false;
    bool replay_seed_file = false;
    bool continue_seed_only = false;
    real seed_parameter = real(0);
    std::string seed_file;
    for(int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if(argument == "--quiet")
        {
            quiet = true;
        }
        else if(argument == "--continue-seed-only")
        {
            continue_seed_only = true;
        }
        else if(argument == "--seed-zero")
        {
            if(index + 1 >= argc)
            {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            seed_parameter = parse_scalar<real>(argv[++index]);
            seed_zero = true;
        }
        else if(argument == "--replay-seed-file")
        {
            if(index + 2 >= argc)
            {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            seed_file = argv[++index];
            seed_parameter = parse_scalar<real>(argv[++index]);
            replay_seed_file = true;
        }
        else if(backend_needs_device_init() &&
            (argument == "auto" || argument == "best_mem" ||
             argument.rfind("dev_num:", 0) == 0 || argument.rfind("pci_id:", 0) == 0))
        {
            device_selector = argument;
        }
        else if(!config_was_set)
        {
            config_file = argument;
            config_was_set = true;
        }
        else if(backend_needs_device_init())
        {
            device_selector = argument;
        }
        else
        {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    try
    {
        if(seed_zero && replay_seed_file)
        {
            throw std::invalid_argument(
                "--seed-zero and --replay-seed-file are mutually exclusive");
        }
        parameters_type parameters = main_classes::read_parameters_json<real>(config_file);
        if(parameters.nonlinear_operator.N_size.size() != 2)
        {
            throw std::invalid_argument("KS2D requires nonlinear_operator.discrete_problem_dimensions=[Nx, Ny]");
        }
        const std::size_t nx = parameters.nonlinear_operator.N_size.at(0);
        const std::size_t ny = parameters.nonlinear_operator.N_size.at(1);
        if(nx < 4 || ny < 4 || nx%2 != 0 || ny%2 != 0)
        {
            throw std::invalid_argument("KS2D grid dimensions must be even and at least four");
        }
        if(backend_needs_device_init())
        {
            std::cout << "Using device " << initialize_device(device_selector) << '\n';
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
        const std::size_t state_size = nx*ny/2 - 2;
        std::cout << "Using KS2D backend: " << KS2D_BACKEND_NAME << '\n'
                  << "KS2D parameters: Nx=" << nx << ", Ny=" << ny
                  << ", state size=" << state_size << ", a=" << a << ", b=" << b << '\n';
        if(!quiet)
        {
            parameters.plot_all();
        }

        vec_ops_real vector_operations(state_size);
        if(parameters.use_high_precision_reduction)
        {
            vector_operations.use_high_precision();
        }
        file_operations_type file_operations(&vector_operations);
        nonlinear_operator_type nonlinear_operator(a, b, nx, ny, &vector_operations);
        log_t log;
        log_t linear_solver_log;
        log.set_verbosity(quiet ? 0 : 1);
        linear_solver_log.set_verbosity(quiet ? 0 : 1);
        finite_actions_type finite_actions(&vector_operations);
        const auto finite_symmetry_group =
            nonlinear_operator.finite_symmetry_group();
        const auto finite_action_workspace =
            nonlinear_operator.configure_finite_symmetry_actions(
                finite_actions,
                finite_symmetry_group);
        if(!quiet)
        {
            std::cout << "KS2D finite symmetry group: order="
                      << finite_symmetry_group.size()
                      << ", fingerprint="
                      << finite_symmetry_group.fingerprint() << '\n';
        }
        identity_adapter_type identity_adapter(&vector_operations);
        residual_translation_aligner_type residual_translation_aligner(
            &vector_operations,
            nonlinear_operator.state_modes());
        quotient_adapter_type quotient_adapter(
            &vector_operations,
            &identity_adapter,
            &finite_actions,
            &residual_translation_aligner);
        solution_storage_type solution_storage(
            &vector_operations,
            50,
            vector_operations.get_l2_size(),
            real(2),
            &quotient_adapter,
            static_cast<double>(
                parameters.deflation_continuation.restart_policy.
                    duplicate_after_deflation_tolerance),
            &log);

        using driver_type = main_classes::deflation_continuation<
            vec_ops_real,
            file_operations_type,
            log_t,
            monitor_type,
            nonlinear_operator_type,
            linear_operator_type,
            preconditioner_type,
            numerical_algos::lin_solvers::bicgstabl,
            nonlinear_operators::system_operator,
            parameters_type,
            solution_storage_type
        >;
        driver_type driver(
            &vector_operations,
            &file_operations,
            &log,
            &linear_solver_log,
            &nonlinear_operator,
            &parameters,
            &solution_storage
        );
        (void)finite_action_workspace;
        driver.set_parameters();
        if(replay_seed_file)
        {
            driver.execute();
            vector_type seed;
            vector_operations.init_vector(seed);
            vector_operations.start_use_vector(seed);
            file_operations.read_vector(seed_file, seed);
            driver.add_solution_curve(seed, seed_parameter);
            vector_operations.stop_use_vector(seed);
            vector_operations.free_vector(seed);
            return EXIT_SUCCESS;
        }
        if(seed_zero)
        {
            vector_type seed;
            vector_operations.init_vector(seed);
            vector_operations.start_use_vector(seed);
            nonlinear_operator.exact_solution(seed_parameter, seed);
            driver.add_solution_curve(seed, seed_parameter);
            vector_operations.stop_use_vector(seed);
            vector_operations.free_vector(seed);
        }
        if(!continue_seed_only || !seed_zero)
        {
            driver.execute();
        }
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr << "KS2D BD failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
