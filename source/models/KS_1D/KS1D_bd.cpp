#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/system_operator.h>

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>

#include "KS1D_backend_typedefs.h"

#if defined(KS1D_VECTOR_BACKEND_CUDA)
#include <common/cuda_init_scfd.h>
#endif

namespace
{

void print_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " [path_to_config_file.json] [--seed-zero lambda] [--continue-seed-only] [--quiet] [device]\n";
}

template<class T>
T parse_scalar(const std::string& value, const char* label)
{
    std::istringstream stream(value);
    T result = T(0);
    stream >> result;
    if(!stream)
    {
        throw std::runtime_error(std::string("failed to parse ") + label + " from '" + value + "'");
    }
    return result;
}

bool backend_needs_device_init()
{
#if defined(KS1D_VECTOR_BACKEND_CUDA)
    return true;
#else
    return false;
#endif
}

int init_device_from_selector(const std::string& selector)
{
#if defined(KS1D_VECTOR_BACKEND_CUDA)
    return common::init_cuda_from_scfd_selector(selector);
#else
    (void)selector;
    return -1;
#endif
}

} // namespace

int main(int argc, char const* argv[])
{
    using files_ops_t = gpu_file_operations<vec_ops_real>;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
    using real_vec = typename vec_ops_real::vector_type;
    using ks1d_t = nonlinear_operators::kuramoto_sivashinskiy_1d<vec_ops_real, fft_backend_t, Blocks_x_>;
    using lin_op_t = nonlinear_operators::linear_operator_KS_1D<vec_ops_real, ks1d_t>;
    using prec_t = nonlinear_operators::preconditioner_KS_1D<vec_ops_real, ks1d_t, lin_op_t>;
    using parameters_t = main_classes::parameters<real>;

    std::string path_to_config_file = "json_project_files/KS1D_sym_test.json";
    std::string device_selector = "auto";
    bool quiet = false;
    bool seed_zero = false;
    bool continue_seed_only = false;
    real seed_lambda = real(0);

    for(int argi = 1; argi < argc; ++argi)
    {
        const std::string arg = argv[argi];
        if(arg == "--quiet")
        {
            quiet = true;
        }
        else if(arg == "--continue-seed-only")
        {
            continue_seed_only = true;
        }
        else if(arg == "--seed-zero")
        {
            if(argi + 1 >= argc)
            {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            try
            {
                seed_lambda = parse_scalar<real>(argv[++argi], "seed lambda");
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                return 2;
            }
            seed_zero = true;
        }
        else if(backend_needs_device_init() && (arg == "auto" || arg == "best_mem" || arg.rfind("dev_num:", 0) == 0 || arg.rfind("pci_id:", 0) == 0))
        {
            device_selector = arg;
        }
        else if(path_to_config_file == "json_project_files/KS1D_sym_test.json")
        {
            path_to_config_file = arg;
        }
        else if(backend_needs_device_init())
        {
            device_selector = arg;
        }
        else
        {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    std::cout << "Using KS1D backend: " << KS1D_BACKEND_NAME << std::endl;
    std::cout << "Reading config file: " << path_to_config_file << std::endl;

    parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file);
    if(!quiet)
    {
        parameters.plot_all();
    }

    if(parameters.nonlinear_operator.N_size.empty())
    {
        std::cerr << "KS1D config must provide nonlinear_operator.discrete_problem_dimensions[0].\n";
        return 2;
    }
    const std::size_t physical_size = parameters.nonlinear_operator.N_size.at(0);
    if(physical_size < 4 || physical_size%2 != 0)
    {
        std::cerr << "KS1D physical grid size must be even and at least 4.\n";
        return 2;
    }
    const std::size_t reduced_size = physical_size/2 - 1;

    if(backend_needs_device_init())
    {
        try
        {
            const int device = init_device_from_selector(device_selector);
            std::cout << "Using device " << device << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << "Failed to initialize device selector '" << device_selector << "': " << e.what() << std::endl;
            return 2;
        }
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
    std::cout << "KS1D parameters: N=" << physical_size
              << ", reduced modes=" << reduced_size
              << ", a=" << a_val
              << ", b=" << b_val << std::endl;

    vec_ops_real vec_ops_R(reduced_size);
    if(parameters.use_high_precision_reduction)
    {
        vec_ops_R.use_high_precision();
    }

    files_ops_t file_ops(&vec_ops_R);
    ks1d_t KS1D(a_val, b_val, physical_size, &vec_ops_R);
    log_t log;
    log_t log_linsolver;
    log.set_verbosity(quiet ? 0 : 1);
    log_linsolver.set_verbosity(quiet ? 0 : 1);

    using deflation_continuation_t = main_classes::deflation_continuation<
        vec_ops_real,
        files_ops_t,
        log_t,
        monitor_t,
        ks1d_t,
        lin_op_t,
        prec_t,
        numerical_algos::lin_solvers::bicgstabl,
        nonlinear_operators::system_operator,
        parameters_t>;

    deflation_continuation_t DC(
        &vec_ops_R,
        &file_ops,
        &log,
        &log_linsolver,
        &KS1D,
        &parameters);

    DC.set_parameters();

    if(seed_zero)
    {
        real_vec x_seed;
        vec_ops_R.init_vector(x_seed);
        vec_ops_R.start_use_vector(x_seed);
        KS1D.exact_solution(seed_lambda, x_seed);
        std::cout << "Exact KS1D zero-branch seed: lambda=" << seed_lambda << std::endl;
        DC.add_solution_curve(x_seed, seed_lambda);
        vec_ops_R.stop_use_vector(x_seed);
        vec_ops_R.free_vector(x_seed);
    }

    if(!continue_seed_only || !seed_zero)
    {
        DC.execute();
    }

    return EXIT_SUCCESS;
}
