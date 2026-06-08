#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <common/scalar_math.h>

#include <nonlinear_operators/star_shaped/convergence_strategy.h>
#include <nonlinear_operators/star_shaped/linear_operator_star_shaped.h>
#include <nonlinear_operators/star_shaped/preconditioner_star_shaped.h>
#include <nonlinear_operators/star_shaped/star_shaped.h>
#include <nonlinear_operators/star_shaped/system_operator.h>

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include "star_shaped_backend_typedefs.h"

#if defined(STAR_SHAPED_VECTOR_BACKEND_HIP)
#include <scfd/utils/init_hip.h>
#elif !defined(STAR_SHAPED_VECTOR_BACKEND_OMP) && !defined(STAR_SHAPED_VECTOR_BACKEND_VAR_PREC)
#include <scfd/utils/init_cuda.h>
#endif

namespace
{

void print_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " [path_to_config_file.json] [--seed-exact lambda] [--continue-seed-only] [--quiet]\n";
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
#if defined(STAR_SHAPED_VECTOR_BACKEND_OMP) || defined(STAR_SHAPED_VECTOR_BACKEND_VAR_PREC)
    return false;
#else
    return true;
#endif
}

int init_device_from_config(int pci_id)
{
#if defined(STAR_SHAPED_VECTOR_BACKEND_HIP)
    return scfd::utils::init_hip(pci_id);
#elif defined(STAR_SHAPED_VECTOR_BACKEND_OMP) || defined(STAR_SHAPED_VECTOR_BACKEND_VAR_PREC)
    (void)pci_id;
    return -1;
#else
    return scfd::utils::init_cuda(pci_id);
#endif
}

} // namespace

int main(int argc, char const* argv[])
{
    using files_ops_t = gpu_file_operations<vec_ops_real>;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
    using real_vec = typename vec_ops_real::vector_type;
    using star_shaped_t = nonlinear_operators::star_shaped<vec_ops_real, Blocks_x_>;
    using lin_op_t = nonlinear_operators::linear_operator_star_shaped<vec_ops_real, star_shaped_t>;
    using prec_t = nonlinear_operators::preconditioner_star_shaped<vec_ops_real, star_shaped_t, lin_op_t>;

#if defined(STAR_SHAPED_VECTOR_BACKEND_VAR_PREC)
    using parameters_real = double;
    using parameters_t = main_classes::parameters<double>;
#else
    using parameters_real = real;
    using parameters_t = main_classes::parameters<real>;
#endif

    std::string path_to_config_file = "json_project_files/star_shaped_test.json";
    bool quiet = false;
    bool seed_exact = false;
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
        else if(arg == "--seed-exact")
        {
            if(argi + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
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
            seed_exact = true;
        }
        else if(path_to_config_file == "json_project_files/star_shaped_test.json")
        {
            path_to_config_file = arg;
        }
        else
        {
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Using star_shaped backend: " << STAR_SHAPED_BACKEND_NAME << std::endl;
    std::cout << "Reading config file: " << path_to_config_file << std::endl;

    parameters_t parameters = main_classes::read_parameters_json<parameters_real>(path_to_config_file);
    if(!quiet)
    {
        parameters.plot_all();
    }

    if(parameters.nonlinear_operator.N_size.empty() || parameters.nonlinear_operator.N_size.at(0) != 1)
    {
        std::cerr << "star_shaped config must provide nonlinear_operator.discrete_problem_dimensions[0] == 1.\n";
        return 2;
    }

    if(backend_needs_device_init())
    {
        const int device = init_device_from_config(parameters.nvidia_pci_id);
        std::cout << "Using device " << device << std::endl;
    }

    real curvature = real(0.2);
    if(!parameters.nonlinear_operator.problem_real_parameters_vector.empty())
    {
        curvature = static_cast<real>(parameters.nonlinear_operator.problem_real_parameters_vector.front());
    }
    std::cout << "Star-shaped curvature C: " << curvature << std::endl;

    vec_ops_real vec_ops_R(1);
    if(parameters.use_high_precision_reduction)
    {
#if defined(STAR_SHAPED_VECTOR_BACKEND_VAR_PREC)
        std::cerr << "Warning: variable-precision vector operations use native reductions.\n";
#else
        vec_ops_R.use_high_precision();
#endif
    }

    files_ops_t file_ops(&vec_ops_R);
    star_shaped_t STAR_SHAPED(1, &vec_ops_R, curvature);
    log_t log;
    log_t log_linsolver;
    log.set_verbosity(quiet ? 0 : 1);
    log_linsolver.set_verbosity(quiet ? 0 : 1);

    using deflation_continuation_t = main_classes::deflation_continuation<
        vec_ops_real,
        files_ops_t,
        log_t,
        monitor_t,
        star_shaped_t,
        lin_op_t,
        prec_t,
        numerical_algos::lin_solvers::exact_wrapper,
        nonlinear_operators::system_operator,
        parameters_t>;

    deflation_continuation_t DC(
        &vec_ops_R,
        &file_ops,
        &log,
        &log_linsolver,
        &STAR_SHAPED,
        &parameters);

    DC.set_parameters();

    if(seed_exact)
    {
        real_vec x_seed;
        vec_ops_R.init_vector(x_seed);
        vec_ops_R.start_use_vector(x_seed);
        STAR_SHAPED.exact_solution(seed_lambda, x_seed);
        std::cout << "Exact star_shaped seed: lambda=" << seed_lambda << std::endl;
        DC.add_solution_curve(x_seed, seed_lambda);
        vec_ops_R.stop_use_vector(x_seed);
        vec_ops_R.free_vector(x_seed);
    }

    if(!continue_seed_only)
    {
        DC.execute();
    }

    return 0;
}
