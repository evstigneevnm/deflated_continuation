#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

#include <common/scalar_math.h>
#include <common/gpu_file_operations.h>

#include <nonlinear_operators/bratu/bratu.h>
#include <nonlinear_operators/bratu/convergence_strategy.h>
#include <nonlinear_operators/bratu/linear_operator_bratu.h>
#include <nonlinear_operators/bratu/preconditioner_bratu.h>
#include <nonlinear_operators/bratu/system_operator.h>

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include "bratu_backend_typedefs.h"

namespace
{

void print_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " [path_to_config_file.json] [--seed-exact-theta theta] [--continue-seed-only] [--quiet]\n";
}

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
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
    using real_vec = typename vec_ops_real::vector_type;
    using bratu_t = nonlinear_operators::bratu<vec_ops_real, Blocks_x_>;
    using lin_op_t = nonlinear_operators::linear_operator_bratu<vec_ops_real, bratu_t>;
    using prec_t = nonlinear_operators::preconditioner_bratu<vec_ops_real, bratu_t, lin_op_t>;

#if defined(BRATU_VECTOR_BACKEND_VAR_PREC)
    using parameters_real = double;
    using parameters_t = main_classes::parameters<double>;
#else
    using parameters_real = real;
    using parameters_t = main_classes::parameters<real>;
#endif

    std::string path_to_config_file = "json_project_files/bratu_test.json";
    bool quiet = false;
    bool seed_exact_theta = false;
    bool continue_seed_only = false;
    real theta_seed = real(1);

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
        else if(arg == "--seed-exact-theta")
        {
            if(argi + 1 >= argc)
            {
                print_usage(argv[0]);
                return 1;
            }
            try
            {
                theta_seed = parse_scalar<real>(argv[++argi], "theta");
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
                return 2;
            }
            seed_exact_theta = true;
        }
        else if(path_to_config_file == "json_project_files/bratu_test.json")
        {
            path_to_config_file = arg;
        }
        else
        {
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Using Bratu backend: " << BRATU_BACKEND_NAME << std::endl;
    std::cout << "Reading config file: " << path_to_config_file << std::endl;

    parameters_t parameters = main_classes::read_parameters_json<parameters_real>(path_to_config_file);
    if(!quiet)
    {
        parameters.plot_all();
    }

    if(parameters.nonlinear_operator.N_size.empty())
    {
        std::cerr << "Bratu config must provide nonlinear_operator.discrete_problem_dimensions[0].\n";
        return 2;
    }
    const std::size_t interior_size = parameters.nonlinear_operator.N_size.at(0);
    const bool use_high_precision_reduction = parameters.use_high_precision_reduction;
    const auto spatial_discretization =
        spatial_discretization_from_parameters<bratu_t>(parameters, path_to_config_file);

    vec_ops_real vec_ops_R(interior_size);
    if(use_high_precision_reduction)
    {
#if defined(BRATU_VECTOR_BACKEND_VAR_PREC)
        std::cerr << "Warning: variable-precision vector operations use native reductions.\n";
#else
        vec_ops_R.use_high_precision();
#endif
    }

    files_ops_t file_ops(&vec_ops_R);
    bratu_t BRATU(interior_size, &vec_ops_R, spatial_discretization);
    std::cout << "Bratu spatial discretization: "
              << bratu_t::discretization_name(spatial_discretization) << std::endl;
    log_t log;
    log_t log_linsolver;
    log.set_verbosity(quiet ? 0 : 1);
    log_linsolver.set_verbosity(quiet ? 0 : 1);

    using deflation_continuation_t = main_classes::deflation_continuation<
        vec_ops_real,
        files_ops_t,
        log_t,
        monitor_t,
        bratu_t,
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
        &BRATU,
        &parameters);

    DC.set_parameters();

    if(seed_exact_theta)
    {
        real_vec x_seed;
        vec_ops_R.init_vector(x_seed);
        vec_ops_R.start_use_vector(x_seed);
        BRATU.exact_solution_from_theta(theta_seed, x_seed);
        const real lambda_seed = BRATU.lambda_from_theta(theta_seed);
        std::cout << "Exact Bratu seed: theta=" << theta_seed << ", lambda=" << lambda_seed << std::endl;
        DC.add_solution_curve(x_seed, lambda_seed);
        vec_ops_R.stop_use_vector(x_seed);
        vec_ops_R.free_vector(x_seed);
    }

    if(!continue_seed_only)
    {
        DC.execute();
    }

    return 0;
}
