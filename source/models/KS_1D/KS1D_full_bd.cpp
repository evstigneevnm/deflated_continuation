#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>

#include <deflation/symmetry_solution_storage.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_policy_json.h>

#include <continuation/projected_system_operator_continuation.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_preconditioner.h>
#include <nonlinear_operators/projected_system_operator.h>

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
        << " [path_to_config_file.json] [--seed-zero lambda | --seed-file lambda vector-file]"
        << " [--continue-seed-only] [--quiet] [device]\n";
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

template<class Policy>
void print_symmetry_stabilizer_policy(const char* label, const Policy& policy)
{
    std::cout
        << label << ": type="
        << symmetry::fourier::real_packed_fourier_1d_stabilizer_policy_name(policy.stabilizer)
        << ", relative_active_mode_tolerance=" << policy.relative_active_mode_tolerance
        << ", continuation_mode_switch_ratio=" << policy.continuation_mode_switch_ratio
        << ", tangent_continuity_weight=" << policy.tangent_continuity_weight
        << ", tangent_backward_penalty=" << policy.tangent_backward_penalty
        << ", mode_min=" << policy.lsq.mode_min
        << ", mode_max=" << policy.lsq.mode_max
        << ", max_active_modes=" << policy.lsq.max_active_modes
        << ", grid_points=" << policy.lsq.grid_points
        << ", newton_iterations=" << policy.lsq.newton_iterations
        << ", prefer_trivial_residual_group=" << (policy.lsq.prefer_trivial_residual_group ? 1 : 0)
        << ", minimum_coprime_relative_score=" << policy.lsq.minimum_coprime_relative_score
        << ", local_representative_relative_tolerance="
        << policy.local_representative_relative_tolerance
        << std::endl;
}

} // namespace

int main(int argc, char const* argv[])
{
    using files_ops_t = gpu_file_operations<vec_ops_real>;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
    using real_vec = typename vec_ops_real::vector_type;
    using ks1d_t = nonlinear_operators::kuramoto_sivashinskiy_1d_full<vec_ops_real, fft_backend_t, Blocks_x_>;
    using lin_op_t = symmetry::linearization::projected_linear_operator<vec_ops_real, ks1d_t>;
    using prec_t = symmetry::linearization::projected_preconditioner<vec_ops_real, ks1d_t, lin_op_t>;
    using parameters_t = main_classes::parameters<real>;
    using symmetry_adapter_t = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vec_ops_real>;
    using finite_actions_t = symmetry::finite_action_registry<vec_ops_real>;
    using quotient_adapter_t = symmetry::finite_quotient_adapter<vec_ops_real, symmetry_adapter_t>;
    using sol_storage_t = deflation::symmetry_solution_storage<vec_ops_real, quotient_adapter_t, log_t>;

    std::string path_to_config_file = "json_project_files/KS1D_test_sym.json";
    std::string device_selector = "auto";
    bool quiet = false;
    bool seed_zero = false;
    bool seed_file = false;
    bool continue_seed_only = false;
    real seed_lambda = real(0);
    std::string seed_file_path;

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
            if(seed_file)
            {
                std::cerr << "--seed-zero and --seed-file are mutually exclusive\n";
                return 2;
            }
            seed_zero = true;
        }
        else if(arg == "--seed-file")
        {
            if(argi + 2 >= argc)
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
            seed_file_path = argv[++argi];
            if(seed_zero)
            {
                std::cerr << "--seed-zero and --seed-file are mutually exclusive\n";
                return 2;
            }
            seed_file = true;
        }
        else if(backend_needs_device_init() && (arg == "auto" || arg == "best_mem" || arg.rfind("dev_num:", 0) == 0 || arg.rfind("pci_id:", 0) == 0))
        {
            device_selector = arg;
        }
        else if(path_to_config_file == "json_project_files/KS1D_test_sym.json")
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

    std::cout << "Using full Fourier KS1D backend: " << KS1D_BACKEND_NAME << std::endl;
    std::cout << "Reading config file: " << path_to_config_file << std::endl;

    parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file);
    if(!quiet)
    {
        parameters.plot_all();
    }

    if(parameters.nonlinear_operator.N_size.empty())
    {
        std::cerr << "Full Fourier KS1D config must provide nonlinear_operator.discrete_problem_dimensions[0].\n";
        return 2;
    }
    const std::size_t physical_size = parameters.nonlinear_operator.N_size.at(0);
    if(physical_size < 4 || physical_size%2 != 0)
    {
        std::cerr << "Full Fourier KS1D physical grid size must be even and at least 4.\n";
        return 2;
    }
    const std::size_t positive_modes = physical_size/2 - 1;
    const std::size_t state_size = 2*positive_modes;

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
    std::cout << "Full Fourier KS1D parameters: N=" << physical_size
              << ", positive modes=" << positive_modes
              << ", state size=" << state_size
              << ", a=" << a_val
              << ", b=" << b_val << std::endl;

    vec_ops_real vec_ops_R(state_size);
    if(parameters.use_high_precision_reduction)
    {
        vec_ops_R.use_high_precision();
    }

    files_ops_t file_ops(&vec_ops_R);
    ks1d_t KS1D(a_val, b_val, physical_size, &vec_ops_R);
    symmetry_adapter_t symmetry_adapter(&vec_ops_R, positive_modes);
    const auto config_json = main_classes::read_json(path_to_config_file);
    auto storage_symmetry_policy =
        symmetry::fourier::read_real_packed_fourier_slice_1d_policy<real>(
            config_json,
            "symmetry_stabilizer",
            ks1d_t::default_continuation_symmetry_policy());
    auto continuation_symmetry_policy =
        symmetry::fourier::read_real_packed_fourier_slice_1d_policy<real>(
            config_json,
            "continuation_symmetry_stabilizer",
            ks1d_t::default_continuation_symmetry_policy());
    if(continuation_symmetry_policy.stabilizer !=
       symmetry::fourier::real_packed_fourier_1d_stabilizer_policy::single_mode)
    {
        throw std::runtime_error(
            "LSQ continuation is disabled until its frozen-chart Jacobian and corrector tests pass");
    }
    symmetry_adapter.configure(storage_symmetry_policy);
    KS1D.configure_continuation_symmetry(continuation_symmetry_policy);
    if(!quiet)
    {
        print_symmetry_stabilizer_policy("symmetry_stabilizer", storage_symmetry_policy);
        print_symmetry_stabilizer_policy(
            "continuation_symmetry_stabilizer", continuation_symmetry_policy);
    }
    finite_actions_t finite_actions(&vec_ops_R);
    KS1D.configure_finite_symmetry_actions(finite_actions);
    quotient_adapter_t quotient_adapter(&vec_ops_R, &symmetry_adapter, &finite_actions);

    log_t log;
    log_t log_linsolver;
    log.set_verbosity(quiet ? 0 : 1);
    log_linsolver.set_verbosity(quiet ? 0 : 1);
    sol_storage_t sol_storage_with_log(&vec_ops_R, 50, vec_ops_R.get_l2_size(), real(2), &quotient_adapter, 1e-10, &log);

    using deflation_continuation_t = main_classes::deflation_continuation<
        vec_ops_real,
        files_ops_t,
        log_t,
        monitor_t,
        ks1d_t,
        lin_op_t,
        prec_t,
        numerical_algos::lin_solvers::bicgstabl,
        nonlinear_operators::projected_system_operator,
        parameters_t,
        sol_storage_t,
        continuation::projected_system_operator_continuation>;

    deflation_continuation_t DC(
        &vec_ops_R,
        &file_ops,
        &log,
        &log_linsolver,
        &KS1D,
        &parameters,
        &sol_storage_with_log);

    DC.set_parameters();

    if(continue_seed_only && (seed_zero || seed_file))
    {
        DC.load_data(parameters.bifurcaiton_diagram_file_name);
    }

    if(seed_zero)
    {
        real_vec x_seed;
        vec_ops_R.init_vector(x_seed);
        vec_ops_R.start_use_vector(x_seed);
        KS1D.exact_solution(seed_lambda, x_seed);
        std::cout << "Exact full Fourier KS1D zero-branch seed: lambda=" << seed_lambda << std::endl;
        DC.add_solution_curve(x_seed, seed_lambda);
        vec_ops_R.stop_use_vector(x_seed);
        vec_ops_R.free_vector(x_seed);
    }

    if(seed_file)
    {
        real_vec x_seed;
        vec_ops_R.init_vector(x_seed);
        vec_ops_R.start_use_vector(x_seed);
        file_ops.read_vector(seed_file_path, x_seed);
        std::cout << "Loaded full Fourier KS1D seed: lambda=" << seed_lambda
                  << ", file=" << seed_file_path << std::endl;
        DC.add_solution_curve(x_seed, seed_lambda);
        vec_ops_R.stop_use_vector(x_seed);
        vec_ops_R.free_vector(x_seed);
    }

    if(!continue_seed_only || (!seed_zero && !seed_file))
    {
        DC.execute();
    }

    return EXIT_SUCCESS;
}
