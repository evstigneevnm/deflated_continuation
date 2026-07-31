#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>

#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>

#include <stability/eigensolvers/direct_scalar_eigensolver.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include "circle_backend_typedefs.h"

#if defined(CIRCLE_VECTOR_BACKEND_HIP)
#include <scfd/utils/init_hip.h>
#elif !defined(CIRCLE_VECTOR_BACKEND_OMP)
#include <scfd/utils/init_cuda.h>
#endif

namespace
{

struct command_line_options
{
    std::string config_file =
        "json_project_files/circle_test.json";
    bool quiet = false;
    bool edit = false;
};

command_line_options parse_command_line(int argc, char** argv)
{
    command_line_options result;
    bool config_was_set = false;
    for(int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if(argument == "--quiet")
            result.quiet = true;
        else if(argument == "--edit")
            result.edit = true;
        else if(!config_was_set)
        {
            result.config_file = argument;
            config_was_set = true;
        }
        else
        {
            throw std::invalid_argument(
                "unexpected circle stability argument: " +
                argument);
        }
    }
    return result;
}

void initialize_device(int pci_id)
{
#if defined(CIRCLE_VECTOR_BACKEND_HIP)
    std::cout
        << "Using device "
        << scfd::utils::init_hip(pci_id) << '\n';
#elif defined(CIRCLE_VECTOR_BACKEND_OMP)
    (void)pci_id;
#else
    std::cout
        << "Using device "
        << scfd::utils::init_cuda(pci_id) << '\n';
#endif
}

void validate_direct_configuration(
    const stability::analysis::
        matrix_free_stability_config<real>& config)
{
    using std::isfinite;
    if(!config.enabled)
    {
        throw std::invalid_argument(
            "circle stability eigensolver is disabled");
    }
    if(
        !isfinite(config.linearization_scale) ||
        config.linearization_scale == real{})
    {
        throw std::invalid_argument(
            "circle stability linearization scale must be finite "
            "and nonzero");
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto command_line =
            parse_command_line(argc, argv);
        using parameters_type = main_classes::parameters<real>;
        parameters_type parameters =
            main_classes::read_parameters_json<real>(
                command_line.config_file);
        const auto& config =
            parameters.stability_continuation.
                matrix_free_eigensolver;
        validate_direct_configuration(config);
        if(
            parameters.nonlinear_operator.N_size.size() != 1 ||
            parameters.nonlinear_operator.N_size.front() != 1)
        {
            throw std::invalid_argument(
                "circle stability requires vector dimension one");
        }
        initialize_device(parameters.nvidia_pci_id);

        using file_operations_type =
            gpu_file_operations<vec_ops_real>;
        using nonlinear_operator_type =
            nonlinear_operators::circle<
                vec_ops_real,
                Blocks_x_>;
        using newton_operator_type =
            nonlinear_operators::linear_operator_circle<
                vec_ops_real,
                nonlinear_operator_type>;
        using newton_preconditioner_type =
            nonlinear_operators::preconditioner_circle<
                vec_ops_real,
                nonlinear_operator_type,
                newton_operator_type>;
        using stability_operator_type =
            stability::eigensolvers::transformations::
                scaled_real_operator<
                    vec_ops_real,
                    newton_operator_type>;
        using eigensolver_type =
            stability::eigensolvers::
                direct_scalar_eigensolver<
                    vec_ops_real,
                    stability_operator_type>;
        using monitor_type =
            numerical_algos::lin_solvers::default_monitor<
                vec_ops_real,
                log_t>;
        using driver_type =
            main_classes::stability_continuation<
                vec_ops_real,
                file_operations_type,
                log_t,
                monitor_type,
                nonlinear_operator_type,
                newton_operator_type,
                newton_preconditioner_type,
                numerical_algos::lin_solvers::bicgstabl,
                nonlinear_operators::system_operator,
                parameters_type,
                eigensolver_type>;

        vec_ops_real vector_space(1);
        if(parameters.use_high_precision_reduction)
            vector_space.use_high_precision();
        file_operations_type file_operations(&vector_space);
        nonlinear_operator_type nonlinear_operator(
            real(1),
            1,
            &vector_space);
        nonlinear_operator_type* nonlinear_operator_pointer =
            &nonlinear_operator;
        newton_operator_type newton_operator(
            nonlinear_operator_pointer);
        stability_operator_type stability_operator(
            vector_space,
            newton_operator,
            config.linearization_scale);
        eigensolver_type eigensolver(
            vector_space,
            stability_operator);
        log_t log;
        log_t linear_solver_log;
        log.set_verbosity(command_line.quiet ? 0 : 1);
        linear_solver_log.set_verbosity(
            command_line.quiet ? 0 : 1);

        if(!command_line.quiet)
        {
            std::cout
                << "Using circle stability backend: "
                << CIRCLE_BACKEND_NAME << '\n';
            parameters.plot_all();
        }

        driver_type driver(
            &vector_space,
            &file_operations,
            &log,
            &linear_solver_log,
            &nonlinear_operator,
            &parameters,
            &eigensolver);
        driver.set_parameters();
        if(command_line.edit)
            driver.edit();
        else
            driver.execute();
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "Circle stability failed: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
