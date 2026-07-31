#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <common/scfd_backend_ext/complex.h>

#include <nonlinear_operators/bratu/bratu.h>
#include <nonlinear_operators/bratu/convergence_strategy.h>
#include <nonlinear_operators/bratu/linear_operator_bratu.h>
#include <nonlinear_operators/bratu/preconditioner_bratu.h>
#include <nonlinear_operators/bratu/system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>

#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include "bratu_backend_typedefs.h"
#include "bratu_model_config.h"

namespace
{

struct command_line_options
{
    std::string config_file =
        "json_project_files/bratu_test.json";
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
                "unexpected Bratu stability argument: " +
                argument);
        }
    }
    return result;
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
        stability::analysis::
            validate_matrix_free_stability_config(config);
        if(parameters.nonlinear_operator.N_size.empty())
        {
            throw std::invalid_argument(
                "Bratu stability requires an interior size");
        }
        const std::size_t state_size =
            parameters.nonlinear_operator.N_size.front();

        using backend_type =
            typename vec_ops_real::backend_type;
        using complex_scalar_type =
            common::scfd_backend_ext::complex_t<
                backend_type,
                real>;
        using complex_space_type =
            scfd_vector_operations<
                backend_type,
                complex_scalar_type>;
        using file_operations_type =
            gpu_file_operations<vec_ops_real>;
        using nonlinear_operator_type =
            nonlinear_operators::bratu<
                vec_ops_real,
                Blocks_x_>;
        using newton_operator_type =
            nonlinear_operators::linear_operator_bratu<
                vec_ops_real,
                nonlinear_operator_type>;
        using newton_preconditioner_type =
            nonlinear_operators::preconditioner_bratu<
                vec_ops_real,
                nonlinear_operator_type,
                newton_operator_type>;
        using base_provider_type =
            stability::eigensolvers::transformations::
                nonlinear_operator_real_affine_inverse_provider<
                    vec_ops_real,
                    nonlinear_operator_type>;
        using stability_operator_type =
            stability::eigensolvers::transformations::
                scaled_real_operator<
                    vec_ops_real,
                    newton_operator_type>;
        using stability_provider_type =
            stability::eigensolvers::transformations::
                scaled_real_affine_inverse_provider<
                    base_provider_type>;
        using factorization_types =
            stability::eigensolvers::transformations::
                matrix_free_complex_factorization_types<
                    vec_ops_real,
                    complex_space_type,
                    stability_operator_type,
                    stability_provider_type>;
        using factor_operator_type =
            typename factorization_types::factor_operator_type;
        using factor_preconditioner_type =
            typename factorization_types::preconditioner_type;
        using inner_monitor_type =
            nmfd::solvers::monitor_krylov<
                complex_space_type,
                log_t>;
        using inner_solver_type =
            nmfd::solvers::gmres<
                complex_space_type,
                inner_monitor_type,
                log_t,
                factor_operator_type,
                factor_preconditioner_type>;
        using dense_lapack_type =
            nmfd::operations::linalg::
                host_small_dense_lapack<real>;
        using eigensolver_type =
            stability::analysis::matrix_free_stability_scan<
                factorization_types,
                inner_solver_type,
                dense_lapack_type>;
        using newton_monitor_type =
            numerical_algos::lin_solvers::default_monitor<
                vec_ops_real,
                log_t>;
        using driver_type =
            main_classes::stability_continuation<
                vec_ops_real,
                file_operations_type,
                log_t,
                newton_monitor_type,
                nonlinear_operator_type,
                newton_operator_type,
                newton_preconditioner_type,
                numerical_algos::lin_solvers::exact_wrapper,
                nonlinear_operators::system_operator,
                parameters_type,
                eigensolver_type>;

        auto real_space =
            std::make_shared<vec_ops_real>(state_size);
        auto complex_space =
            std::make_shared<complex_space_type>(state_size);
        if(parameters.use_high_precision_reduction)
            real_space->use_high_precision();
        file_operations_type file_operations(real_space.get());
        const auto discretization =
            bratu_model::
                spatial_discretization_from_parameters<
                    nonlinear_operator_type>(
                        parameters,
                        command_line.config_file);
        nonlinear_operator_type nonlinear_operator(
            state_size,
            real_space.get(),
            discretization);
        newton_operator_type newton_operator(
            &nonlinear_operator);
        stability_operator_type stability_operator(
            *real_space,
            newton_operator,
            config.linearization_scale);
        auto base_provider =
            std::make_shared<base_provider_type>(
                *real_space,
                nonlinear_operator);
        auto stability_provider =
            std::make_shared<stability_provider_type>(
                base_provider,
                config.linearization_scale);

        log_t log;
        log_t linear_solver_log;
        log.set_verbosity(command_line.quiet ? 0 : 1);
        linear_solver_log.set_verbosity(
            command_line.quiet ? 0 : 1);
        eigensolver_type eigensolver(
            real_space,
            complex_space,
            stability_operator,
            stability_provider,
            stability::analysis::
                make_matrix_free_stability_scans<
                    eigensolver_type>(config),
            stability::analysis::
                make_matrix_free_inner_solver_parameters<
                    typename eigensolver_type::
                        inner_parameters_type>(config),
            stability::analysis::
                make_spectrum_scan_aggregation_options(config),
            !command_line.quiet &&
                config.inner_solver.verbose
                ? &linear_solver_log
                : nullptr);

        if(!command_line.quiet)
        {
            std::cout
                << "Using Bratu stability backend: "
                << BRATU_BACKEND_NAME << '\n'
                << "Bratu spatial discretization: "
                << nonlinear_operator_type::discretization_name(
                       discretization)
                << '\n';
            parameters.plot_all();
        }

        driver_type driver(
            real_space.get(),
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
            << "Bratu stability failed: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
