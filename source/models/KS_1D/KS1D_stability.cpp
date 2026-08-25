#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <common/gpu_file_operations.h>
#include <common/scfd_backend_ext/complex.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>

#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>

#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include "KS1D_backend_typedefs.h"
#include "KS1D_stability_cli.h"

int main(int argc, char** argv)
{
    try
    {
        const auto command_line =
            ks1d_stability_model::parse_command_line(
                argc,
                argv,
                "json_project_files/KS1D_test_sym.json");
        using parameters_type = main_classes::parameters<real>;
        parameters_type parameters =
            main_classes::read_parameters_json<real>(
                command_line.config_file);
        const auto& config =
            parameters.stability_continuation.
                matrix_free_eigensolver;
        stability::analysis::
            validate_matrix_free_stability_config(config);

        if(
            parameters.nonlinear_operator.N_size.empty() ||
            parameters.nonlinear_operator.N_size.front() < 4 ||
            parameters.nonlinear_operator.N_size.front()%2 != 0)
        {
            throw std::invalid_argument(
                "KS1D stability requires an even physical size >= 4");
        }
        if(ks1d_stability_model::backend_needs_device_init())
        {
            const int device =
                ks1d_stability_model::initialize_device(
                    command_line.device_selector);
            std::cout << "Using device " << device << '\n';
        }

        const std::size_t physical_size =
            parameters.nonlinear_operator.N_size.front();
        const std::size_t state_size = physical_size/2 - 1;
        real a_value = real(2);
        real b_value = real(4);
        if(!parameters.nonlinear_operator.
               problem_real_parameters_vector.empty())
        {
            a_value = parameters.nonlinear_operator.
                problem_real_parameters_vector.at(0);
        }
        if(parameters.nonlinear_operator.
               problem_real_parameters_vector.size() > 1)
        {
            b_value = parameters.nonlinear_operator.
                problem_real_parameters_vector.at(1);
        }

        using backend_type = typename vec_ops_real::backend_type;
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
            nonlinear_operators::kuramoto_sivashinskiy_1d<
                vec_ops_real,
                fft_backend_t,
                Blocks_x_>;
        using newton_operator_type =
            nonlinear_operators::linear_operator_KS_1D<
                vec_ops_real,
                nonlinear_operator_type>;
        using newton_preconditioner_type =
            nonlinear_operators::preconditioner_KS_1D<
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
                numerical_algos::lin_solvers::bicgstabl,
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
        nonlinear_operator_type nonlinear_operator(
            a_value,
            b_value,
            physical_size,
            real_space.get());
        newton_operator_type newton_operator(&nonlinear_operator);
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
            !command_line.quiet && config.inner_solver.verbose
                ? &linear_solver_log
                : nullptr);
        stability::analysis::configure_matrix_free_stability_reuse(
            eigensolver,
            config);

        if(!command_line.quiet)
        {
            std::cout
                << "Using KS1D stability backend: "
                << KS1D_BACKEND_NAME << '\n';
            parameters.plot_all();
            ks1d_stability_model::print_scan_configuration(config);
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
        if(!command_line.second_state_file.empty())
        {
            const auto result =
                driver.execute_single_transition(
                    command_line.state_file,
                    static_cast<real>(
                        command_line.state_parameter),
                    command_line.second_state_file,
                    static_cast<real>(
                        command_line.second_state_parameter),
                    command_line.confirm);
            std::cout
                << "Two-state stability transition result: status="
                << stability::analysis::
                       stability_transition_status_name(
                           result.status)
                << ", lambda=" << std::setprecision(17)
                << result.parameter
                << ", before=("
                << result.before_stability.unstable.real << ','
                << result.before_stability.unstable.complex_pairs
                << "), after=("
                << result.after_stability.unstable.real << ','
                << result.after_stability.unstable.complex_pairs
                << "), iterations=" << result.iterations;
            if(!result.diagnostic.empty())
                std::cout << ", diagnostic=" << result.diagnostic;
            std::cout << '\n';
            if(!result.succeeded())
                throw std::runtime_error(
                    "two-state stability transition replay failed");
        }
        else if(!command_line.state_file.empty())
        {
            const auto result = driver.execute_single_state(
                command_line.state_file,
                static_cast<real>(
                    command_line.state_parameter),
                command_line.confirm);
            std::cout
                << "Single-state stability result: status="
                << stability::analysis::
                       spectrum_classification_status_name(
                           result.classification_status)
                << ", unstable=("
                << result.unstable.real << ','
                << result.unstable.complex_pairs
                << "), attempts="
                << result.classification_attempts
                << ", diagnostic="
                << result.diagnostic << '\n';
            if(!result.succeeded())
                throw std::runtime_error(
                    "single-state stability replay failed");
        }
        else if(command_line.edit)
            driver.edit();
        else
            driver.execute();
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "KS1D stability failed: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
