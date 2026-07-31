#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <common/scfd_backend_ext/complex.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <nonlinear_operators/projected_system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>

#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>

#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/analysis/dimension_guarded_eigensolver.h>
#include <stability/eigensolvers/host_dense_operator_eigensolver.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include <symmetry/fourier/real_packed_fourier_slice_1d_policy_json.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>
#include <symmetry/linearization/projected_affine_inverse_provider.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_linearization_provider.h>
#include <symmetry/linearization/projected_preconditioner.h>
#include <symmetry/linearization/projected_stability_linear_operator.h>
#include <symmetry/linearization/projected_stability_gauge.h>

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
                "json_project_files/KS1D_test_full.json");
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
                "full KS1D stability requires an even physical "
                "size >= 4");
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
        const std::size_t positive_modes =
            physical_size/2 - 1;
        const std::size_t state_size = 2*positive_modes;
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
            nonlinear_operators::
                kuramoto_sivashinskiy_1d_full<
                    vec_ops_real,
                    fft_backend_t,
                    Blocks_x_>;
        using symmetry_adapter_type =
            symmetry::fourier::
                real_packed_fourier_slice_1d_adapter<
                    vec_ops_real>;
        using finite_actions_type =
            symmetry::finite_action_registry<vec_ops_real>;
        using quotient_adapter_type =
            symmetry::finite_quotient_adapter<
                vec_ops_real,
                symmetry_adapter_type>;
        using newton_operator_type =
            symmetry::linearization::
                projected_linear_operator<
                    vec_ops_real,
                    nonlinear_operator_type>;
        using newton_preconditioner_type =
            symmetry::linearization::
                projected_preconditioner<
                    vec_ops_real,
                    nonlinear_operator_type,
                    newton_operator_type>;
        using base_provider_type =
            stability::eigensolvers::transformations::
                nonlinear_operator_real_affine_inverse_provider<
                    vec_ops_real,
                    nonlinear_operator_type>;
        using quotient_provider_type =
            symmetry::linearization::
                projected_affine_inverse_provider<
                    vec_ops_real,
                    nonlinear_operator_type,
                    base_provider_type>;
        using quotient_operator_type =
            symmetry::linearization::
                projected_stability_linear_operator<
                    vec_ops_real,
                    nonlinear_operator_type>;
        using stability_operator_type =
            stability::eigensolvers::transformations::
                scaled_real_operator<
                    vec_ops_real,
                    quotient_operator_type>;
        using stability_provider_type =
            stability::eigensolvers::transformations::
                scaled_real_affine_inverse_provider<
                    quotient_provider_type>;
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
        using matrix_free_eigensolver_type =
            stability::analysis::matrix_free_stability_scan<
                factorization_types,
                inner_solver_type,
                dense_lapack_type>;
        using small_system_eigensolver_type =
            stability::eigensolvers::
                host_dense_operator_eigensolver<
                    vec_ops_real,
                    stability_operator_type,
                    dense_lapack_type>;
        using eigensolver_type =
            stability::analysis::dimension_guarded_eigensolver<
                matrix_free_eigensolver_type,
                small_system_eigensolver_type>;
        using newton_monitor_type =
            numerical_algos::lin_solvers::default_monitor<
                vec_ops_real,
                log_t>;
        using linearization_provider_type =
            symmetry::linearization::
                projected_linearization_provider<
                    vec_ops_real,
                    nonlinear_operator_type>;
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
                nonlinear_operators::projected_system_operator,
                parameters_type,
                eigensolver_type,
                linearization_provider_type>;

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
        const auto config_json =
            main_classes::read_json(command_line.config_file);
        const auto transition_alignment_policy =
            symmetry::fourier::
                read_real_packed_fourier_slice_1d_policy<real>(
                    config_json,
                    "symmetry_stabilizer",
                    nonlinear_operator_type::
                        default_continuation_symmetry_policy());
        const auto continuation_symmetry_policy =
            symmetry::fourier::
                read_real_packed_fourier_slice_1d_policy<real>(
                    config_json,
                    "continuation_symmetry_stabilizer",
                    nonlinear_operator_type::
                        default_continuation_symmetry_policy());
        if(
            continuation_symmetry_policy.stabilizer !=
            symmetry::fourier::
                real_packed_fourier_1d_stabilizer_policy::
                    single_mode)
        {
            throw std::invalid_argument(
                "full KS1D stability currently requires the tested "
                "single-mode frozen chart");
        }
        nonlinear_operator.configure_continuation_symmetry(
            continuation_symmetry_policy);
        symmetry_adapter_type transition_symmetry_adapter(
            real_space.get(),
            positive_modes);
        transition_symmetry_adapter.configure(
            transition_alignment_policy);
        finite_actions_type transition_finite_actions(
            real_space.get());
        nonlinear_operator.configure_finite_symmetry_actions(
            transition_finite_actions);
        quotient_adapter_type transition_quotient_adapter(
            real_space.get(),
            &transition_symmetry_adapter,
            &transition_finite_actions);

        const auto stability_gauge =
            symmetry::linearization::
                make_projected_stability_gauge<real>(
                    config.linearization_scale,
                    parameters.stability_continuation.
                        linear_operator_stable_eigenvalues_left_halfplane);
        quotient_operator_type quotient_operator(
            &nonlinear_operator,
            stability_gauge.operator_completion);
        stability_operator_type stability_operator(
            *real_space,
            quotient_operator,
            config.linearization_scale);
        auto base_provider =
            std::make_shared<base_provider_type>(
                *real_space,
                nonlinear_operator);
        auto quotient_provider =
            std::make_shared<quotient_provider_type>(
            *real_space,
            nonlinear_operator,
            base_provider,
            stability_gauge.operator_completion);
        auto stability_provider =
            std::make_shared<stability_provider_type>(
                quotient_provider,
                config.linearization_scale);
        linearization_provider_type linearization_provider(
            nonlinear_operator);

        log_t log;
        log_t linear_solver_log;
        log.set_verbosity(command_line.quiet ? 0 : 1);
        linear_solver_log.set_verbosity(
            command_line.quiet ? 0 : 1);
        matrix_free_eigensolver_type matrix_free_eigensolver(
            real_space,
            complex_space,
            stability_operator,
            stability_provider,
            stability::analysis::
                make_matrix_free_stability_scans<
                    matrix_free_eigensolver_type>(config),
            stability::analysis::
                make_matrix_free_inner_solver_parameters<
                    typename matrix_free_eigensolver_type::
                        inner_parameters_type>(config),
            stability::analysis::
                make_spectrum_scan_aggregation_options(config),
            !command_line.quiet && config.inner_solver.verbose
                ? &linear_solver_log
                : nullptr);
        dense_lapack_type small_system_lapack;
        typename small_system_eigensolver_type::options_type
            small_system_options;
        small_system_options.absolute_residual_tolerance =
            config.small_system.absolute_residual_tolerance;
        small_system_options.relative_residual_tolerance =
            config.small_system.relative_residual_tolerance;
        small_system_eigensolver_type small_system_eigensolver(
            *real_space,
            stability_operator,
            small_system_lapack,
            small_system_options);
        eigensolver_type eigensolver(
            matrix_free_eigensolver,
            small_system_eigensolver,
            state_size,
            config.small_system.enabled
                ? config.small_system.maximum_dimension
                : std::size_t(0),
            config.small_system.prefer);

        if(!command_line.quiet)
        {
            std::cout
                << "Using full KS1D stability backend: "
                << KS1D_BACKEND_NAME << '\n'
                << "Projected stability gauge: operator completion="
                << stability_gauge.operator_completion
                << ", scaled eigenvalue="
                << stability_gauge.scaled_eigenvalue << '\n';
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
            &eigensolver,
            &linearization_provider);
        driver.set_transition_state_aligner(
            &transition_quotient_adapter);
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
            if(!command_line.quiet)
            {
                nonlinear_operator.log_projection_diagnostics(
                    &log,
                    "two-state stability replay final chart");
            }
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
            if(!command_line.quiet)
            {
                nonlinear_operator.log_projection_diagnostics(
                    &log,
                    "single-state stability replay chart");
            }
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
            << "full KS1D stability failed: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
