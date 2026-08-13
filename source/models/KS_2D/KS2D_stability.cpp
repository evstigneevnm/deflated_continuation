#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <common/gpu_file_operations.h>
#include <common/scfd_backend_ext/complex.h>
#include <deflation/symmetry_solution_storage.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>

#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <symmetry/finite_action_registry.h>
#include <symmetry/finite_group_manifest.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/analysis/dimension_guarded_eigensolver.h>
#include <stability/eigensolvers/host_dense_operator_eigensolver.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_affine_inverse_provider.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include "KS2D_backend_typedefs.h"
#include "KS2D_stability_cli.h"

int main(int argc, char** argv)
{
    try
    {
        const auto command_line = ks2d_stability_model::parse_command_line(
            argc,
            argv,
            "json_project_files/KS2D_test_sym.json"
        );
        using parameters_type = main_classes::parameters<real>;
        parameters_type parameters = main_classes::read_parameters_json<real>(command_line.config_file);
        const auto& config = parameters.stability_continuation.matrix_free_eigensolver;
        stability::analysis::validate_matrix_free_stability_config(config);
        if(parameters.nonlinear_operator.N_size.size() != 2)
        {
            throw std::invalid_argument("KS2D stability requires [Nx, Ny]");
        }
        const std::size_t nx = parameters.nonlinear_operator.N_size.at(0);
        const std::size_t ny = parameters.nonlinear_operator.N_size.at(1);
        if(nx < 4 || ny < 4 || nx%2 != 0 || ny%2 != 0)
        {
            throw std::invalid_argument("KS2D stability requires even grid dimensions >= 4");
        }
        if(ks2d_stability_model::backend_needs_device_init())
        {
            std::cout << "Using device "
                      << ks2d_stability_model::initialize_device(command_line.device_selector)
                      << '\n';
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

        using backend_type = typename vec_ops_real::backend_type;
        using complex_scalar_type = common::scfd_backend_ext::complex_t<backend_type, real>;
        using complex_space_type = scfd_vector_operations<backend_type, complex_scalar_type>;
        using file_operations_type = gpu_file_operations<vec_ops_real>;
        using nonlinear_operator_type = nonlinear_operators::kuramoto_sivashinskiy_2d<
            vec_ops_real,
            fft_backend_t
        >;
        using newton_operator_type = nonlinear_operators::linear_operator_KS_2D<
            vec_ops_real,
            nonlinear_operator_type
        >;
        using newton_preconditioner_type = nonlinear_operators::preconditioner_KS_2D<
            vec_ops_real,
            nonlinear_operator_type,
            newton_operator_type
        >;
        using base_provider_type = stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<vec_ops_real, nonlinear_operator_type>;
        using stability_operator_type = stability::eigensolvers::transformations::
            scaled_real_operator<vec_ops_real, newton_operator_type>;
        using stability_provider_type = stability::eigensolvers::transformations::
            scaled_real_affine_inverse_provider<base_provider_type>;
        using factorization_types = stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                vec_ops_real,
                complex_space_type,
                stability_operator_type,
                stability_provider_type
            >;
        using factor_operator_type = typename factorization_types::factor_operator_type;
        using factor_preconditioner_type = typename factorization_types::preconditioner_type;
        using inner_monitor_type = nmfd::solvers::monitor_krylov<complex_space_type, log_t>;
        using inner_solver_type = nmfd::solvers::gmres<
            complex_space_type,
            inner_monitor_type,
            log_t,
            factor_operator_type,
            factor_preconditioner_type
        >;
        using dense_lapack_type = nmfd::operations::linalg::host_small_dense_lapack<real>;
        using matrix_free_eigensolver_type = stability::analysis::matrix_free_stability_scan<
            factorization_types,
            inner_solver_type,
            dense_lapack_type
        >;
        using small_system_eigensolver_type = stability::eigensolvers::host_dense_operator_eigensolver<
            vec_ops_real,
            stability_operator_type,
            dense_lapack_type
        >;
        using eigensolver_type = stability::analysis::dimension_guarded_eigensolver<
            matrix_free_eigensolver_type,
            small_system_eigensolver_type
        >;
        using newton_monitor_type = numerical_algos::lin_solvers::default_monitor<vec_ops_real, log_t>;
        using finite_actions_type = symmetry::finite_action_registry<vec_ops_real>;
        using identity_adapter_type = deflation::identity_symmetry_adapter<vec_ops_real>;
        using residual_translation_aligner_type =
            symmetry::fourier::residual_translation_orbit_aligner_2d<
                vec_ops_real>;
        using quotient_adapter_type = symmetry::finite_quotient_adapter<
            vec_ops_real,
            identity_adapter_type,
            residual_translation_aligner_type>;
        using driver_type = main_classes::stability_continuation<
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
            eigensolver_type
        >;

        auto real_space = std::make_shared<vec_ops_real>(state_size);
        auto complex_space = std::make_shared<complex_space_type>(state_size);
        if(parameters.use_high_precision_reduction)
        {
            real_space->use_high_precision();
        }
        file_operations_type file_operations(real_space.get());
        nonlinear_operator_type nonlinear_operator(a, b, nx, ny, real_space.get());
        newton_operator_type newton_operator(&nonlinear_operator);
        stability_operator_type stability_operator(*real_space, newton_operator, config.linearization_scale);
        auto base_provider = std::make_shared<base_provider_type>(*real_space, nonlinear_operator);
        auto stability_provider = std::make_shared<stability_provider_type>(
            base_provider,
            config.linearization_scale
        );
        log_t log;
        log_t linear_solver_log;
        log.set_verbosity(command_line.quiet ? 0 : 1);
        linear_solver_log.set_verbosity(command_line.quiet ? 0 : 1);
        finite_actions_type finite_actions(real_space.get());
        const auto finite_symmetry_group =
            nonlinear_operator.finite_symmetry_group();
        const auto finite_action_workspace =
            nonlinear_operator.configure_finite_symmetry_actions(
                finite_actions,
                finite_symmetry_group);
        const int axis_swap_action =
            finite_actions.find("axis_swap");
        if(axis_swap_action < 0)
        {
            throw std::runtime_error(
                "KS2D stability requires the axis-swap action");
        }
        const bool uses_preferred_small_system =
            config.small_system.enabled &&
            config.small_system.prefer &&
            state_size <= config.small_system.maximum_dimension;
        if(
            !uses_preferred_small_system &&
            (config.aggregation.probe_count < 4 ||
             config.aggregation.probe_count % 2 != 0))
        {
            throw std::invalid_argument(
                "KS2D stability requires an even multiplicity probe "
                "count of at least four for paired axis-swap probes");
        }
        identity_adapter_type identity_adapter(real_space.get());
        residual_translation_aligner_type residual_translation_aligner(
            real_space.get(),
            nonlinear_operator.state_modes());
        quotient_adapter_type quotient_adapter(
            real_space.get(),
            &identity_adapter,
            &finite_actions,
            &residual_translation_aligner);
        const auto symmetry_manifest =
            symmetry::load_finite_group_manifest(
                std::filesystem::path(parameters.path_to_project)/
                "symmetry_group.json");
        if(!symmetry_manifest.succeeded())
        {
            throw std::runtime_error(
                "KS2D stability requires a validated symmetry_group.json: " +
                symmetry_manifest.message);
        }
        if(!symmetry::finite_group_manifest_matches(
               symmetry_manifest.manifest,
               quotient_adapter.symmetry_definition_fingerprint(),
               quotient_adapter.symmetry_action_names()))
        {
            throw std::runtime_error(
                "KS2D stability symmetry definition differs from the "
                "bifurcation archive; rebuild or audit the bifurcation "
                "diagram first");
        }
        matrix_free_eigensolver_type matrix_free_eigensolver(
            real_space,
            complex_space,
            stability_operator,
            stability_provider,
            stability::analysis::make_matrix_free_stability_scans<matrix_free_eigensolver_type>(config),
            stability::analysis::make_matrix_free_inner_solver_parameters<
                typename matrix_free_eigensolver_type::inner_parameters_type
            >(config),
            stability::analysis::make_spectrum_scan_aggregation_options(config),
            !command_line.quiet && config.inner_solver.verbose ? &linear_solver_log : nullptr
        );
        matrix_free_eigensolver.set_recycling_options(
            stability::analysis::
                make_recycled_ritz_subspace_options(config));
        auto paired_probe_base = std::make_shared<
            stability::analysis::detail::vector_workspace<
                vec_ops_real>>(real_space.get());
        matrix_free_eigensolver.set_probe_generator(
            [real_space,
             &nonlinear_operator,
             &finite_actions,
             paired_probe_base,
             axis_swap_action](
                std::size_t probe_index,
                const typename vec_ops_real::vector_type& initial_probe,
                typename vec_ops_real::vector_type& probe)
            {
                if(probe_index == 1)
                {
                    finite_actions.apply(
                        static_cast<std::size_t>(axis_swap_action),
                        initial_probe,
                        probe);
                    return;
                }
                if(probe_index % 2 == 0)
                {
                    nonlinear_operator.randomize_stability_vector(
                        probe);
                    real_space->assign(
                        probe,
                        paired_probe_base->get());
                    return;
                }
                finite_actions.apply(
                    static_cast<std::size_t>(axis_swap_action),
                    paired_probe_base->get(),
                    probe);
            });
        dense_lapack_type small_system_lapack;
        typename small_system_eigensolver_type::options_type small_system_options;
        small_system_options.absolute_residual_tolerance =
            config.small_system.absolute_residual_tolerance;
        small_system_options.relative_residual_tolerance =
            config.small_system.relative_residual_tolerance;
        small_system_eigensolver_type small_system_eigensolver(
            *real_space,
            stability_operator,
            small_system_lapack,
            small_system_options
        );
        eigensolver_type eigensolver(
            matrix_free_eigensolver,
            small_system_eigensolver,
            state_size,
            config.small_system.enabled
                ? config.small_system.maximum_dimension
                : std::size_t(0),
            config.small_system.prefer
        );

        if(!command_line.quiet)
        {
            std::cout << "Using KS2D stability backend: " << KS2D_BACKEND_NAME
                      << ", state size=" << state_size
                      << ", finite symmetry order="
                      << finite_symmetry_group.size()
                      << ", fingerprint="
                      << finite_symmetry_group.fingerprint() << '\n';
            parameters.plot_all();
        }
        driver_type driver(
            real_space.get(),
            &file_operations,
            &log,
            &linear_solver_log,
            &nonlinear_operator,
            &parameters,
            &eigensolver
        );
        (void)finite_action_workspace;
        driver.set_transition_state_aligner(&quotient_adapter);
        driver.set_parameters();
        if(!command_line.second_state_file.empty())
        {
            const auto result = driver.execute_single_transition(
                command_line.state_file,
                static_cast<real>(command_line.state_parameter),
                command_line.second_state_file,
                static_cast<real>(command_line.second_state_parameter),
                command_line.confirm);
            std::cout
                << "Two-state stability transition result: status="
                << stability::analysis::stability_transition_status_name(
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
                static_cast<real>(command_line.state_parameter),
                command_line.confirm);
            std::cout
                << "Single-state stability result: status="
                << stability::analysis::spectrum_classification_status_name(
                       result.classification_status)
                << ", unstable=("
                << result.unstable.real << ','
                << result.unstable.complex_pairs
                << "), attempts="
                << result.classification_attempts
                << ", diagnostic=" << result.diagnostic << '\n';
            if(!result.succeeded())
                throw std::runtime_error(
                    "single-state stability replay failed");
        }
        else if(command_line.edit)
        {
            driver.edit();
        }
        else
        {
            driver.execute();
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << "KS2D stability failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
