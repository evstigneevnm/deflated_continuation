#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/log.h>

#include <common/gpu_file_operations.h>
#include <common/scfd_vector_operations.h>
#include <containers/bifurcation_diagram/diagram_archive.h>
#include <containers/bifurcation_diagram.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/curve_helper_container.h>
#include <containers/stability_diagram.h>
#include <deflation/solution_storage.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/system_operator.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>
#include <stability/eigensolvers/eigensolver_result.h>

namespace
{

template<class VectorOperations>
class fixed_stability_adapter
{
public:
    using vector_type = typename VectorOperations::vector_type;
    using real_type = typename VectorOperations::scalar_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit fixed_stability_adapter(
        std::size_t failure_after_successes =
            std::numeric_limits<std::size_t>::max())
        : failure_after_successes_(failure_after_successes)
    {
    }

    result_type execute(const vector_type&) const
    {
        const std::size_t call_index = call_count_++;
        result_type result;
        result.scans_requested = 1;
        if(call_index >= failure_after_successes_)
        {
            result.status =
                stability::eigensolvers::eigensolver_status::
                    inner_solver_failure;
            result.scans_succeeded = 0;
            result.coverage_complete = false;
            result.diagnostic =
                "injected stability classification failure";
            return result;
        }

        result.status =
            stability::eigensolvers::eigensolver_status::success;
        result.scans_succeeded = 1;
        result.coverage_complete = true;
        stability::eigensolvers::eigenpair_estimate<real_type>
            estimate;
        estimate.value = std::complex<real_type>(-1, 0);
        estimate.residual = real_type(0);
        estimate.relative_residual = real_type(0);
        estimate.converged = true;
        result.eigenpairs.push_back(estimate);
        return result;
    }

    std::size_t call_count() const
    {
        return call_count_;
    }

private:
    std::size_t failure_after_successes_;
    mutable std::size_t call_count_ = 0;
};

template<class NonlinearOperator>
class forwarding_linearization_provider
{
public:
    using vector_type = typename NonlinearOperator::T_vec;
    using scalar_type = typename NonlinearOperator::T;

    explicit forwarding_linearization_provider(
        NonlinearOperator* nonlinear_operator)
        : nonlinear_operator_(nonlinear_operator)
    {
    }

    void set_linearization_point(
        const vector_type& state,
        scalar_type parameter)
    {
        nonlinear_operator_->set_linearization_point(
            state,
            parameter);
    }

private:
    NonlinearOperator* nonlinear_operator_;
};

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if(!input)
        throw std::runtime_error(
            "failed to read test artifact: " + path.string());
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

bool contains_temporary_file(
    const std::filesystem::path& directory)
{
    if(!std::filesystem::exists(directory))
        return false;

    for(const auto& entry :
        std::filesystem::recursive_directory_iterator(directory))
    {
        if(
            entry.is_regular_file() &&
            entry.path().extension() == ".tmp")
            return true;
    }
    return false;
}

} // namespace

int main()
{
    using real_type = double;
    using vector_operations_type =
        scfd_vector_operations<
            scfd::backend::serial_cpu,
            real_type>;
    using file_operations_type =
        gpu_file_operations<vector_operations_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        numerical_algos::lin_solvers::default_monitor<
            vector_operations_type,
            log_type>;
    using nonlinear_operator_type =
        nonlinear_operators::circle<
            vector_operations_type,
            64>;
    using linear_operator_type =
        nonlinear_operators::linear_operator_circle<
            vector_operations_type,
            nonlinear_operator_type>;
    using preconditioner_type =
        nonlinear_operators::preconditioner_circle<
            vector_operations_type,
            nonlinear_operator_type,
            linear_operator_type>;
    using linear_solver_type =
        numerical_algos::lin_solvers::bicgstabl<
            linear_operator_type,
            preconditioner_type,
            vector_operations_type,
            monitor_type,
            log_type>;
    using convergence_type =
        nonlinear_operators::newton_method::convergence_strategy<
            vector_operations_type,
            nonlinear_operator_type,
            log_type>;
    using system_operator_type =
        nonlinear_operators::system_operator<
            vector_operations_type,
            nonlinear_operator_type,
            linear_operator_type,
            linear_solver_type>;
    using newton_type =
        numerical_algos::newton_method::newton_solver<
            vector_operations_type,
            nonlinear_operator_type,
            system_operator_type,
            convergence_type>;
    using solution_storage_type =
        deflation::solution_storage<vector_operations_type>;
    using curve_helper_type =
        container::curve_helper_container<vector_operations_type>;
    using bifurcation_curve_type =
        container::bifurcation_diagram_curve<
            vector_operations_type,
            file_operations_type,
            log_type,
            nonlinear_operator_type,
            newton_type,
            solution_storage_type,
            curve_helper_type>;
    using bifurcation_diagram_type =
        container::bifurcation_diagram<
            vector_operations_type,
            file_operations_type,
            log_type,
            nonlinear_operator_type,
            newton_type,
            solution_storage_type,
            bifurcation_curve_type,
            curve_helper_type>;
    using stability_diagram_type =
        container::stability_diagram<
            vector_operations_type,
            file_operations_type,
            log_type>;
    using parameters_type =
        main_classes::parameters<real_type>;
    using adapter_type =
        fixed_stability_adapter<vector_operations_type>;
    using provider_type =
        forwarding_linearization_provider<
            nonlinear_operator_type>;
    using driver_type =
        main_classes::stability_continuation<
            vector_operations_type,
            file_operations_type,
            log_type,
            monitor_type,
            nonlinear_operator_type,
            linear_operator_type,
            preconditioner_type,
            numerical_algos::lin_solvers::bicgstabl,
            nonlinear_operators::system_operator,
            parameters_type,
            adapter_type>;
    using custom_driver_type =
        main_classes::stability_continuation<
            vector_operations_type,
            file_operations_type,
            log_type,
            monitor_type,
            nonlinear_operator_type,
            linear_operator_type,
            preconditioner_type,
            numerical_algos::lin_solvers::bicgstabl,
            nonlinear_operators::system_operator,
            parameters_type,
            adapter_type,
            provider_type>;

    vector_operations_type vector_operations(1);
    file_operations_type file_operations(&vector_operations);
    nonlinear_operator_type nonlinear_operator(
        real_type(1),
        1,
        &vector_operations);
    parameters_type parameters;
    parameters.set_default();
    const auto unique_id =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
    const std::filesystem::path fixture_directory =
        std::filesystem::temp_directory_path() /
        (
            "deflated_continuation_stability_interface_" +
            std::to_string(unique_id));
    std::error_code filesystem_error;
    std::filesystem::remove_all(
        fixture_directory,
        filesystem_error);
    filesystem_error.clear();
    std::filesystem::create_directories(
        fixture_directory,
        filesystem_error);
    if(filesystem_error)
    {
        std::cerr
            << "failed to create stability interface fixture: "
            << filesystem_error.message() << '\n';
        return EXIT_FAILURE;
    }
    parameters.path_to_project = fixture_directory.string();
    parameters.deflation_continuation.skip_files = 1;
    parameters.stability_continuation.
        spectrum_classification_retries = 2;
    log_type log;
    log_type linear_solver_log;
    log.set_verbosity(0);
    linear_solver_log.set_verbosity(0);

    try
    {
        adapter_type construction_adapter;
        provider_type provider(&nonlinear_operator);
        {
            driver_type driver(
                &vector_operations,
                &file_operations,
                &log,
                &linear_solver_log,
                &nonlinear_operator,
                &parameters,
                &construction_adapter);
            driver.set_parameters();
        }
        {
            custom_driver_type driver(
                &vector_operations,
                &file_operations,
                &log,
                &linear_solver_log,
                &nonlinear_operator,
                &parameters,
                &construction_adapter,
                &provider);
            driver.set_parameters();
        }

        auto* vector_operations_ptr = &vector_operations;
        auto* nonlinear_operator_ptr = &nonlinear_operator;
        auto* log_ptr = &log;
        linear_operator_type fixture_linear_operator(
            nonlinear_operator_ptr);
        auto* fixture_linear_operator_ptr =
            &fixture_linear_operator;
        preconditioner_type fixture_preconditioner(
            nonlinear_operator_ptr);
        linear_solver_type fixture_linear_solver(
            &vector_operations,
            &linear_solver_log);
        fixture_linear_solver.set_preconditioner(
            &fixture_preconditioner);
        auto* fixture_linear_solver_ptr =
            &fixture_linear_solver;
        convergence_type fixture_convergence(
            vector_operations_ptr,
            log_ptr);
        system_operator_type fixture_system_operator(
            vector_operations_ptr,
            fixture_linear_operator_ptr,
            fixture_linear_solver_ptr);
        newton_type fixture_newton(
            &vector_operations,
            &fixture_system_operator,
            &fixture_convergence);

        bifurcation_diagram_type source_diagram(
            &vector_operations,
            &file_operations,
            &log,
            &nonlinear_operator,
            &fixture_newton,
            fixture_directory.string(),
            1);

        typename vector_operations_type::vector_type state;
        vector_operations.init_vector(state);
        vector_operations.start_use_vector(state);

        source_diagram.init_new_curve();
        vector_operations.assign_scalar(real_type(0.6), state);
        source_diagram.get_current_ref()->add(
            real_type(-0.8),
            state,
            true);
        vector_operations.assign_scalar(real_type(0.8), state);
        source_diagram.get_current_ref()->add(
            real_type(-0.6),
            state,
            true);
        source_diagram.close_curve();

        source_diagram.init_new_curve();
        vector_operations.assign_scalar(
            -std::sqrt(real_type(0.96)),
            state);
        source_diagram.get_current_ref()->add(
            real_type(0.2),
            state,
            true);
        vector_operations.assign_scalar(
            -std::sqrt(real_type(0.84)),
            state);
        source_diagram.get_current_ref()->add(
            real_type(0.4),
            state,
            true);
        source_diagram.close_curve();

        vector_operations.stop_use_vector(state);
        vector_operations.free_vector(state);

        const std::filesystem::path bifurcation_archive =
            fixture_directory /
            parameters.bifurcaiton_diagram_file_name;
        const auto bifurcation_save =
            container::save_diagram_archive(
                bifurcation_archive.string(),
                source_diagram);
        require(
            bifurcation_save.succeeded(),
            "two-curve bifurcation fixture was not saved");

        adapter_type failing_adapter(2);
        bool injected_failure_observed = false;
        {
            driver_type driver(
                &vector_operations,
                &file_operations,
                &log,
                &linear_solver_log,
                &nonlinear_operator,
                &parameters,
                &failing_adapter);
            driver.set_parameters();
            try
            {
                driver.execute();
            }
            catch(const std::runtime_error& error)
            {
                injected_failure_observed =
                    std::string(error.what()).find(
                        "failed to traverse bifurcation curve 1") !=
                    std::string::npos;
            }
        }
        require(
            injected_failure_observed,
            "injected curve-1 classification failure was not propagated");
        require(
            failing_adapter.call_count() == 3,
            "interrupted traversal did not fail on curve 1's first point");

        const std::filesystem::path stability_archive =
            fixture_directory /
            parameters.stability_diagram_file_name;
        stability_diagram_type interrupted_diagram(
            &vector_operations,
            &file_operations,
            &log,
            fixture_directory.string());
        const auto interrupted_load =
            container::load_diagram_archive(
                stability_archive.string(),
                interrupted_diagram);
        require(
            interrupted_load.succeeded(),
            "interrupted stability archive was not readable");
        require(
            interrupted_diagram.curve_count() == 1 &&
                interrupted_diagram.current_curve() == 1,
            "interrupted run did not commit exactly curve 0");

        const std::filesystem::path curve0_legacy =
            fixture_directory / "0" /
            "debug_curve_stability.dat";
        const std::filesystem::path curve0_plot =
            fixture_directory / "0" /
            "debug_curve_stability_plot.dat";
        const std::string committed_curve0_legacy =
            read_file(curve0_legacy);
        const std::string committed_curve0_plot =
            read_file(curve0_plot);
        require(
            !std::filesystem::exists(
                fixture_directory / "1" /
                "debug_curve_stability.dat") &&
                !std::filesystem::exists(
                    fixture_directory / "1" /
                    "debug_curve_stability_plot.dat"),
            "failed curve 1 left committed stability sidecars");
        require(
            !contains_temporary_file(fixture_directory),
            "interrupted traversal left temporary files");

        adapter_type restart_adapter;
        {
            driver_type driver(
                &vector_operations,
                &file_operations,
                &log,
                &linear_solver_log,
                &nonlinear_operator,
                &parameters,
                &restart_adapter);
            driver.set_parameters();
            driver.execute();
        }
        require(
            restart_adapter.call_count() == 2,
            "restart did not resume at curve 1");

        stability_diagram_type completed_diagram(
            &vector_operations,
            &file_operations,
            &log,
            fixture_directory.string());
        const auto completed_load =
            container::load_diagram_archive(
                stability_archive.string(),
                completed_diagram);
        require(
            completed_load.succeeded() &&
                completed_diagram.curve_count() == 2,
            "restart did not commit curve 1");
        require(
            read_file(curve0_legacy) ==
                committed_curve0_legacy &&
                read_file(curve0_plot) ==
                committed_curve0_plot,
            "restart modified the previously committed curve 0");
        require(
            std::filesystem::is_regular_file(
                fixture_directory / "1" /
                "debug_curve_stability.dat") &&
                std::filesystem::is_regular_file(
                    fixture_directory / "1" /
                    "debug_curve_stability_plot.dat") &&
                !contains_temporary_file(fixture_directory),
            "completed restart did not commit clean curve-1 sidecars");

        const std::string completed_archive =
            read_file(stability_archive);
        const std::string completed_curve1_legacy =
            read_file(
                fixture_directory / "1" /
                "debug_curve_stability.dat");
        const std::string completed_curve1_plot =
            read_file(
                fixture_directory / "1" /
                "debug_curve_stability_plot.dat");
        adapter_type no_op_adapter;
        {
            driver_type driver(
                &vector_operations,
                &file_operations,
                &log,
                &linear_solver_log,
                &nonlinear_operator,
                &parameters,
                &no_op_adapter);
            driver.set_parameters();
            driver.execute();
        }
        require(
            no_op_adapter.call_count() == 0,
            "completed stability restart reprocessed existing curves");
        require(
            read_file(stability_archive) == completed_archive &&
                read_file(
                    fixture_directory / "1" /
                    "debug_curve_stability.dat") ==
                    completed_curve1_legacy &&
                read_file(
                    fixture_directory / "1" /
                    "debug_curve_stability_plot.dat") ==
                    completed_curve1_plot,
            "no-op restart changed committed stability data");
    }
    catch(const std::exception& error)
    {
        std::filesystem::remove_all(
            fixture_directory,
            filesystem_error);
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::filesystem::remove_all(
        fixture_directory,
        filesystem_error);

    std::cout
        << "stability_continuation interface/restart PASSED\n";
    return EXIT_SUCCESS;
}
