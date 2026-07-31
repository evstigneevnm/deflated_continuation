#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/utils/log.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/io/matrix_market.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/krylov_schur.h>
#include <stability/eigensolvers/rotated_projected_spectrum_recovery.h>
#include <stability/eigensolvers/transformations/complex_shift_block_operator.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/inverse_composed_operator.h>
#include <stability/eigensolvers/transformations/stability_polynomial_operator.h>
#include <stability/eigensolvers/transformations/tracked_linear_solver.h>

#include "common/host_csr_operator.h"

namespace
{

using complex_type = std::complex<double>;

std::string matrix_path(int argc, char** argv)
{
    if(argc > 1)
        return argv[1];
    return
        "data/external/suitesparse/Rommes/S80PI_n1/S80PI_n1.mtx";
}

double rotation_angle(int argc, char** argv)
{
    if(argc > 2)
        return std::stod(argv[2]);
    return std::acos(-1.0) / 6.5;
}

std::size_t maximum_restarts(int argc, char** argv)
{
    if(argc > 3)
        return static_cast<std::size_t>(std::stoul(argv[3]));
    return 25;
}

unsigned inner_basis_size(int argc, char** argv)
{
    if(argc > 4)
        return static_cast<unsigned>(std::stoul(argv[4]));
    return 30;
}

unsigned inner_maximum_iterations(int argc, char** argv)
{
    if(argc > 5)
        return static_cast<unsigned>(std::stoul(argv[5]));
    return 300;
}

double inner_relative_tolerance(int argc, char** argv)
{
    if(argc > 6)
        return std::stod(argv[6]);
    return 1.0e-11;
}

std::string stability_polynomial_name(int argc, char** argv)
{
    if(argc > 7)
        return argv[7];
    return "euler";
}

std::size_t mapping_repetitions(int argc, char** argv)
{
    if(argc > 8)
        return static_cast<std::size_t>(std::stoul(argv[8]));
    return 3;
}

std::vector<double> stability_polynomial_coefficients(
    const std::string& name)
{
    if(name == "euler")
    {
        return stability::eigensolvers::transformations::
            explicit_euler_stability_polynomial<double>();
    }
    if(name == "rk4")
    {
        return stability::eigensolvers::transformations::
            classical_rk4_stability_polynomial<double>();
    }
    throw std::invalid_argument(
        "unsupported stability polynomial: " + name);
}

std::vector<double> initial_vector(std::size_t dimension)
{
    std::vector<double> result(2 * dimension);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        result[index] =
            1.0 +
            0.013 * static_cast<double>(index % 29) -
            0.007 * static_cast<double>((index * index) % 31);
        result[dimension + index] =
            -0.4 +
            0.011 * static_cast<double>(index % 37) +
            0.005 * static_cast<double>((index * index) % 23);
    }
    return result;
}

bool near_report_target(const complex_type& value)
{
    const double absolute_imaginary = std::abs(value.imag());
    const bool correct_frequency =
        std::abs(absolute_imaginary - 1.87) < 0.04 ||
        std::abs(absolute_imaginary - 1.68) < 0.04;
    return std::abs(value.real()) < 2.0e-3 && correct_frequency;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        using component_space_type =
            scfd_vector_operations<scfd::backend::omp, double>;
        using product_space_type =
            nmfd::operations::two_block_vector_space<
                component_space_type>;
        using product_vector_type =
            typename product_space_type::vector_type;
        using matrix_operator_type =
            stability::tests::host_csr_operator<
                component_space_type>;
        using polynomial_type =
            stability::eigensolvers::transformations::
                stability_polynomial_operator<
                    component_space_type,
                    matrix_operator_type>;
        using denominator_type =
            stability::eigensolvers::transformations::
                complex_shift_block_operator<
                    product_space_type,
                    polynomial_type>;
        using identity_type =
            stability::eigensolvers::transformations::identity_operator<
                product_space_type>;
        using log_type = scfd::utils::log_std;
        using monitor_type =
            nmfd::solvers::monitor_krylov<
                product_space_type,
                log_type>;
        using gmres_type =
            nmfd::solvers::gmres<
                product_space_type,
                monitor_type,
                log_type,
                denominator_type>;
        using tracked_solver_type =
            stability::eigensolvers::transformations::
                tracked_linear_solver<
                    gmres_type,
                    denominator_type>;
        using transformed_operator_type =
            stability::eigensolvers::transformations::
                inverse_composed_operator<
                    product_space_type,
                    identity_type,
                    tracked_solver_type>;
        using lapack_type =
            nmfd::operations::linalg::host_small_dense_lapack<double>;
        using eigensolver_type =
            stability::eigensolvers::krylov_schur<
                product_space_type,
                lapack_type>;

        const auto coordinate =
            nmfd::operations::io::read_matrix_market_file<double>(
                matrix_path(argc, argv));
        if(
            coordinate.metadata.rows != 4028 ||
            coordinate.metadata.columns != 4028)
        {
            throw std::runtime_error(
                "S80PI_n1 has unexpected dimensions");
        }
        const auto matrix = coordinate.to_host_csr();
        const std::size_t dimension =
            static_cast<std::size_t>(matrix.rows());

        auto component_space =
            std::make_shared<component_space_type>(dimension);
        auto product_space =
            std::make_shared<product_space_type>(
                *component_space,
                *component_space);
        matrix_operator_type original(
            *component_space,
            matrix,
            nmfd::operations::sparse::host_csr_execution::openmp);

        constexpr double total_time = 0.3;
        const std::size_t power = mapping_repetitions(argc, argv);
        if(power == 0)
            throw std::invalid_argument(
                "mapping repetitions must be positive");
        const double step =
            total_time / static_cast<double>(power);
        constexpr double shift_delta = 0.01;
        constexpr double shift_radius = 1.0 + shift_delta;
        const double phi = rotation_angle(argc, argv);
        const complex_type shift =
            std::polar(shift_radius, phi);
        const std::string polynomial_name =
            stability_polynomial_name(argc, argv);
        const auto polynomial_coefficients =
            stability_polynomial_coefficients(polynomial_name);

        polynomial_type polynomial(
            *component_space,
            original,
            step,
            power,
            polynomial_coefficients);
        denominator_type denominator(
            *product_space,
            polynomial,
            shift.real(),
            shift.imag());
        identity_type identity(*product_space);

        typename gmres_type::params inner_parameters;
        inner_parameters.basis_size = inner_basis_size(argc, argv);
        inner_parameters.batch_size =
            std::min(10u, inner_parameters.basis_size);
        inner_parameters.orthogonalization = "mgs";
        inner_parameters.reorthogonalization_policy = "dgks";
        inner_parameters.max_orthogonalization_passes = 2;
        inner_parameters.monitor.rel_tol =
            inner_relative_tolerance(argc, argv);
        inner_parameters.monitor.abs_tol = 1.0e-13;
        inner_parameters.monitor.max_iters_num =
            inner_maximum_iterations(argc, argv);
        inner_parameters.monitor.divide_out_norms_by_rel_base = false;
        gmres_type inner_solver(
            product_space,
            nullptr,
            inner_parameters);
        tracked_solver_type tracked_solver(
            inner_solver,
            denominator);
        transformed_operator_type transformed(
            *product_space,
            identity,
            tracked_solver);

        lapack_type lapack;
        eigensolver_type eigensolver(*product_space, lapack);
        typename eigensolver_type::options_type options;
        options.desired_eigenvalues = 4;
        options.krylov_dimension = 14;
        options.restart_dimension = 4;
        options.max_restarts = maximum_restarts(argc, argv);
        options.absolute_tolerance = 1.0e-9;
        options.relative_tolerance = 1.0e-8;
        options.preserve_conjugate_pairs = true;
        options.target.kind =
            stability::eigensolvers::spectrum_target::largest_magnitude;
        options.orthogonalization.method =
            nmfd::solvers::krylov::orthogonalization_method::
                modified_gram_schmidt;
        options.orthogonalization.reorthogonalization =
            nmfd::solvers::krylov::reorthogonalization_policy::dgks;
        options.orthogonalization.max_passes = 2;

        product_vector_type initial;
        product_space->init_vector(initial);
        product_space->start_use_vector(initial);
        const auto initial_host = initial_vector(dimension);
        product_space->set(
            initial_host.data(),
            initial,
            initial_host.size());

        stability::eigensolvers::ritz_vector_storage<
            product_space_type> transformed_vectors(
                *product_space,
                5);
        const auto transformed_result = eigensolver.execute(
            [&transformed](
                const product_vector_type& source,
                product_vector_type& destination)
            {
                return transformed.apply(source, destination);
            },
            initial,
            options,
            &transformed_vectors);

        std::cout << std::setprecision(17)
                  << "case=S80PI_n1"
                  << " mode=rotated_inexact_exponential"
                  << " total_time=" << total_time
                  << " power=" << power
                  << " step=" << step
                  << " shift_delta=" << shift_delta
                  << " shift_radius=" << shift_radius
                  << " phi=" << phi
                  << " stability_polynomial=" << polynomial_name
                  << " polynomial_degree=" << polynomial.degree()
                  << " inner_relative_tolerance="
                  << inner_parameters.monitor.rel_tol
                  << " status="
                  << stability::eigensolvers::eigensolver_status_name(
                         transformed_result.status)
                  << " iterations=" << transformed_result.iterations
                  << " restarts=" << transformed_result.restarts
                  << " transformed_calls="
                  << transformed_result.operator_calls
                  << " inner_solves=" << tracked_solver.solve_calls()
                  << " inner_failures=" << tracked_solver.failed_solves()
                  << " inner_iterations="
                  << tracked_solver.total_iterations()
                  << " inner_max_iterations="
                  << tracked_solver.maximum_iterations()
                  << " inner_last_residual="
                  << tracked_solver.last_residual()
                  << " original_operator_calls="
                  << original.operator_calls()
                  << '\n';
        for(std::size_t index = 0;
            index < transformed_result.eigenpairs.size();
            ++index)
        {
            const auto& estimate =
                transformed_result.eigenpairs[index];
            std::cout
                << "mapped_eigenpair=" << index
                << " value_real=" << estimate.value.real()
                << " value_imag=" << estimate.value.imag()
                << " residual=" << estimate.residual
                << " converged=" << (estimate.converged ? 1 : 0)
                << '\n';
        }

        if(
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    operator_failure ||
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    invalid_input ||
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    dense_solver_failure)
        {
            throw std::runtime_error(
                "transformed eigensolver failed structurally: " +
                transformed_result.diagnostic);
        }

        std::size_t report_targets = 0;
        if(transformed_vectors.size() != 0)
        {
            stability::eigensolvers::
                rotated_projected_spectrum_recovery<
                    component_space_type,
                    product_space_type,
                    lapack_type>
                recovery(
                    *component_space,
                    *product_space,
                    lapack);
            const auto recovered = recovery.execute(
                original,
                transformed_vectors);
            std::cout
                << "recovery_status="
                << stability::eigensolvers::eigensolver_status_name(
                       recovered.status)
                << " projection_dimension="
                << recovered.projection_dimension
                << " recovery_operator_calls="
                << recovered.original_operator_calls
                << '\n';
            if(!recovered.succeeded())
            {
                throw std::runtime_error(
                    "projected recovery failed: " +
                    recovered.diagnostic);
            }
            for(std::size_t index = 0;
                index < recovered.eigenvalues.size();
                ++index)
            {
                const auto value = recovered.eigenvalues[index];
                if(near_report_target(value))
                    ++report_targets;
                std::cout
                    << "recovered_eigenvalue=" << index
                    << " value_real=" << value.real()
                    << " value_imag=" << value.imag()
                    << " report_target="
                    << (near_report_target(value) ? 1 : 0)
                    << '\n';
            }
        }
        std::cout
            << "report_targets_recovered=" << report_targets
            << " expected_total_across_angle_scan=4"
            << '\n';

        product_space->stop_use_vector(initial);
        product_space->free_vector(initial);
        std::cout
            << "S80PI_n1 transformed characterization: COMPLETED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "S80PI_n1 transformed characterization: FAILED: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
