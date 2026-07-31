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
#include <stability/eigensolvers/transformations/inexact_exponential_operator.h>
#include <stability/eigensolvers/transformations/inverse_composed_operator.h>
#include <stability/eigensolvers/transformations/tracked_linear_solver.h>

#include "common/host_csr_operator.h"

namespace
{

using complex_type = std::complex<double>;

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

std::string matrix_path(int argc, char** argv)
{
    if(argc > 1)
        return argv[1];
    return "data/external/suitesparse/Bai/rdb450/rdb450.mtx";
}

std::vector<double> initial_vector(std::size_t dimension)
{
    std::vector<double> result(2 * dimension);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        result[index] =
            1.0 +
            0.017 * static_cast<double>(index % 23) +
            0.003 * static_cast<double>((index * index) % 19);
        result[dimension + index] =
            -0.25 +
            0.011 * static_cast<double>(index % 31) -
            0.002 * static_cast<double>((index * index) % 17);
    }
    return result;
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
                inexact_exponential_operator<
                    component_space_type,
                    matrix_operator_type>;
        using identity_type =
            stability::eigensolvers::transformations::identity_operator<
                product_space_type>;
        using denominator_type =
            stability::eigensolvers::transformations::
                complex_shift_block_operator<
                    product_space_type,
                    polynomial_type>;
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
        require(
            coordinate.metadata.rows == 450 &&
            coordinate.metadata.columns == 450,
            "rdb450 has unexpected dimensions");
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

        // The report figure labels exp(0.04 A). Four explicit Euler
        // substeps retain that total time while exercising the matrix-free
        // inexact-exponential implementation.
        constexpr double total_time = 0.04;
        constexpr std::size_t power = 4;
        constexpr double step = total_time / power;
        constexpr double shift_radius = 1.05;
        const double rotation_angle = std::acos(-1.0) / 11.0;
        const complex_type shift =
            std::polar(shift_radius, rotation_angle);
        polynomial_type polynomial(
            *component_space,
            original,
            step,
            power);
        denominator_type denominator(
            *product_space,
            polynomial,
            shift.real(),
            shift.imag());
        identity_type identity(*product_space);

        typename gmres_type::params inner_parameters;
        inner_parameters.basis_size = 40;
        inner_parameters.batch_size = 10;
        inner_parameters.orthogonalization = "mgs";
        inner_parameters.reorthogonalization_policy = "dgks";
        inner_parameters.max_orthogonalization_passes = 2;
        inner_parameters.monitor.rel_tol = 1.0e-12;
        inner_parameters.monitor.abs_tol = 1.0e-14;
        inner_parameters.monitor.max_iters_num = 200;
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
        options.desired_eigenvalues = 2;
        options.krylov_dimension = 30;
        options.restart_dimension = 12;
        options.max_restarts = 50;
        options.absolute_tolerance = 1.0e-10;
        options.relative_tolerance = 1.0e-9;
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

        stability::eigensolvers::ritz_vector_storage<product_space_type>
            transformed_vectors(*product_space, 3);
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
                  << "case=rdb450"
                  << " mode=inexact_exponential_shift_inverse"
                  << " total_time=" << total_time
                  << " power=" << power
                  << " step=" << step
                  << " shift_radius=" << shift_radius
                  << " rotation_angle=" << rotation_angle
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
                  << " original_operator_calls="
                  << original.operator_calls()
                  << '\n';
        require(
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::success,
            "rdb450 transformed Krylov-Schur did not converge: " +
                transformed_result.diagnostic);
        require(
            transformed_vectors.size() == 2,
            "rdb450 transformed Ritz-vector count mismatch");
        require(
            tracked_solver.failed_solves() == 0,
            "rdb450 inner GMRES failed");

        stability::eigensolvers::rotated_projected_spectrum_recovery<
            component_space_type,
            product_space_type,
            lapack_type>
            recovery(
                *component_space,
                *product_space,
                lapack);
        const auto recovered =
            recovery.execute(original, transformed_vectors);
        require(
            recovered.succeeded(),
            "rdb450 projected recovery failed: " +
                recovered.diagnostic);
        require(
            recovered.projection_dimension == 2,
            "rdb450 projected recovery dimension mismatch");

        const std::vector<complex_type> expected{
            complex_type(
                -2.47220948810223023e-01,
                1.61074797405032522e+00),
            complex_type(
                -2.47220948810223023e-01,
                -1.61074797405032522e+00)};
        std::vector<bool> matched(expected.size(), false);
        for(std::size_t index = 0;
            index < recovered.eigenvalues.size();
            ++index)
        {
            const auto value = recovered.eigenvalues[index];
            const std::size_t nearest = static_cast<std::size_t>(
                std::min_element(
                    expected.begin(),
                    expected.end(),
                    [&value](
                        const complex_type& lhs,
                        const complex_type& rhs)
                    {
                        return std::abs(lhs - value) <
                            std::abs(rhs - value);
                    }) -
                expected.begin());
            const double error =
                std::abs(value - expected[nearest]);
            matched[nearest] = true;
            std::cout
                << "recovered_eigenvalue=" << index
                << " value_real=" << value.real()
                << " value_imag=" << value.imag()
                << " error=" << error
                << '\n';
            require(
                error < 1.0e-7,
                "rdb450 transformed eigenvalue mismatch");
        }
        require(
            std::all_of(
                matched.begin(),
                matched.end(),
                [](bool value)
                {
                    return value;
                }),
            "rdb450 transformed run missed a conjugate eigenvalue");

        product_space->stop_use_vector(initial);
        product_space->free_vector(initial);
        std::cout
            << "rdb450 transformed Krylov-Schur report test: PASSED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "rdb450 transformed Krylov-Schur report test: FAILED: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
