#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/io/matrix_market.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/eigensolvers/krylov_schur.h>

#include "common/host_csr_operator.h"

namespace
{

using complex_type = std::complex<double>;
using matrix_type =
    nmfd::operations::sparse::host_csr_matrix<double>;

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
    std::vector<double> result(dimension);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        result[index] =
            1.0 +
            0.017 * static_cast<double>(index % 23) +
            0.003 * static_cast<double>((index * index) % 19);
    }
    return result;
}

double physical_relative_residual(
    const matrix_type& matrix,
    const complex_type& eigenvalue,
    const std::vector<double>& real_vector,
    const std::vector<double>& imaginary_vector)
{
    const auto matrix_real = matrix.apply(real_vector);
    const auto matrix_imaginary = matrix.apply(imaginary_vector);
    long double residual_squared = 0.0L;
    long double vector_squared = 0.0L;
    for(std::size_t row = 0; row < real_vector.size(); ++row)
    {
        const long double residual_real =
            matrix_real[row] -
            eigenvalue.real() * real_vector[row] +
            eigenvalue.imag() * imaginary_vector[row];
        const long double residual_imaginary =
            matrix_imaginary[row] -
            eigenvalue.imag() * real_vector[row] -
            eigenvalue.real() * imaginary_vector[row];
        residual_squared +=
            residual_real * residual_real +
            residual_imaginary * residual_imaginary;
        vector_squared +=
            static_cast<long double>(real_vector[row]) * real_vector[row] +
            static_cast<long double>(imaginary_vector[row]) *
                imaginary_vector[row];
    }
    return static_cast<double>(
        std::sqrt(residual_squared / vector_squared));
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        using vector_space_type =
            scfd_vector_operations<scfd::backend::omp, double>;
        using vector_type = typename vector_space_type::vector_type;
        using lapack_type =
            nmfd::operations::linalg::host_small_dense_lapack<double>;
        using solver_type =
            stability::eigensolvers::krylov_schur<
                vector_space_type,
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

        vector_space_type vector_space(dimension);
        stability::tests::host_csr_operator<vector_space_type> matrix_operator(
            vector_space,
            matrix,
            nmfd::operations::sparse::host_csr_execution::openmp);
        lapack_type lapack;
        solver_type solver(vector_space, lapack);

        vector_type initial;
        vector_type real_vector;
        vector_type imaginary_vector;
        vector_space.init_vector(initial);
        vector_space.init_vector(real_vector);
        vector_space.init_vector(imaginary_vector);
        vector_space.start_use_vector(initial);
        vector_space.start_use_vector(real_vector);
        vector_space.start_use_vector(imaginary_vector);
        const auto initial_host = initial_vector(dimension);
        vector_space.set(initial_host.data(), initial, dimension);

        typename solver_type::options_type options;
        options.desired_eigenvalues = 2;
        options.krylov_dimension = 60;
        options.restart_dimension = 24;
        options.max_restarts = 150;
        options.absolute_tolerance = 1.0e-11;
        options.relative_tolerance = 1.0e-10;
        options.preserve_conjugate_pairs = true;
        options.target.kind =
            stability::eigensolvers::spectrum_target::largest_real;
        options.orthogonalization.method =
            nmfd::solvers::krylov::orthogonalization_method::
                modified_gram_schmidt;
        options.orthogonalization.reorthogonalization =
            nmfd::solvers::krylov::reorthogonalization_policy::dgks;
        options.orthogonalization.max_passes = 2;

        stability::eigensolvers::ritz_vector_storage<vector_space_type>
            recovered(vector_space, 3);
        const auto result = solver.execute(
            [&matrix_operator](
                const vector_type& source,
                vector_type& destination)
            {
                return matrix_operator.apply(source, destination);
            },
            initial,
            options,
            &recovered);

        std::cout << std::setprecision(17)
                  << "case=rdb450"
                  << " status="
                  << stability::eigensolvers::eigensolver_status_name(
                         result.status)
                  << " iterations=" << result.iterations
                  << " restarts=" << result.restarts
                  << " operator_calls=" << result.operator_calls
                  << '\n';
        require(
            result.status ==
                stability::eigensolvers::eigensolver_status::success,
            "rdb450 Krylov-Schur did not converge: " + result.diagnostic);
        require(
            result.operator_calls == matrix_operator.operator_calls(),
            "rdb450 operator-call accounting mismatch");
        require(
            result.eigenpairs.size() == 2 && recovered.size() == 2,
            "rdb450 did not return the leading conjugate pair");

        const std::vector<complex_type> expected{
            complex_type(
                -2.47220948810223023e-01,
                1.61074797405032522e+00),
            complex_type(
                -2.47220948810223023e-01,
                -1.61074797405032522e+00)};
        std::vector<bool> matched(expected.size(), false);
        for(std::size_t index = 0; index < result.eigenpairs.size(); ++index)
        {
            vector_space.assign(
                recovered.real(),
                static_cast<typename vector_space_type::ordinal_type>(
                    recovered.capacity()),
                static_cast<typename vector_space_type::ordinal_type>(index),
                real_vector);
            vector_space.assign(
                recovered.imaginary(),
                static_cast<typename vector_space_type::ordinal_type>(
                    recovered.capacity()),
                static_cast<typename vector_space_type::ordinal_type>(index),
                imaginary_vector);
            std::vector<double> real_host(dimension);
            std::vector<double> imaginary_host(dimension);
            vector_space.get(real_vector, real_host.data(), dimension);
            vector_space.get(
                imaginary_vector,
                imaginary_host.data(),
                dimension);

            const auto value = result.eigenpairs[index].value;
            const auto nearest = static_cast<std::size_t>(
                std::min_element(
                    expected.begin(),
                    expected.end(),
                    [&value](const complex_type& lhs, const complex_type& rhs)
                    {
                        return std::abs(lhs - value) <
                               std::abs(rhs - value);
                    }) -
                expected.begin());
            const double eigenvalue_error =
                std::abs(value - expected[nearest]);
            const double physical_residual =
                physical_relative_residual(
                    matrix,
                    value,
                    real_host,
                    imaginary_host);
            matched[nearest] = true;
            std::cout
                << "eigenpair=" << index
                << " value_real=" << value.real()
                << " value_imag=" << value.imag()
                << " estimate_residual=" << result.eigenpairs[index].residual
                << " physical_residual=" << physical_residual
                << " eigenvalue_error=" << eigenvalue_error
                << '\n';
            require(
                eigenvalue_error < 1.0e-8,
                "rdb450 leading eigenvalue mismatch");
            require(
                physical_residual < 1.0e-8,
                "rdb450 physical eigenpair residual is too large");
        }
        require(
            std::all_of(
                matched.begin(),
                matched.end(),
                [](bool value)
                {
                    return value;
                }),
            "rdb450 did not recover both conjugate eigenvalues");

        vector_space.stop_use_vector(imaginary_vector);
        vector_space.stop_use_vector(real_vector);
        vector_space.stop_use_vector(initial);
        vector_space.free_vector(imaginary_vector);
        vector_space.free_vector(real_vector);
        vector_space.free_vector(initial);
        std::cout << "rdb450 Krylov-Schur report test: PASSED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr << "rdb450 Krylov-Schur report test: FAILED: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
