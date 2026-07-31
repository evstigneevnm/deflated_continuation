#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
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

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

std::string matrix_path(int argc, char** argv)
{
    if(argc > 1)
        return argv[1];
    return
        "data/external/suitesparse/Rommes/S80PI_n1/S80PI_n1.mtx";
}

std::vector<double> initial_vector(std::size_t dimension)
{
    std::vector<double> result(dimension);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        result[index] =
            1.0 +
            0.013 * static_cast<double>(index % 29) -
            0.007 * static_cast<double>((index * index) % 31);
    }
    return result;
}

bool near_report_target(const complex_type& value)
{
    const double absolute_imaginary = std::abs(value.imag());
    const bool correct_frequency =
        std::abs(absolute_imaginary - 1.87) < 0.03 ||
        std::abs(absolute_imaginary - 1.68) < 0.03;
    return std::abs(value.real()) < 1.0e-4 && correct_frequency;
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
            coordinate.metadata.rows == 4028 &&
            coordinate.metadata.columns == 4028,
            "S80PI_n1 has unexpected dimensions");
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
        vector_space.init_vector(initial);
        vector_space.start_use_vector(initial);
        const auto initial_host = initial_vector(dimension);
        vector_space.set(initial_host.data(), initial, dimension);

        typename solver_type::options_type options;
        options.desired_eigenvalues = 4;
        options.krylov_dimension = 14;
        options.restart_dimension = 4;
        options.max_restarts = 25;
        options.absolute_tolerance = 1.0e-10;
        options.relative_tolerance = 1.0e-8;
        options.preserve_conjugate_pairs = true;
        options.target.kind =
            stability::eigensolvers::spectrum_target::largest_real;
        options.orthogonalization.method =
            nmfd::solvers::krylov::orthogonalization_method::
                modified_gram_schmidt;
        options.orthogonalization.reorthogonalization =
            nmfd::solvers::krylov::reorthogonalization_policy::dgks;
        options.orthogonalization.max_passes = 2;

        const auto result = solver.execute(
            [&matrix_operator](
                const vector_type& source,
                vector_type& destination)
            {
                return matrix_operator.apply(source, destination);
            },
            initial,
            options);

        require(
            result.status !=
                stability::eigensolvers::eigensolver_status::invalid_input &&
            result.status !=
                stability::eigensolvers::eigensolver_status::operator_failure &&
            result.status !=
                stability::eigensolvers::eigensolver_status::dense_solver_failure,
            "S80PI_n1 direct characterization failed structurally: " +
                result.diagnostic);
        require(
            result.operator_calls == matrix_operator.operator_calls(),
            "S80PI_n1 operator-call accounting mismatch");

        std::size_t report_targets = 0;
        std::cout << std::setprecision(17)
                  << "case=S80PI_n1"
                  << " mode=direct_largest_real"
                  << " status="
                  << stability::eigensolvers::eigensolver_status_name(
                         result.status)
                  << " iterations=" << result.iterations
                  << " restarts=" << result.restarts
                  << " operator_calls=" << result.operator_calls
                  << '\n';
        for(std::size_t index = 0; index < result.eigenpairs.size(); ++index)
        {
            const auto& estimate = result.eigenpairs[index];
            if(near_report_target(estimate.value))
                ++report_targets;
            std::cout
                << "eigenpair=" << index
                << " value_real=" << estimate.value.real()
                << " value_imag=" << estimate.value.imag()
                << " residual=" << estimate.residual
                << " converged=" << (estimate.converged ? 1 : 0)
                << '\n';
        }
        std::cout
            << "report_targets_recovered=" << report_targets
            << " expected=4"
            << '\n';
        if(report_targets != 4)
        {
            std::cout
                << "characterization=direct_untransformed_solver_does_not_"
                   "recover_report_targets\n";
        }
        else
        {
            std::cout
                << "characterization=direct_untransformed_solver_recovered_"
                   "report_targets\n";
        }

        vector_space.stop_use_vector(initial);
        vector_space.free_vector(initial);
        std::cout << "S80PI_n1 direct characterization: COMPLETED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr << "S80PI_n1 direct characterization: FAILED: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
