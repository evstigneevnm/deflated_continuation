#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/eigensolvers/krylov_schur.h>

#include "common/analytical_dense_operator.h"
#include "common/analytical_eigenproblem.h"
#include "common/eigensolver_test_harness.h"

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << std::endl;
    }
}

stability::tests::analytical_eigenproblem<double>
restarted_diagonal_problem()
{
    using complex = std::complex<double>;
    using pair = stability::tests::analytical_eigenpair<double>;
    constexpr std::size_t dimension = 12;
    std::vector<double> matrix(dimension*dimension, 0.0);
    for(std::size_t index = 0; index < dimension; ++index)
        matrix[index + dimension*index] = static_cast<double>(index + 1);

    std::vector<pair> expected;
    for(std::size_t offset = 0; offset < 3; ++offset)
    {
        const std::size_t index = dimension - 1 - offset;
        std::vector<complex> vector(dimension, complex{});
        vector[index] = complex(1.0);
        expected.push_back(
            pair{complex(static_cast<double>(index + 1)), std::move(vector)});
    }
    return stability::tests::analytical_eigenproblem<double>(
        "restarted_diagonal_12",
        dimension,
        std::move(matrix),
        std::move(expected));
}

stability::tests::analytical_eigenproblem<double>
restarted_complex_pair_problem()
{
    using complex = std::complex<double>;
    using pair = stability::tests::analytical_eigenpair<double>;
    constexpr std::size_t dimension = 6;
    std::vector<double> matrix(dimension*dimension, 0.0);
    matrix[0 + dimension*0] = 5.0;
    matrix[1 + dimension*0] = 2.0;
    matrix[0 + dimension*1] = -2.0;
    matrix[1 + dimension*1] = 5.0;
    for(std::size_t index = 2; index < dimension; ++index)
        matrix[index + dimension*index] = 6.0 - static_cast<double>(index);

    const double inverse_sqrt_two = 1.0/std::sqrt(2.0);
    return stability::tests::analytical_eigenproblem<double>(
        "restarted_complex_pair_6",
        dimension,
        std::move(matrix),
        {
            pair{
                complex(5.0, 2.0),
                {
                    complex(inverse_sqrt_two),
                    complex(0.0, -inverse_sqrt_two),
                    complex{},
                    complex{},
                    complex{},
                    complex{}
                }},
            pair{
                complex(5.0, -2.0),
                {
                    complex(inverse_sqrt_two),
                    complex(0.0, inverse_sqrt_two),
                    complex{},
                    complex{},
                    complex{},
                    complex{}
                }}
        });
}

template<class Backend>
void run_case(
    const std::string& label,
    const stability::tests::analytical_eigenproblem<double>& problem,
    std::size_t krylov_dimension,
    std::size_t restart_dimension)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using solver_type =
        stability::eigensolvers::krylov_schur<
            vector_space_type,
            lapack_type>;

    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<vector_space_type, double>
        matrix_operator(vector_space, problem);
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
    std::vector<double> initial_host(problem.dimension());
    for(std::size_t index = 0; index < initial_host.size(); ++index)
    {
        initial_host[index] =
            1.0 + 0.173*static_cast<double>(index) +
            0.01*static_cast<double>(index*index);
    }
    vector_space.set(initial_host.data(), initial, initial_host.size());

    typename solver_type::options_type options;
    options.desired_eigenvalues = problem.eigenpairs().size();
    options.krylov_dimension = krylov_dimension;
    options.restart_dimension = restart_dimension;
    options.max_restarts = 100;
    options.absolute_tolerance = 1e-11;
    options.relative_tolerance = 1e-10;
    options.target.kind =
        stability::eigensolvers::spectrum_target::largest_real;
    options.orthogonalization.method =
        nmfd::solvers::krylov::orthogonalization_method::
            modified_gram_schmidt;
    options.orthogonalization.reorthogonalization =
        nmfd::solvers::krylov::reorthogonalization_policy::dgks;
    options.orthogonalization.max_passes = 2;

    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        recovered(
            vector_space,
            options.desired_eigenvalues + 1);
    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    const auto result =
        solver.execute(apply, initial, options, &recovered);

    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " solver status " +
            stability::eigensolvers::eigensolver_status_name(result.status));
    require(
        result.operator_calls == matrix_operator.operator_calls(),
        label + " operator-call accounting");
    require(
        recovered.size() == result.eigenpairs.size(),
        label + " recovered vector count");

    stability::tests::eigensolver_result<double> converted;
    converted.status =
        result.status == stability::eigensolvers::eigensolver_status::success
        ? stability::tests::eigensolver_status::success
        : stability::tests::eigensolver_status::no_convergence;
    converted.iterations = result.iterations;
    converted.restarts = result.restarts;
    converted.operator_calls = result.operator_calls;

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
        std::vector<double> real_host(problem.dimension());
        std::vector<double> imaginary_host(problem.dimension());
        vector_space.get(real_vector, real_host.data(), real_host.size());
        vector_space.get(
            imaginary_vector,
            imaginary_host.data(),
            imaginary_host.size());

        stability::tests::computed_eigenpair<double> pair;
        pair.value = result.eigenpairs[index].value;
        pair.right_eigenvector.resize(problem.dimension());
        for(std::size_t row = 0; row < problem.dimension(); ++row)
        {
            pair.right_eigenvector[row] =
                std::complex<double>(real_host[row], imaginary_host[row]);
        }
        converted.eigenpairs.emplace_back(std::move(pair));
    }

    stability::tests::eigensolver_test_tolerances<double> tolerances;
    tolerances.eigenvalue_absolute = 1e-8;
    tolerances.eigenvalue_relative = 1e-8;
    tolerances.residual_relative = 1e-8;
    tolerances.eigenvector_alignment = 1e-6;
    const auto report = stability::tests::validate_eigensolver_result(
        problem,
        converted,
        tolerances);
    checks += report.checks;
    for(const auto& failure : report.failures)
    {
        ++failures;
        std::cout << "FAIL " << label << " " << failure << std::endl;
    }

    vector_space.stop_use_vector(imaginary_vector);
    vector_space.stop_use_vector(real_vector);
    vector_space.stop_use_vector(initial);
    vector_space.free_vector(imaginary_vector);
    vector_space.free_vector(real_vector);
    vector_space.free_vector(initial);
}

void run_failure_case()
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, double>;
    using vector_type = typename vector_space_type::vector_type;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using solver_type =
        stability::eigensolvers::krylov_schur<
            vector_space_type,
            lapack_type>;

    const auto problem = restarted_diagonal_problem();
    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<vector_space_type, double>
        matrix_operator(vector_space, problem);
    matrix_operator.fail_after(2);
    lapack_type lapack;
    solver_type solver(vector_space, lapack);

    vector_type initial;
    vector_space.init_vector(initial);
    vector_space.start_use_vector(initial);
    const std::vector<double> initial_host(problem.dimension(), 1.0);
    vector_space.set(initial_host.data(), initial, initial_host.size());

    typename solver_type::options_type options;
    options.desired_eigenvalues = 2;
    options.krylov_dimension = 6;
    options.max_restarts = 2;
    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    const auto result = solver.execute(apply, initial, options);
    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::operator_failure,
        "Krylov-Schur propagates operator failure");
    require(
        result.operator_calls == matrix_operator.operator_calls(),
        "Krylov-Schur failure call accounting");

    vector_space.stop_use_vector(initial);
    vector_space.free_vector(initial);
}

} // namespace

int main()
{
    run_case<scfd::backend::serial_cpu>(
        "serial restarted diagonal",
        restarted_diagonal_problem(),
        6,
        4);
    run_case<scfd::backend::omp>(
        "OMP restarted diagonal",
        restarted_diagonal_problem(),
        6,
        4);
    run_case<scfd::backend::serial_cpu>(
        "serial restarted complex pair",
        restarted_complex_pair_problem(),
        4,
        2);
    run_failure_case();

    std::cout << "Checks: " << checks << ", failures: " << failures
              << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
