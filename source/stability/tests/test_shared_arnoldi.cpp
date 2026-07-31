#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include <nmfd/solvers/krylov/arnoldi.h>
#include <nmfd/solvers/krylov/basis_storage.h>

#include "common/analytical_dense_operator.h"
#include "common/analytical_eigenproblem.h"

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

template<class Backend>
void run_arnoldi_case(
    const std::string& label,
    nmfd::solvers::krylov::orthogonalization_method method,
    nmfd::solvers::krylov::reorthogonalization_policy reorthogonalization)
{
    using scalar_type = double;
    using vector_space_type = scfd_vector_operations<Backend, scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using dense_backend_type =
        nmfd::operations::linalg::host_small_dense_backend<
            std::ptrdiff_t,
            scalar_type>;

    const auto problem = stability::tests::diagonal_eigenproblem<scalar_type>();
    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<
        vector_space_type,
        scalar_type> matrix_operator(vector_space, problem);

    constexpr std::size_t steps = 4;
    constexpr std::size_t basis_capacity = steps + 1;
    nmfd::solvers::krylov::basis_storage<vector_space_type> basis(
        vector_space,
        basis_capacity);

    vector_type initial;
    vector_type basis_vector;
    vector_type candidate;
    vector_type residual;
    vector_space.init_vector(initial);
    vector_space.init_vector(basis_vector);
    vector_space.init_vector(candidate);
    vector_space.init_vector(residual);
    vector_space.start_use_vector(initial);
    vector_space.start_use_vector(basis_vector);
    vector_space.start_use_vector(candidate);
    vector_space.start_use_vector(residual);

    const std::vector<scalar_type> initial_host = {1.0, -2.0, 3.0, 4.0};
    vector_space.set(initial_host.data(), initial, initial_host.size());

    dense_backend_type dense_backend(
        static_cast<std::ptrdiff_t>(basis_capacity),
        static_cast<std::ptrdiff_t>(steps));
    typename dense_backend_type::matrix_type hessenberg;
    dense_backend.init_matrix(hessenberg);
    dense_backend.assign_scalar_matrix(0.0, hessenberg);
    std::vector<scalar_type> coefficients(basis_capacity, 0.0);
    std::vector<scalar_type> pass_coefficients(basis_capacity, 0.0);

    nmfd::solvers::krylov::orthogonalization_options<scalar_type> options;
    options.method = method;
    options.reorthogonalization = reorthogonalization;
    options.max_passes =
        reorthogonalization ==
                nmfd::solvers::krylov::reorthogonalization_policy::none
            ? 1
            : 2;
    options.compute_orthogonality_error = true;
    options.breakdown_relative = 1e-12;

    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    const auto result =
        nmfd::solvers::krylov::build_arnoldi_factorization(
            vector_space,
            apply,
            initial,
            basis.data(),
            basis.capacity(),
            steps,
            basis_vector,
            candidate,
            coefficients.data(),
            pass_coefficients.data(),
            hessenberg,
            options);

    require(
        result.status == nmfd::solvers::krylov::arnoldi_status::happy_breakdown,
        label + " reaches invariant-subspace happy breakdown");
    require(result.completed_steps == steps, label + " completed step count");
    require(result.operator_calls == steps, label + " operator call count");

    for(std::size_t left = 0; left < steps; ++left)
    {
        vector_space.assign(
            basis.data(),
            basis.capacity(),
            left,
            basis_vector);
        for(std::size_t right = 0; right < steps; ++right)
        {
            vector_space.assign(
                basis.data(),
                basis.capacity(),
                right,
                candidate);
            const scalar_type product =
                vector_space.scalar_prod(basis_vector, candidate);
            const scalar_type expected = left == right ? 1.0 : 0.0;
            require(
                std::abs(product - expected) <= 2e-10,
                label + " orthogonality (" + std::to_string(left) + "," +
                    std::to_string(right) + ")");
        }
    }

    for(std::size_t column = 0; column < steps; ++column)
    {
        vector_space.assign(
            basis.data(),
            basis.capacity(),
            column,
            basis_vector);
        require(
            matrix_operator.apply(basis_vector, residual),
            label + " relation operator apply");
        const std::size_t last_row =
            std::min(column + 1, steps - 1);
        for(std::size_t row = 0; row <= last_row; ++row)
        {
            vector_space.add_lin_comb(
                -hessenberg(row, column),
                basis.data(),
                basis.capacity(),
                row,
                1.0,
                residual);
        }
        require(
            vector_space.norm(residual) <= 5e-10,
            label + " Arnoldi relation column " + std::to_string(column));
    }

    vector_space.stop_use_vector(residual);
    vector_space.stop_use_vector(candidate);
    vector_space.stop_use_vector(basis_vector);
    vector_space.stop_use_vector(initial);
    vector_space.free_vector(residual);
    vector_space.free_vector(candidate);
    vector_space.free_vector(basis_vector);
    vector_space.free_vector(initial);
}

template<class Backend>
void run_operator_failure_case(const std::string& label)
{
    using scalar_type = double;
    using vector_space_type = scfd_vector_operations<Backend, scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using matrix_type =
        nmfd::operations::linalg::host_dense_matrix<scalar_type>;

    const auto problem = stability::tests::diagonal_eigenproblem<scalar_type>();
    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<
        vector_space_type,
        scalar_type> matrix_operator(vector_space, problem);
    matrix_operator.fail_after(1);

    nmfd::solvers::krylov::basis_storage<vector_space_type> basis(
        vector_space,
        4);
    vector_type initial;
    vector_type basis_vector;
    vector_type candidate;
    vector_space.init_vector(initial);
    vector_space.init_vector(basis_vector);
    vector_space.init_vector(candidate);
    vector_space.start_use_vector(initial);
    vector_space.start_use_vector(basis_vector);
    vector_space.start_use_vector(candidate);
    const std::vector<scalar_type> initial_host = {1.0, 1.0, 1.0, 1.0};
    vector_space.set(initial_host.data(), initial, initial_host.size());

    matrix_type hessenberg(4, 3);
    std::vector<scalar_type> coefficients(4, 0.0);
    std::vector<scalar_type> pass_coefficients(4, 0.0);
    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    const auto result =
        nmfd::solvers::krylov::build_arnoldi_factorization(
            vector_space,
            apply,
            initial,
            basis.data(),
            basis.capacity(),
            3,
            basis_vector,
            candidate,
            coefficients.data(),
            pass_coefficients.data(),
            hessenberg);
    require(
        result.status == nmfd::solvers::krylov::arnoldi_status::operator_failure,
        label + " propagates operator failure");
    require(result.operator_calls == 2, label + " failure operator call count");

    vector_space.stop_use_vector(candidate);
    vector_space.stop_use_vector(basis_vector);
    vector_space.stop_use_vector(initial);
    vector_space.free_vector(candidate);
    vector_space.free_vector(basis_vector);
    vector_space.free_vector(initial);
}

} // namespace

int main()
{
    using method = nmfd::solvers::krylov::orthogonalization_method;
    using reorth = nmfd::solvers::krylov::reorthogonalization_policy;

    run_arnoldi_case<scfd::backend::serial_cpu>(
        "serial MGS/DGKS",
        method::modified_gram_schmidt,
        reorth::dgks);
    run_arnoldi_case<scfd::backend::serial_cpu>(
        "serial CGS2",
        method::classical_gram_schmidt,
        reorth::always);
    run_arnoldi_case<scfd::backend::omp>(
        "OMP MGS/DGKS",
        method::modified_gram_schmidt,
        reorth::dgks);
    run_operator_failure_case<scfd::backend::serial_cpu>("serial");

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
