#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <stability/eigensolvers/inverse_iteration.h>

#include "common/analytical_dense_operator.h"
#include "common/analytical_eigenproblem.h"
#include "common/analytical_shifted_solver.h"

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
void run_case(
    const std::string& label,
    const stability::tests::analytical_eigenproblem<double>& problem,
    double shift,
    double expected)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;

    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<vector_space_type, double>
        matrix_operator(vector_space, problem);
    stability::tests::analytical_shifted_solver<vector_space_type, double>
        shifted_solver(vector_space, problem, shift);
    stability::eigensolvers::inverse_iteration<vector_space_type> solver(
        vector_space);

    vector_type initial;
    vector_type eigenvector;
    vector_space.init_vector(initial);
    vector_space.init_vector(eigenvector);
    vector_space.start_use_vector(initial);
    vector_space.start_use_vector(eigenvector);
    std::vector<double> initial_host(problem.dimension(), 1.0);
    for(std::size_t index = 0; index < initial_host.size(); ++index)
        initial_host[index] += 0.125*static_cast<double>(index);
    vector_space.set(initial_host.data(), initial, initial_host.size());

    stability::eigensolvers::inverse_iteration_options<double> options;
    options.shift = shift;
    options.max_iterations = 100;
    options.absolute_tolerance = 1e-12;
    options.relative_tolerance = 1e-10;
    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    auto solve = [&shifted_solver](
                     const vector_type& source,
                     vector_type& destination)
    {
        return shifted_solver.solve(source, destination);
    };
    const auto result =
        solver.execute(apply, solve, initial, eigenvector, options);

    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " converged");
    require(result.eigenpairs.size() == 1, label + " eigenpair count");
    if(!result.eigenpairs.empty())
    {
        require(
            std::abs(result.eigenpairs.front().value.real() - expected) <=
                1e-9,
            label + " eigenvalue");
        require(
            std::abs(result.eigenpairs.front().value.imag()) <= 1e-14,
            label + " real eigenvalue");
        require(
            result.eigenpairs.front().residual <= 1e-9,
            label + " residual");
    }
    require(
        result.inner_solver_calls == shifted_solver.solve_calls(),
        label + " solve-call accounting");
    require(
        result.operator_calls == matrix_operator.operator_calls(),
        label + " operator-call accounting");

    vector_space.stop_use_vector(eigenvector);
    vector_space.stop_use_vector(initial);
    vector_space.free_vector(eigenvector);
    vector_space.free_vector(initial);
}

void run_failure_cases()
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, double>;
    using vector_type = typename vector_space_type::vector_type;

    const auto problem = stability::tests::diagonal_eigenproblem<double>();
    vector_space_type vector_space(problem.dimension());
    stability::tests::analytical_dense_operator<vector_space_type, double>
        matrix_operator(vector_space, problem);
    stability::tests::analytical_shifted_solver<vector_space_type, double>
        shifted_solver(vector_space, problem, 0.0);
    stability::eigensolvers::inverse_iteration<vector_space_type> solver(
        vector_space);

    vector_type initial;
    vector_type output;
    vector_space.init_vector(initial);
    vector_space.init_vector(output);
    vector_space.start_use_vector(initial);
    vector_space.start_use_vector(output);
    const std::vector<double> initial_host(problem.dimension(), 1.0);
    vector_space.set(initial_host.data(), initial, initial_host.size());

    shifted_solver.fail_after(0);
    auto apply = [&matrix_operator](
                     const vector_type& source,
                     vector_type& destination)
    {
        return matrix_operator.apply(source, destination);
    };
    auto solve = [&shifted_solver](
                     const vector_type& source,
                     vector_type& destination)
    {
        return shifted_solver.solve(source, destination);
    };
    const auto solve_failure = solver.execute(
        apply,
        solve,
        initial,
        output);
    require(
        solve_failure.status ==
            stability::eigensolvers::eigensolver_status::
                inner_solver_failure,
        "inverse iteration propagates inner-solver failure");

    stability::tests::analytical_shifted_solver<vector_space_type, double>
        working_solver(vector_space, problem, 0.0);
    matrix_operator.fail_after(0);
    auto working_solve = [&working_solver](
                             const vector_type& source,
                             vector_type& destination)
    {
        return working_solver.solve(source, destination);
    };
    const auto operator_failure = solver.execute(
        apply,
        working_solve,
        initial,
        output);
    require(
        operator_failure.status ==
            stability::eigensolvers::eigensolver_status::operator_failure,
        "inverse iteration propagates original-operator failure");

    vector_space.stop_use_vector(output);
    vector_space.stop_use_vector(initial);
    vector_space.free_vector(output);
    vector_space.free_vector(initial);
}

} // namespace

int main()
{
    run_case<scfd::backend::serial_cpu>(
        "serial unshifted diagonal",
        stability::tests::diagonal_eigenproblem<double>(),
        0.0,
        -1.0);
    run_case<scfd::backend::serial_cpu>(
        "serial shifted symmetric",
        stability::tests::symmetric_eigenproblem<double>(),
        0.75,
        1.0);
    run_case<scfd::backend::omp>(
        "OMP shifted nonnormal",
        stability::tests::nonnormal_eigenproblem<double>(),
        1.75,
        2.0);
    run_failure_cases();

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
