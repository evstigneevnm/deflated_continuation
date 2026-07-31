#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/log.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>

#include <stability/tests/common/analytical_dense_operator.h>
#include <stability/tests/common/analytical_eigenproblem.h>

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
void run_solve_case(
    const std::string& label,
    const std::string& orthogonalization,
    const std::string& reorthogonalization)
{
    using scalar_type = double;
    using vector_space_type = scfd_vector_operations<Backend, scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using log_type = scfd::utils::log_std;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            scalar_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    const auto problem = stability::tests::diagonal_eigenproblem<scalar_type>();
    auto vector_space =
        std::make_shared<vector_space_type>(problem.dimension());
    operator_type matrix_operator(*vector_space, problem);

    vector_type rhs;
    vector_type solution;
    vector_space->init_vector(rhs);
    vector_space->init_vector(solution);
    vector_space->start_use_vector(rhs);
    vector_space->start_use_vector(solution);

    const std::vector<scalar_type> reference = {0.5, -2.0, 3.0, -1.5};
    std::vector<scalar_type> rhs_host(problem.dimension(), 0.0);
    for(std::size_t row = 0; row < problem.dimension(); ++row)
        for(std::size_t col = 0; col < problem.dimension(); ++col)
            rhs_host[row] += problem.matrix(row, col)*reference[col];
    vector_space->set(rhs_host.data(), rhs, rhs_host.size());
    vector_space->assign_scalar(0.0, solution);

    typename solver_type::params parameters;
    parameters.basis_size = 4;
    parameters.batch_size = 1;
    parameters.orthogonalization = orthogonalization;
    parameters.reorthogonalization_policy = reorthogonalization;
    parameters.max_orthogonalization_passes =
        reorthogonalization == "none" ? 1 : 2;
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 20;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    solver_type solver(vector_space, nullptr, parameters);
    const bool converged = solver.solve(matrix_operator, rhs, solution);
    require(converged, label + " converged");

    std::vector<scalar_type> solution_host(problem.dimension(), 0.0);
    vector_space->get(solution, solution_host.data(), solution_host.size());
    for(std::size_t index = 0; index < reference.size(); ++index)
    {
        require(
            std::abs(solution_host[index] - reference[index]) <= 2e-10,
            label + " solution component " + std::to_string(index));
    }
    require(
        matrix_operator.operator_calls() >= 2,
        label + " records operator calls");
    require(
        solver.monitor().iters_performed() <=
            parameters.monitor.max_iters_num,
        label + " respects iteration limit");

    vector_space->stop_use_vector(solution);
    vector_space->stop_use_vector(rhs);
    vector_space->free_vector(solution);
    vector_space->free_vector(rhs);
}

template<class Backend>
void run_complex_solve_case(const std::string& label)
{
    using scalar_type = std::complex<double>;
    using vector_space_type =
        scfd_vector_operations<Backend, scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using log_type = scfd::utils::log_std;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    const auto problem = stability::tests::diagonal_eigenproblem<double>();
    auto vector_space =
        std::make_shared<vector_space_type>(problem.dimension());
    operator_type matrix_operator(*vector_space, problem);
    vector_type rhs;
    vector_type solution;
    vector_space->init_vector(rhs);
    vector_space->init_vector(solution);
    vector_space->start_use_vector(rhs);
    vector_space->start_use_vector(solution);

    const std::vector<scalar_type> reference{
        {0.5, -0.25},
        {-2.0, 0.75},
        {3.0, 1.5},
        {-1.5, -0.4}};
    std::vector<scalar_type> rhs_host(
        problem.dimension(),
        scalar_type{});
    for(std::size_t row = 0; row < problem.dimension(); ++row)
        for(std::size_t col = 0; col < problem.dimension(); ++col)
            rhs_host[row] += problem.matrix(row, col)*reference[col];
    vector_space->set(rhs_host.data(), rhs, rhs_host.size());
    vector_space->assign_scalar(scalar_type{}, solution);

    typename solver_type::params parameters;
    parameters.basis_size = 4;
    parameters.batch_size = 1;
    parameters.orthogonalization = "mgs";
    parameters.reorthogonalization_policy = "dgks";
    parameters.max_orthogonalization_passes = 2;
    parameters.monitor.rel_tol = 1e-12;
    parameters.monitor.abs_tol = 1e-14;
    parameters.monitor.max_iters_num = 20;
    parameters.monitor.divide_out_norms_by_rel_base = false;

    solver_type solver(vector_space, nullptr, parameters);
    require(
        solver.solve(matrix_operator, rhs, solution),
        label + " converged");
    std::vector<scalar_type> solution_host(
        problem.dimension(),
        scalar_type{});
    vector_space->get(
        solution,
        solution_host.data(),
        solution_host.size());
    for(std::size_t index = 0; index < reference.size(); ++index)
    {
        require(
            std::abs(solution_host[index] - reference[index]) <= 2e-10,
            label + " solution component " + std::to_string(index));
    }

    vector_space->stop_use_vector(solution);
    vector_space->stop_use_vector(rhs);
    vector_space->free_vector(solution);
    vector_space->free_vector(rhs);
}

void run_failure_case()
{
    using scalar_type = double;
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using log_type = scfd::utils::log_std;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            scalar_type>;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using solver_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            operator_type>;

    const auto problem = stability::tests::diagonal_eigenproblem<scalar_type>();
    auto vector_space =
        std::make_shared<vector_space_type>(problem.dimension());
    operator_type matrix_operator(*vector_space, problem);
    matrix_operator.fail_after(0);

    vector_type rhs;
    vector_type solution;
    vector_space->init_vector(rhs);
    vector_space->init_vector(solution);
    vector_space->start_use_vector(rhs);
    vector_space->start_use_vector(solution);
    const std::vector<scalar_type> rhs_host(problem.dimension(), 1.0);
    vector_space->set(rhs_host.data(), rhs, rhs_host.size());
    vector_space->assign_scalar(0.0, solution);

    typename solver_type::params parameters;
    parameters.basis_size = 4;
    parameters.monitor.max_iters_num = 10;
    solver_type solver(vector_space, nullptr, parameters);
    require(
        !solver.solve(matrix_operator, rhs, solution),
        "GMRES propagates an operator failure");

    vector_space->stop_use_vector(solution);
    vector_space->stop_use_vector(rhs);
    vector_space->free_vector(solution);
    vector_space->free_vector(rhs);
}

} // namespace

int main()
{
    run_solve_case<scfd::backend::serial_cpu>(
        "serial MGS",
        "mgs",
        "none");
    run_solve_case<scfd::backend::serial_cpu>(
        "serial MGS/DGKS",
        "mgs",
        "dgks");
    run_solve_case<scfd::backend::serial_cpu>(
        "serial CGS2",
        "cgs",
        "always");
    run_solve_case<scfd::backend::omp>(
        "OMP MGS/DGKS",
        "mgs",
        "dgks");
    run_complex_solve_case<scfd::backend::serial_cpu>(
        "serial complex MGS/DGKS");
    run_complex_solve_case<scfd::backend::omp>(
        "OMP complex MGS/DGKS");
    run_failure_case();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
