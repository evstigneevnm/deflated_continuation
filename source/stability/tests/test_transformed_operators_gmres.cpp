#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/utils/log.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/transformations/affine_pencil_operator.h>
#include <stability/eigensolvers/transformations/complex_shift_block_operator.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/inexact_exponential_operator.h>
#include <stability/eigensolvers/transformations/inverse_composed_operator.h>
#include <stability/eigensolvers/transformations/tracked_linear_solver.h>

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
        std::cout << "FAIL " << message << '\n';
    }
}

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

template<class Solver>
typename Solver::params gmres_parameters(
    unsigned basis_size,
    unsigned maximum_iterations)
{
    typename Solver::params parameters;
    parameters.basis_size = basis_size;
    parameters.batch_size = basis_size;
    parameters.orthogonalization = "mgs";
    parameters.reorthogonalization_policy = "dgks";
    parameters.max_orthogonalization_passes = 2;
    parameters.monitor.rel_tol = 1.0e-13;
    parameters.monitor.abs_tol = 1.0e-14;
    parameters.monitor.max_iters_num = maximum_iterations;
    parameters.monitor.divide_out_norms_by_rel_base = false;
    return parameters;
}

template<class Backend>
void run_unrotated_case(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using polynomial_type =
        stability::eigensolvers::transformations::
            inexact_exponential_operator<
                vector_space_type,
                operator_type>;
    using identity_type =
        stability::eigensolvers::transformations::identity_operator<
            vector_space_type>;
    using denominator_type =
        stability::eigensolvers::transformations::affine_pencil_operator<
            vector_space_type,
            polynomial_type,
            identity_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using gmres_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            denominator_type>;
    using tracked_solver_type =
        stability::eigensolvers::transformations::tracked_linear_solver<
            gmres_type,
            denominator_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                vector_space_type,
                identity_type,
                tracked_solver_type>;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> eigenvalues{-4.0, -1.0, 2.0, 7.0};
    const double step = 0.1;
    const std::size_t power = 3;
    const double shift = 1.05;

    auto vector_space =
        std::make_shared<vector_space_type>(problem.dimension());
    operator_type original(*vector_space, problem);
    polynomial_type polynomial(
        *vector_space,
        original,
        step,
        power);
    identity_type identity(*vector_space);
    denominator_type denominator(
        *vector_space,
        polynomial,
        identity,
        1.0,
        -shift);
    gmres_type gmres(
        vector_space,
        nullptr,
        gmres_parameters<gmres_type>(4, 8));
    tracked_solver_type tracked(gmres, denominator);
    mapped_type mapped(*vector_space, identity, tracked);

    vector_type source;
    vector_type destination;
    vector_space->init_vector(source);
    vector_space->init_vector(destination);
    vector_space->start_use_vector(source);
    vector_space->start_use_vector(destination);
    const std::vector<double> host_source{1.0, -2.0, 0.5, 3.0};
    vector_space->set(
        host_source.data(),
        source,
        host_source.size());

    require(
        mapped.apply(source, destination),
        label + " unrotated GMRES inverse");
    std::vector<double> actual(problem.dimension());
    vector_space->get(destination, actual.data(), actual.size());
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
    {
        const double denominator_value =
            std::pow(1.0 + step * eigenvalues[index], power) -
            shift;
        require_close(
            actual[index],
            host_source[index] / denominator_value,
            2.0e-11,
            label + " unrotated component " + std::to_string(index));
    }
    require(
        tracked.solve_calls() == 1 &&
        tracked.failed_solves() == 0 &&
        tracked.total_iterations() > 0 &&
        tracked.total_iterations() <= 4,
        label + " unrotated inner accounting");

    vector_space->stop_use_vector(destination);
    vector_space->stop_use_vector(source);
    vector_space->free_vector(destination);
    vector_space->free_vector(source);
}

template<class Backend>
void run_rotated_case(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<component_space_type>;
    using product_vector_type =
        typename product_space_type::vector_type;
    using operator_type =
        stability::tests::analytical_dense_operator<
            component_space_type,
            double>;
    using polynomial_type =
        stability::eigensolvers::transformations::
            inexact_exponential_operator<
                component_space_type,
                operator_type>;
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
        nmfd::solvers::monitor_krylov<product_space_type, log_type>;
    using gmres_type =
        nmfd::solvers::gmres<
            product_space_type,
            monitor_type,
            log_type,
            denominator_type>;
    using tracked_solver_type =
        stability::eigensolvers::transformations::tracked_linear_solver<
            gmres_type,
            denominator_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                product_space_type,
                identity_type,
                tracked_solver_type>;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> eigenvalues{-4.0, -1.0, 2.0, 7.0};
    const double step = 0.1;
    const std::size_t power = 3;
    const std::complex<double> shift(1.0, 0.2);

    auto component_space =
        std::make_shared<component_space_type>(problem.dimension());
    auto product_space =
        std::make_shared<product_space_type>(
            *component_space,
            *component_space);
    operator_type original(*component_space, problem);
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
    gmres_type gmres(
        product_space,
        nullptr,
        gmres_parameters<gmres_type>(8, 12));
    tracked_solver_type tracked(gmres, denominator);
    mapped_type mapped(*product_space, identity, tracked);

    product_vector_type source;
    product_vector_type destination;
    product_space->init_vector(source);
    product_space->init_vector(destination);
    product_space->start_use_vector(source);
    product_space->start_use_vector(destination);
    const std::vector<double> host_source{
        1.0, -2.0, 0.5, 3.0,
        -0.5, 1.5, 2.0, -1.0};
    product_space->set(
        host_source.data(),
        source,
        host_source.size());

    require(
        mapped.apply(source, destination),
        label + " rotated GMRES inverse");
    std::vector<double> actual(host_source.size());
    product_space->get(destination, actual.data(), actual.size());
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
    {
        const std::complex<double> rhs(
            host_source[index],
            host_source[eigenvalues.size() + index]);
        const std::complex<double> polynomial_value(
            std::pow(1.0 + step * eigenvalues[index], power));
        const std::complex<double> expected =
            rhs / (polynomial_value - shift);
        require_close(
            actual[index],
            expected.real(),
            2.0e-11,
            label + " rotated real component " +
                std::to_string(index));
        require_close(
            actual[eigenvalues.size() + index],
            expected.imag(),
            2.0e-11,
            label + " rotated imaginary component " +
                std::to_string(index));
    }
    require(
        tracked.solve_calls() == 1 &&
        tracked.failed_solves() == 0 &&
        tracked.total_iterations() > 0 &&
        tracked.total_iterations() <= 8,
        label + " rotated inner accounting");

    product_space->stop_use_vector(destination);
    product_space->stop_use_vector(source);
    product_space->free_vector(destination);
    product_space->free_vector(source);
}

void run_failure_case()
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, double>;
    using vector_type = typename vector_space_type::vector_type;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using identity_type =
        stability::eigensolvers::transformations::identity_operator<
            vector_space_type>;
    using denominator_type =
        stability::eigensolvers::transformations::affine_pencil_operator<
            vector_space_type,
            operator_type,
            identity_type>;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<vector_space_type, log_type>;
    using gmres_type =
        nmfd::solvers::gmres<
            vector_space_type,
            monitor_type,
            log_type,
            denominator_type>;
    using tracked_solver_type =
        stability::eigensolvers::transformations::tracked_linear_solver<
            gmres_type,
            denominator_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                vector_space_type,
                identity_type,
                tracked_solver_type>;

    const auto problem =
        stability::tests::nonnormal_eigenproblem<double>();
    auto vector_space =
        std::make_shared<vector_space_type>(problem.dimension());
    operator_type original(*vector_space, problem);
    identity_type identity(*vector_space);
    denominator_type denominator(
        *vector_space,
        original,
        identity,
        1.0,
        -0.25);
    gmres_type gmres(
        vector_space,
        nullptr,
        gmres_parameters<gmres_type>(1, 1));
    tracked_solver_type tracked(gmres, denominator);
    mapped_type mapped(*vector_space, identity, tracked);

    vector_type source;
    vector_type destination;
    vector_space->init_vector(source);
    vector_space->init_vector(destination);
    vector_space->start_use_vector(source);
    vector_space->start_use_vector(destination);
    const std::vector<double> host_source{1.0, 2.0, -1.0};
    vector_space->set(
        host_source.data(),
        source,
        host_source.size());
    require(
        !mapped.apply(source, destination),
        "GMRES nonconvergence propagates to transformed operator");
    require(
        tracked.solve_calls() == 1 &&
        tracked.failed_solves() == 1 &&
        mapped.inner_solver_failures() == 1,
        "GMRES nonconvergence accounting");

    original.fail_after(original.operator_calls());
    require(
        !mapped.apply(source, destination),
        "early denominator-operator failure propagates");
    require(
        tracked.solve_calls() == 2 &&
        tracked.failed_solves() == 2 &&
        tracked.last_iterations() == 0 &&
        !tracked.last_residual_available(),
        "early denominator failure has no residual diagnostic");

    vector_space->stop_use_vector(destination);
    vector_space->stop_use_vector(source);
    vector_space->free_vector(destination);
    vector_space->free_vector(source);
}

} // namespace

int main()
{
    run_unrotated_case<scfd::backend::serial_cpu>("serial");
    run_unrotated_case<scfd::backend::omp>("OMP");
    run_rotated_case<scfd::backend::serial_cpu>("serial");
    run_rotated_case<scfd::backend::omp>("OMP");
    run_failure_case();

    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}
