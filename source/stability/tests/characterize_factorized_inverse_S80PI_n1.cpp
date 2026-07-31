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
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/io/matrix_market.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/transformations/affine_pencil_operator.h>
#include <stability/eigensolvers/transformations/complex_shift_block_operator.h>
#include <stability/eigensolvers/transformations/euler_polynomial_factorization.h>
#include <stability/eigensolvers/transformations/factorized_inverse_solver.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/stability_polynomial_operator.h>
#include <stability/eigensolvers/transformations/tracked_linear_solver.h>

#include "common/host_csr_operator.h"
#include "common/umfpack_complex_affine_solver.h"

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

unsigned basis_size(int argc, char** argv)
{
    if(argc > 3)
        return static_cast<unsigned>(std::stoul(argv[3]));
    return 120;
}

unsigned maximum_iterations(int argc, char** argv)
{
    if(argc > 4)
        return static_cast<unsigned>(std::stoul(argv[4]));
    return 2400;
}

double relative_tolerance(int argc, char** argv)
{
    if(argc > 5)
        return std::stod(argv[5]);
    return 1.0e-9;
}

std::vector<double> deterministic_vector(std::size_t dimension)
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
        using matrix_type =
            nmfd::operations::sparse::host_csr_matrix<double>;
        using matrix_operator_type =
            stability::tests::host_csr_operator<
                component_space_type>;
        using component_identity_type =
            stability::eigensolvers::transformations::identity_operator<
                component_space_type>;
        using euler_step_type =
            stability::eigensolvers::transformations::
                affine_pencil_operator<
                    component_space_type,
                    component_identity_type,
                    matrix_operator_type>;
        using factor_operator_type =
            stability::eigensolvers::transformations::
                complex_shift_block_operator<
                    product_space_type,
                    euler_step_type>;
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
                factor_operator_type>;
        using tracked_solver_type =
            stability::eigensolvers::transformations::
                tracked_linear_solver<
                    gmres_type,
                    factor_operator_type>;
        using iterative_inverse_type =
            stability::eigensolvers::transformations::
                factorized_inverse_solver<
                    product_space_type,
                    tracked_solver_type>;
        using exact_factor_solver_type =
            stability::tests::umfpack_complex_affine_solver<
                product_space_type,
                matrix_type>;
        using exact_inverse_type =
            stability::eigensolvers::transformations::
                factorized_inverse_solver<
                    product_space_type,
                    exact_factor_solver_type>;
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
        const matrix_type matrix = coordinate.to_host_csr();
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
        component_identity_type component_identity(*component_space);

        constexpr std::size_t power = 3;
        constexpr double total_time = 0.3;
        constexpr double step = total_time / power;
        constexpr double shift_radius = 1.01;
        const double phi = rotation_angle(argc, argv);
        const complex_type shift = std::polar(shift_radius, phi);
        const auto factors =
            stability::eigensolvers::transformations::
                euler_denominator_factors(step, power, shift);

        euler_step_type euler_step(
            *component_space,
            component_identity,
            original,
            1.0,
            step);

        typename gmres_type::params inner_parameters;
        inner_parameters.basis_size = basis_size(argc, argv);
        inner_parameters.batch_size =
            std::min(10u, inner_parameters.basis_size);
        inner_parameters.orthogonalization = "mgs";
        inner_parameters.reorthogonalization_policy = "dgks";
        inner_parameters.max_orthogonalization_passes = 2;
        inner_parameters.monitor.rel_tol =
            relative_tolerance(argc, argv);
        inner_parameters.monitor.abs_tol = 1.0e-13;
        inner_parameters.monitor.max_iters_num =
            maximum_iterations(argc, argv);
        inner_parameters.monitor.divide_out_norms_by_rel_base = false;

        std::vector<std::shared_ptr<factor_operator_type>>
            factor_operators;
        std::vector<std::shared_ptr<gmres_type>> factor_gmres;
        std::vector<std::shared_ptr<tracked_solver_type>>
            tracked_factors;
        factor_operators.reserve(power);
        factor_gmres.reserve(power);
        tracked_factors.reserve(power);
        for(const auto& factor : factors)
        {
            const complex_type root =
                complex_type(1.0, 0.0) - factor.diagonal_shift;
            auto factor_operator =
                std::make_shared<factor_operator_type>(
                    *product_space,
                    euler_step,
                    root.real(),
                    root.imag());
            auto factor_solver =
                std::make_shared<gmres_type>(
                    product_space,
                    nullptr,
                    inner_parameters);
            auto tracked =
                std::make_shared<tracked_solver_type>(
                    *factor_solver,
                    *factor_operator);
            factor_operators.push_back(std::move(factor_operator));
            factor_gmres.push_back(std::move(factor_solver));
            tracked_factors.push_back(std::move(tracked));
        }
        iterative_inverse_type iterative_inverse(
            *product_space,
            tracked_factors);

        std::vector<std::shared_ptr<exact_factor_solver_type>>
            exact_factors;
        exact_factors.reserve(power);
        for(const auto& factor : factors)
        {
            exact_factors.push_back(
                std::make_shared<exact_factor_solver_type>(
                    *product_space,
                    matrix,
                    factor.operator_scale,
                    factor.diagonal_shift));
        }
        exact_inverse_type exact_inverse(
            *product_space,
            exact_factors);

        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> right(*product_space);
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> iterative_solution(*product_space);
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> exact_solution(*product_space);
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> residual(*product_space);
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> error(*product_space);

        const auto right_host = deterministic_vector(dimension);
        product_space->set(
            right_host.data(),
            *right,
            right_host.size());
        product_space->assign_scalar(0.0, *iterative_solution);
        product_space->assign_scalar(0.0, *exact_solution);

        const bool iterative_succeeded =
            iterative_inverse.solve(*right, *iterative_solution);
        if(!exact_inverse.solve(*right, *exact_solution))
            throw std::runtime_error(
                "UMFPACK denominator oracle failed");

        polynomial_type euler_polynomial(
            *component_space,
            original,
            step,
            power,
            stability::eigensolvers::transformations::
                explicit_euler_stability_polynomial<double>());
        denominator_type denominator(
            *product_space,
            euler_polynomial,
            shift.real(),
            shift.imag());
        if(
            !denominator.apply(
                *iterative_solution,
                *residual))
        {
            throw std::runtime_error(
                "Euler denominator residual application failed");
        }
        product_space->add_lin_comb(
            -1.0,
            *right,
            1.0,
            *residual);
        product_space->assign_lin_comb(
            1.0,
            *iterative_solution,
            -1.0,
            *exact_solution,
            *error);

        const double relative_residual =
            product_space->norm(*residual) /
            product_space->norm(*right);
        const double relative_solution_error =
            product_space->norm(*error) /
            product_space->norm(*exact_solution);

        std::cout
            << std::setprecision(17)
            << "case=S80PI_n1"
            << " mode=factorized_iterative_denominator"
            << " phi=" << phi
            << " power=" << power
            << " step=" << step
            << " basis_size=" << inner_parameters.basis_size
            << " maximum_iterations="
            << inner_parameters.monitor.max_iters_num
            << " relative_tolerance="
            << inner_parameters.monitor.rel_tol
            << " status="
            << (iterative_succeeded ? "success" : "failure")
            << " relative_residual=" << relative_residual
            << " relative_solution_error="
            << relative_solution_error
            << " original_operator_calls="
            << original.operator_calls()
            << '\n';
        for(std::size_t index = 0; index < power; ++index)
        {
            const complex_type root =
                complex_type(1.0, 0.0) -
                factors[index].diagonal_shift;
            std::cout
                << "factor=" << index
                << " root_real=" << root.real()
                << " root_imag=" << root.imag()
                << " status="
                << (tracked_factors[index]->failed_solves() == 0
                        ? "success"
                        : "failure")
                << " iterations="
                << tracked_factors[index]->total_iterations()
                << " residual="
                << tracked_factors[index]->last_residual()
                << " operator_calls="
                << factor_operators[index]->operator_calls()
                << " exact_rcond="
                << exact_factors[index]->
                    reciprocal_condition_estimate()
                << '\n';
        }

        if(!iterative_succeeded)
            return EXIT_FAILURE;
        std::cout
            << "S80PI_n1 factorized inverse characterization: COMPLETED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "S80PI_n1 factorized inverse characterization: FAILED: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
