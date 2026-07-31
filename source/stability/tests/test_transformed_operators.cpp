#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/eigensolvers/krylov_schur.h>
#include <stability/eigensolvers/projected_spectrum_recovery.h>
#include <stability/eigensolvers/rotated_projected_spectrum_recovery.h>
#include <stability/eigensolvers/transformations/affine_pencil_operator.h>
#include <stability/eigensolvers/transformations/complex_shift_block_operator.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/inexact_exponential_operator.h>
#include <stability/eigensolvers/transformations/inverse_composed_operator.h>
#include <stability/eigensolvers/transformations/spectral_mapping.h>

#include "common/analytical_dense_operator.h"
#include "common/analytical_eigenproblem.h"
#include "common/analytical_matrix_solver.h"

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

std::vector<double> diagonal_matrix(
    const std::vector<double>& diagonal)
{
    std::vector<double> result(
        diagonal.size() * diagonal.size(),
        0.0);
    for(std::size_t index = 0; index < diagonal.size(); ++index)
        result[index * diagonal.size() + index] = diagonal[index];
    return result;
}

std::vector<double> complex_shift_block_matrix(
    const std::vector<double>& diagonal,
    double shift_real,
    double shift_imaginary)
{
    const std::size_t component_dimension = diagonal.size();
    const std::size_t dimension = 2 * component_dimension;
    std::vector<double> result(dimension * dimension, 0.0);
    for(std::size_t index = 0; index < component_dimension; ++index)
    {
        result[index * dimension + index] =
            diagonal[index] - shift_real;
        result[index * dimension + component_dimension + index] =
            shift_imaginary;
        result[(component_dimension + index) * dimension + index] =
            -shift_imaginary;
        result[(component_dimension + index) * dimension +
               component_dimension + index] =
            diagonal[index] - shift_real;
    }
    return result;
}

template<class VectorSpace>
void set_vector(
    const VectorSpace& vector_space,
    const std::vector<double>& host,
    typename VectorSpace::vector_type& vector)
{
    vector_space.set(host.data(), vector, host.size());
}

template<class VectorSpace>
std::vector<double> get_vector(
    const VectorSpace& vector_space,
    const typename VectorSpace::vector_type& vector,
    std::size_t dimension)
{
    std::vector<double> result(dimension);
    vector_space.get(vector, result.data(), result.size());
    return result;
}

template<class Backend>
void run_cayley_case(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using identity_type =
        stability::eigensolvers::transformations::identity_operator<
            vector_space_type>;
    using numerator_type =
        stability::eigensolvers::transformations::affine_pencil_operator<
            vector_space_type,
            operator_type,
            identity_type>;
    using solver_type =
        stability::tests::analytical_matrix_solver<vector_space_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                vector_space_type,
                numerator_type,
                solver_type>;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> eigenvalues{-4.0, -1.0, 2.0, 7.0};
    const double sigma = 1.5;
    const double sigma_zero = 0.5;
    vector_space_type vector_space(problem.dimension());
    operator_type original(vector_space, problem);
    identity_type identity(vector_space);
    numerator_type numerator(
        vector_space,
        original,
        identity,
        1.0,
        sigma_zero);

    auto denominator_matrix = diagonal_matrix(eigenvalues);
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
        denominator_matrix[index * eigenvalues.size() + index] -= sigma;
    solver_type denominator_solver(
        vector_space,
        problem.dimension(),
        denominator_matrix);
    mapped_type mapped(vector_space, numerator, denominator_solver);

    vector_type source;
    vector_type destination;
    vector_space.init_vector(source);
    vector_space.init_vector(destination);
    vector_space.start_use_vector(source);
    vector_space.start_use_vector(destination);
    const std::vector<double> host_source{1.0, -2.0, 0.5, 3.0};
    set_vector(vector_space, host_source, source);
    require(mapped.apply(source, destination), label + " Cayley apply");
    const auto actual =
        get_vector(vector_space, destination, problem.dimension());
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
    {
        const auto mapped_eigenvalue =
            stability::eigensolvers::transformations::cayley_map(
                std::complex<double>(eigenvalues[index]),
                sigma,
                sigma_zero);
        require_close(
            actual[index],
            mapped_eigenvalue.real() * host_source[index],
            1.0e-12,
            label + " Cayley component " + std::to_string(index));
        const auto recovered =
            stability::eigensolvers::transformations::cayley_inverse_map(
                mapped_eigenvalue,
                sigma,
                sigma_zero);
        require_close(
            recovered.real(),
            eigenvalues[index],
            1.0e-12,
            label + " Cayley inverse map " + std::to_string(index));
    }
    require(
        mapped.inner_solver_calls() == 1 &&
        denominator_solver.solve_calls() == 1,
        label + " Cayley inner-solve accounting");

    denominator_solver.fail_after(1);
    require(
        !mapped.apply(source, destination),
        label + " Cayley propagates inner-solver failure");
    require(
        mapped.inner_solver_failures() == 1,
        label + " Cayley failure accounting");

    vector_space.stop_use_vector(destination);
    vector_space.stop_use_vector(source);
    vector_space.free_vector(destination);
    vector_space.free_vector(source);
}

template<class Backend>
void run_inexact_exponential_case(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using identity_type =
        stability::eigensolvers::transformations::identity_operator<
            vector_space_type>;
    using polynomial_type =
        stability::eigensolvers::transformations::
            inexact_exponential_operator<
                vector_space_type,
                operator_type>;
    using shifted_polynomial_type =
        stability::eigensolvers::transformations::affine_pencil_operator<
            vector_space_type,
            polynomial_type,
            identity_type>;
    using exact_solver_type =
        stability::tests::analytical_matrix_solver<vector_space_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                vector_space_type,
                identity_type,
                exact_solver_type>;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using eigensolver_type =
        stability::eigensolvers::krylov_schur<
            vector_space_type,
            lapack_type>;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> eigenvalues{-4.0, -1.0, 2.0, 7.0};
    const double step = 0.1;
    const std::size_t power = 3;
    const double sigma = 1.05;
    vector_space_type vector_space(problem.dimension());
    operator_type original(vector_space, problem);
    identity_type identity(vector_space);
    polynomial_type polynomial(
        vector_space,
        original,
        step,
        power);
    shifted_polynomial_type shifted_polynomial(
        vector_space,
        polynomial,
        identity,
        1.0,
        -sigma);

    vector_type source;
    vector_type destination;
    vector_type shifted_destination;
    vector_space.init_vector(source);
    vector_space.init_vector(destination);
    vector_space.init_vector(shifted_destination);
    vector_space.start_use_vector(source);
    vector_space.start_use_vector(destination);
    vector_space.start_use_vector(shifted_destination);
    const std::vector<double> host_source{1.0, -2.0, 0.5, 3.0};
    set_vector(vector_space, host_source, source);

    require(
        polynomial.apply(source, destination),
        label + " inexact polynomial apply");
    const auto polynomial_actual =
        get_vector(vector_space, destination, problem.dimension());
    require(
        shifted_polynomial.apply(source, shifted_destination),
        label + " shifted polynomial apply");
    const auto shifted_actual =
        get_vector(vector_space, shifted_destination, problem.dimension());

    std::vector<double> denominator_diagonal(eigenvalues.size());
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
    {
        const double polynomial_eigenvalue =
            std::pow(1.0 + step * eigenvalues[index], power);
        denominator_diagonal[index] =
            polynomial_eigenvalue - sigma;
        require_close(
            polynomial_actual[index],
            polynomial_eigenvalue * host_source[index],
            1.0e-12,
            label + " polynomial component " + std::to_string(index));
        require_close(
            shifted_actual[index],
            denominator_diagonal[index] * host_source[index],
            1.0e-12,
            label + " shifted polynomial component " +
                std::to_string(index));
    }
    require(
        polynomial.original_operator_calls() == 2 * power,
        label + " polynomial original-operator accounting");

    exact_solver_type denominator_solver(
        vector_space,
        problem.dimension(),
        diagonal_matrix(denominator_diagonal));
    mapped_type mapped(vector_space, identity, denominator_solver);
    require(
        mapped.apply(source, destination),
        label + " inexact shift-inverse apply");
    const auto mapped_actual =
        get_vector(vector_space, destination, problem.dimension());
    for(std::size_t index = 0; index < eigenvalues.size(); ++index)
    {
        const auto mapped_eigenvalue =
            stability::eigensolvers::transformations::
                inexact_exponential_map(
                    std::complex<double>(eigenvalues[index]),
                    step,
                    power,
                    std::complex<double>(sigma));
        require_close(
            mapped_actual[index],
            mapped_eigenvalue.real() * host_source[index],
            1.0e-12,
            label + " inexact inverse component " +
                std::to_string(index));
    }

    lapack_type lapack;
    eigensolver_type eigensolver(vector_space, lapack);
    typename eigensolver_type::options_type options;
    options.desired_eigenvalues = 2;
    options.krylov_dimension = 4;
    options.restart_dimension = 2;
    options.max_restarts = 10;
    options.absolute_tolerance = 1.0e-12;
    options.relative_tolerance = 1.0e-11;
    options.target.kind =
        stability::eigensolvers::spectrum_target::largest_magnitude;
    options.orthogonalization.reorthogonalization =
        nmfd::solvers::krylov::reorthogonalization_policy::dgks;
    options.orthogonalization.max_passes = 2;

    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        transformed_vectors(vector_space, 3);
    const auto transformed_result = eigensolver.execute(
        [&mapped](
            const vector_type& input,
            vector_type& output)
        {
            return mapped.apply(input, output);
        },
        source,
        options,
        &transformed_vectors);
    require(
        transformed_result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " transformed Krylov-Schur convergence");

    stability::eigensolvers::projected_spectrum_recovery<
        vector_space_type,
        lapack_type>
        recovery(vector_space, lapack);
    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        physical_vectors(vector_space, 4);
    const auto recovered =
        recovery.execute(original, transformed_vectors, {}, &physical_vectors);
    require(
        recovered.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " projected spectrum recovery");
    require(
        recovered.projection_dimension == 2,
        label + " projected spectrum dimension");
    require(
        recovered.eigenpairs.size() == 2 &&
            physical_vectors.size() == recovered.eigenpairs.size() &&
            recovered.all_converged(),
        label + " projected physical Ritz recovery");

    const std::vector<double> expected_original{-1.0, 2.0};
    for(const double expected : expected_original)
    {
        const auto nearest = std::min_element(
            recovered.eigenvalues.begin(),
            recovered.eigenvalues.end(),
            [expected](
                const std::complex<double>& lhs,
                const std::complex<double>& rhs)
            {
                return std::abs(lhs - std::complex<double>(expected)) <
                       std::abs(rhs - std::complex<double>(expected));
            });
        require(
            nearest != recovered.eigenvalues.end() &&
            std::abs(*nearest - std::complex<double>(expected)) < 1.0e-9,
            label + " recovered original eigenvalue " +
                std::to_string(expected));
    }

    vector_space.stop_use_vector(shifted_destination);
    vector_space.stop_use_vector(destination);
    vector_space.stop_use_vector(source);
    vector_space.free_vector(shifted_destination);
    vector_space.free_vector(destination);
    vector_space.free_vector(source);
}

template<class Backend>
void run_rotated_inexact_exponential_case(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<component_space_type>;
    using component_operator_type =
        stability::tests::analytical_dense_operator<
            component_space_type,
            double>;
    using polynomial_type =
        stability::eigensolvers::transformations::
            inexact_exponential_operator<
                component_space_type,
                component_operator_type>;
    using block_operator_type =
        stability::eigensolvers::transformations::
            complex_shift_block_operator<
                product_space_type,
                polynomial_type>;
    using identity_type =
        stability::eigensolvers::transformations::identity_operator<
            product_space_type>;
    using solver_type =
        stability::tests::analytical_matrix_solver<product_space_type>;
    using mapped_type =
        stability::eigensolvers::transformations::
            inverse_composed_operator<
                product_space_type,
                identity_type,
                solver_type>;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using eigensolver_type =
        stability::eigensolvers::krylov_schur<
            product_space_type,
            lapack_type>;
    using product_vector_type =
        typename product_space_type::vector_type;

    const auto problem =
        stability::tests::diagonal_eigenproblem<double>();
    const std::vector<double> original_eigenvalues{-4.0, -1.0, 2.0, 7.0};
    const double step = 0.1;
    const std::size_t power = 3;
    const std::complex<double> shift(1.0, 0.2);

    component_space_type component_space(problem.dimension());
    product_space_type product_space(
        component_space,
        component_space);
    component_operator_type original(component_space, problem);
    polynomial_type polynomial(
        component_space,
        original,
        step,
        power);
    block_operator_type block(
        product_space,
        polynomial,
        shift.real(),
        shift.imag());

    std::vector<double> polynomial_eigenvalues(
        original_eigenvalues.size());
    for(std::size_t index = 0;
        index < original_eigenvalues.size();
        ++index)
    {
        polynomial_eigenvalues[index] = std::pow(
            1.0 + step * original_eigenvalues[index],
            power);
    }
    const auto block_matrix = complex_shift_block_matrix(
        polynomial_eigenvalues,
        shift.real(),
        shift.imag());

    product_vector_type source;
    product_vector_type destination;
    product_space.init_vector(source);
    product_space.init_vector(destination);
    product_space.start_use_vector(source);
    product_space.start_use_vector(destination);
    const std::vector<double> host_source{
        1.0, -2.0, 0.5, 3.0,
        -0.5, 1.5, 2.0, -1.0};
    product_space.set(
        host_source.data(),
        source,
        host_source.size());

    require(
        block.apply(source, destination),
        label + " rotated block apply");
    const auto block_actual =
        get_vector(product_space, destination, host_source.size());
    for(std::size_t index = 0;
        index < original_eigenvalues.size();
        ++index)
    {
        const double diagonal =
            polynomial_eigenvalues[index] - shift.real();
        require_close(
            block_actual[index],
            diagonal * host_source[index] +
                shift.imag() *
                    host_source[original_eigenvalues.size() + index],
            1.0e-12,
            label + " rotated real block " + std::to_string(index));
        require_close(
            block_actual[original_eigenvalues.size() + index],
            -shift.imag() * host_source[index] +
                diagonal *
                    host_source[original_eigenvalues.size() + index],
            1.0e-12,
            label + " rotated imaginary block " +
                std::to_string(index));
    }
    require(
        block.component_operator_calls() == 2,
        label + " rotated component-operator accounting");

    identity_type identity(product_space);
    solver_type denominator_solver(
        product_space,
        host_source.size(),
        block_matrix);
    mapped_type mapped(
        product_space,
        identity,
        denominator_solver);
    require(
        mapped.apply(source, destination),
        label + " rotated inverse apply");
    const auto mapped_actual =
        get_vector(product_space, destination, host_source.size());
    for(std::size_t index = 0;
        index < original_eigenvalues.size();
        ++index)
    {
        const std::complex<double> rhs(
            host_source[index],
            host_source[original_eigenvalues.size() + index]);
        const std::complex<double> expected =
            rhs /
            (std::complex<double>(polynomial_eigenvalues[index]) -
             shift);
        require_close(
            mapped_actual[index],
            expected.real(),
            1.0e-12,
            label + " rotated inverse real " + std::to_string(index));
        require_close(
            mapped_actual[original_eigenvalues.size() + index],
            expected.imag(),
            1.0e-12,
            label + " rotated inverse imaginary " +
                std::to_string(index));

        const auto mapped_eigenvalue =
            stability::eigensolvers::transformations::
                complex_shift_inverse_map(
                    std::complex<double>(polynomial_eigenvalues[index]),
                    shift);
        const auto recovered =
            stability::eigensolvers::transformations::
                complex_shift_inverse_inverse_map(
                    mapped_eigenvalue,
                    shift);
        require(
            std::abs(
                recovered -
                std::complex<double>(polynomial_eigenvalues[index])) <
                1.0e-12,
            label + " complex shift mapping round trip " +
                std::to_string(index));
    }

    solver_type eigensolver_denominator(
        product_space,
        host_source.size(),
        block_matrix);
    mapped_type eigensolver_mapped(
        product_space,
        identity,
        eigensolver_denominator);
    lapack_type lapack;
    eigensolver_type eigensolver(product_space, lapack);
    typename eigensolver_type::options_type options;
    options.desired_eigenvalues = 4;
    options.krylov_dimension = 7;
    options.restart_dimension = 4;
    options.max_restarts = 20;
    options.absolute_tolerance = 1.0e-12;
    options.relative_tolerance = 1.0e-10;
    options.target.kind =
        stability::eigensolvers::spectrum_target::largest_magnitude;
    options.orthogonalization.reorthogonalization =
        nmfd::solvers::krylov::reorthogonalization_policy::dgks;
    options.orthogonalization.max_passes = 2;

    stability::eigensolvers::ritz_vector_storage<product_space_type>
        transformed_vectors(product_space, 5);
    const auto transformed_result = eigensolver.execute(
        [&eigensolver_mapped](
            const product_vector_type& input,
            product_vector_type& output)
        {
            return eigensolver_mapped.apply(input, output);
        },
        source,
        options,
        &transformed_vectors);
    require(
        transformed_result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " rotated transformed Krylov-Schur: " +
            transformed_result.diagnostic);

    stability::eigensolvers::rotated_projected_spectrum_recovery<
        component_space_type,
        product_space_type,
        lapack_type>
        recovery(component_space, product_space, lapack);
    stability::eigensolvers::ritz_vector_storage<component_space_type>
        physical_vectors(component_space, 8);
    const auto recovered =
        recovery.execute(original, transformed_vectors, {}, &physical_vectors);
    require(
        recovered.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " rotated projected spectrum recovery: " +
            recovered.diagnostic);
    require(
        recovered.projection_dimension == 2,
        label + " rotated projected spectrum dimension");
    require(
        recovered.eigenpairs.size() == 2 &&
            physical_vectors.size() == recovered.eigenpairs.size() &&
            recovered.all_converged(),
        label + " rotated projected physical Ritz recovery");

    const std::vector<double> expected_original{-1.0, 2.0};
    for(const double expected : expected_original)
    {
        const auto nearest = std::min_element(
            recovered.eigenvalues.begin(),
            recovered.eigenvalues.end(),
            [expected](
                const std::complex<double>& lhs,
                const std::complex<double>& rhs)
            {
                return std::abs(lhs - std::complex<double>(expected)) <
                    std::abs(rhs - std::complex<double>(expected));
            });
        require(
            nearest != recovered.eigenvalues.end() &&
            std::abs(*nearest - std::complex<double>(expected)) < 1.0e-8,
            label + " rotated recovered original eigenvalue " +
                std::to_string(expected));
    }

    denominator_solver.fail_after(1);
    require(
        !mapped.apply(source, destination),
        label + " rotated inner failure propagation");

    product_space.stop_use_vector(destination);
    product_space.stop_use_vector(source);
    product_space.free_vector(destination);
    product_space.free_vector(source);
}

} // namespace

int main()
{
    run_cayley_case<scfd::backend::serial_cpu>("serial");
    run_cayley_case<scfd::backend::omp>("OMP");
    run_inexact_exponential_case<scfd::backend::serial_cpu>("serial");
    run_inexact_exponential_case<scfd::backend::omp>("OMP");
    run_rotated_inexact_exponential_case<scfd::backend::serial_cpu>(
        "serial");
    run_rotated_inexact_exponential_case<scfd::backend::omp>("OMP");

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
