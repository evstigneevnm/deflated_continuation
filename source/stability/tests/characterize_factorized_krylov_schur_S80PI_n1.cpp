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
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/scfd_complex_vector_bridge.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/eigensolvers/krylov_schur.h>
#include <stability/eigensolvers/rotated_projected_spectrum_recovery.h>
#include <stability/eigensolvers/transformations/affine_pencil_operator.h>
#include <stability/eigensolvers/transformations/complex_solver_product_adapter.h>
#include <stability/eigensolvers/transformations/euler_polynomial_factorization.h>
#include <stability/eigensolvers/transformations/identity_operator.h>
#include <stability/eigensolvers/transformations/iterative_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/pointwise_diagonal_preconditioner.h>

#include "common/host_csr_operator.h"

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

double real_argument(int argc, char** argv, int index, double fallback)
{
    return argc > index ? std::stod(argv[index]) : fallback;
}

std::size_t size_argument(
    int argc,
    char** argv,
    int index,
    std::size_t fallback)
{
    return argc > index
        ? static_cast<std::size_t>(std::stoul(argv[index]))
        : fallback;
}

std::vector<double> initial_vector(std::size_t dimension)
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

template<class Matrix>
std::vector<typename Matrix::scalar_type> matrix_diagonal(
    const Matrix& matrix)
{
    using scalar_type = typename Matrix::scalar_type;
    const std::size_t dimension =
        static_cast<std::size_t>(matrix.rows());
    std::vector<scalar_type> result(
        dimension,
        scalar_type{});
    for(std::size_t row = 0; row < dimension; ++row)
    {
        for(
            std::size_t at = matrix.row_offsets()[row];
            at < matrix.row_offsets()[row + 1];
            ++at)
        {
            if(
                static_cast<std::size_t>(
                    matrix.column_indices()[at]) == row)
            {
                result[row] = matrix.values()[at];
                break;
            }
        }
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        using backend_type = scfd::backend::omp;
        using component_space_type =
            scfd_vector_operations<backend_type, double>;
        using product_space_type =
            nmfd::operations::two_block_vector_space<
                component_space_type>;
        using product_vector_type =
            typename product_space_type::vector_type;
        using complex_scalar_type =
            common::scfd_backend_ext::complex_t<
                backend_type,
                double>;
        using complex_space_type =
            scfd_vector_operations<
                backend_type,
                complex_scalar_type>;
        using matrix_operator_type =
            stability::tests::host_csr_operator<
                component_space_type>;
        using complex_matrix_operator_type =
            stability::tests::host_csr_operator<
                complex_space_type>;
        using complex_identity_type =
            stability::eigensolvers::transformations::identity_operator<
                complex_space_type>;
        using factor_operator_type =
            stability::eigensolvers::transformations::
                affine_pencil_operator<
                    complex_space_type,
                    complex_matrix_operator_type,
                    complex_identity_type>;
        using log_type = scfd::utils::log_std;
        using monitor_type =
            nmfd::solvers::monitor_krylov<
                complex_space_type,
                log_type>;
        using preconditioner_type =
            stability::eigensolvers::transformations::
                pointwise_diagonal_preconditioner<
                    complex_space_type,
                    factor_operator_type>;
        using gmres_type =
            nmfd::solvers::gmres<
                complex_space_type,
                monitor_type,
                log_type,
                factor_operator_type,
                preconditioner_type>;
        using factor_bundle_type =
            stability::eigensolvers::transformations::
                iterative_factor_solver_bundle<
                    complex_space_type,
                    factor_operator_type,
                    preconditioner_type,
                    gmres_type>;
        using bridge_type =
            nmfd::operations::scfd_complex_vector_bridge<
                product_space_type,
                complex_space_type>;
        using transformed_solver_type =
            stability::eigensolvers::transformations::
                complex_solver_product_adapter<
                    product_space_type,
                    complex_space_type,
                    bridge_type,
                    factor_bundle_type>;
        using lapack_type =
            nmfd::operations::linalg::host_small_dense_lapack<double>;
        using eigensolver_type =
            stability::eigensolvers::krylov_schur<
                product_space_type,
                lapack_type>;

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
        const auto matrix = coordinate.to_host_csr();
        const auto complex_coordinate =
            nmfd::operations::io::read_matrix_market_file<
                complex_scalar_type>(
                    matrix_path(argc, argv));
        const auto complex_matrix =
            complex_coordinate.to_host_csr();
        const std::size_t dimension =
            static_cast<std::size_t>(matrix.rows());

        auto component_space =
            std::make_shared<component_space_type>(dimension);
        auto product_space =
            std::make_shared<product_space_type>(
                *component_space,
                *component_space);
        auto complex_space =
            std::make_shared<complex_space_type>(dimension);
        matrix_operator_type original(
            *component_space,
            matrix,
            nmfd::operations::sparse::host_csr_execution::openmp);
        complex_matrix_operator_type complex_original(
            *complex_space,
            complex_matrix,
            nmfd::operations::sparse::host_csr_execution::openmp);
        complex_identity_type complex_identity(*complex_space);

        constexpr std::size_t power = 3;
        constexpr double total_time = 0.3;
        constexpr double step = total_time / power;
        constexpr double shift_radius = 1.01;
        const double phi = real_argument(
            argc,
            argv,
            2,
            std::acos(-1.0) / 6.5);
        const complex_type shift = std::polar(shift_radius, phi);
        const auto factors =
            stability::eigensolvers::transformations::
                euler_denominator_factors(step, power, shift);

        typename gmres_type::params inner_parameters;
        inner_parameters.basis_size = static_cast<unsigned>(
            size_argument(argc, argv, 7, 160));
        inner_parameters.batch_size =
            std::min(10u, inner_parameters.basis_size);
        inner_parameters.orthogonalization = "mgs";
        inner_parameters.reorthogonalization_policy = "dgks";
        inner_parameters.max_orthogonalization_passes = 2;
        inner_parameters.preconditioner_side = 'L';
        inner_parameters.monitor.rel_tol =
            real_argument(argc, argv, 9, 1.0e-11);
        inner_parameters.monitor.abs_tol = 1.0e-13;
        inner_parameters.monitor.max_iters_num = static_cast<int>(
            size_argument(argc, argv, 8, 4000));
        inner_parameters.monitor.divide_out_norms_by_rel_base = false;

        const auto original_diagonal =
            matrix_diagonal(complex_matrix);
        std::vector<complex_scalar_type> diagonal(dimension);
        nmfd::detail::vector_wrap<
            complex_space_type,
            true,
            true> diagonal_vector(*complex_space);
        using factor_components_type =
            typename factor_bundle_type::components_type;
        auto factor_factory =
            [&](
                const typename factor_bundle_type::factor_type& factor,
                std::size_t)
            {
                const complex_scalar_type operator_scale(
                    factor.operator_scale.real(),
                    factor.operator_scale.imag());
                const complex_scalar_type diagonal_shift(
                    factor.diagonal_shift.real(),
                    factor.diagonal_shift.imag());
                auto factor_operator =
                    std::make_shared<factor_operator_type>(
                        *complex_space,
                        complex_original,
                        complex_identity,
                        operator_scale,
                        diagonal_shift);
                for(std::size_t row = 0; row < dimension; ++row)
                {
                    diagonal[row] =
                        operator_scale * original_diagonal[row] +
                        diagonal_shift;
                }
                complex_space->set(
                    diagonal.data(),
                    *diagonal_vector,
                    dimension);
                auto factor_preconditioner =
                    std::make_shared<preconditioner_type>(
                        *complex_space,
                        *diagonal_vector);
                return factor_components_type{
                    std::move(factor_operator),
                    std::move(factor_preconditioner)};
            };
        factor_bundle_type factor_bundle(
            complex_space,
            factors,
            inner_parameters,
            factor_factory);
        bridge_type bridge(*product_space, *complex_space);
        transformed_solver_type transformed(
            *product_space,
            *complex_space,
            bridge,
            factor_bundle);

        lapack_type lapack;
        eigensolver_type eigensolver(*product_space, lapack);
        typename eigensolver_type::options_type options;
        options.max_restarts = size_argument(argc, argv, 3, 25);
        options.krylov_dimension = size_argument(argc, argv, 4, 50);
        options.restart_dimension = size_argument(argc, argv, 5, 20);
        options.desired_eigenvalues = size_argument(argc, argv, 6, 2);
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

        stability::eigensolvers::ritz_vector_storage<
            product_space_type> transformed_vectors(
                *product_space,
                options.desired_eigenvalues + 1);
        const auto transformed_result = eigensolver.execute(
            [&transformed](
                const product_vector_type& source,
                product_vector_type& destination)
            {
                return transformed.solve(source, destination);
            },
            initial,
            options,
            &transformed_vectors);

        std::cout
            << std::setprecision(17)
            << "case=S80PI_n1"
            << " mode=factorized_complex_euler_shift_inverse"
            << " phi=" << phi
            << " power=" << power
            << " step=" << step
            << " krylov_dimension=" << options.krylov_dimension
            << " restart_dimension=" << options.restart_dimension
            << " desired_eigenvalues=" << options.desired_eigenvalues
            << " inner_basis_size=" << inner_parameters.basis_size
            << " inner_maximum_iterations="
            << inner_parameters.monitor.max_iters_num
            << " inner_relative_tolerance="
            << inner_parameters.monitor.rel_tol
            << " status="
            << stability::eigensolvers::eigensolver_status_name(
                   transformed_result.status)
            << " iterations=" << transformed_result.iterations
            << " restarts=" << transformed_result.restarts
            << " transformed_calls="
            << transformed_result.operator_calls
            << " factor_solves="
            << factor_bundle.factor_solve_calls()
            << " factor_failures=" << transformed.failed_solves()
            << " denominator_operator_calls="
            << complex_original.operator_calls()
            << '\n';
        for(std::size_t index = 0;
            index < factor_bundle.factor_count();
            ++index)
        {
            const auto& factor_state =
                factor_bundle.factor(index);
            const auto& tracked =
                factor_state.tracked_solver();
            const complex_type root =
                complex_type(1.0, 0.0) -
                factor_state.descriptor().diagonal_shift;
            std::cout
                << "factor=" << index
                << " root_real=" << root.real()
                << " root_imag=" << root.imag()
                << " solve_calls="
                << tracked.solve_calls()
                << " failures="
                << tracked.failed_solves()
                << " total_iterations="
                << tracked.total_iterations()
                << " maximum_iterations="
                << tracked.maximum_iterations()
                << " last_residual="
                << tracked.last_residual()
                << " operator_calls="
                << factor_state.linear_operator().operator_calls()
                << " preconditioner_calls="
                << factor_state.preconditioner().apply_calls()
                << '\n';
        }
        for(std::size_t index = 0;
            index < transformed_result.eigenpairs.size();
            ++index)
        {
            const auto& estimate =
                transformed_result.eigenpairs[index];
            std::cout
                << "mapped_eigenpair=" << index
                << " value_real=" << estimate.value.real()
                << " value_imag=" << estimate.value.imag()
                << " residual=" << estimate.residual
                << " converged=" << (estimate.converged ? 1 : 0)
                << '\n';
        }

        if(
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    operator_failure ||
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    invalid_input ||
            transformed_result.status ==
                stability::eigensolvers::eigensolver_status::
                    dense_solver_failure)
        {
            throw std::runtime_error(
                "factorized transformed eigensolver failed structurally: " +
                transformed_result.diagnostic);
        }

        if(transformed_vectors.size() != 0)
        {
            stability::eigensolvers::
                rotated_projected_spectrum_recovery<
                    component_space_type,
                    product_space_type,
                    lapack_type>
                recovery(
                    *component_space,
                    *product_space,
                    lapack);
            const auto recovered = recovery.execute(
                original,
                transformed_vectors);
            std::cout
                << "recovery_status="
                << stability::eigensolvers::eigensolver_status_name(
                       recovered.status)
                << " projection_dimension="
                << recovered.projection_dimension
                << " recovery_operator_calls="
                << recovered.original_operator_calls
                << '\n';
            if(!recovered.succeeded())
            {
                throw std::runtime_error(
                    "projected recovery failed: " +
                    recovered.diagnostic);
            }
            for(std::size_t index = 0;
                index < recovered.eigenvalues.size();
                ++index)
            {
                const auto value = recovered.eigenvalues[index];
                std::cout
                    << "recovered_eigenvalue=" << index
                    << " value_real=" << value.real()
                    << " value_imag=" << value.imag()
                    << '\n';
            }
        }

        product_space->stop_use_vector(initial);
        product_space->free_vector(initial);
        std::cout
            << "S80PI_n1 factorized transformed characterization: "
            << "COMPLETED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr
            << "S80PI_n1 factorized transformed characterization: "
            << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
