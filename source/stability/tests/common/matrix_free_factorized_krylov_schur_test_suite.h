#ifndef __STABILITY_TESTS_COMMON_MATRIX_FREE_FACTORIZED_KRYLOV_SCHUR_TEST_SUITE_H__
#define __STABILITY_TESTS_COMMON_MATRIX_FREE_FACTORIZED_KRYLOV_SCHUR_TEST_SUITE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/utils/log.h>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <stability/analysis/matrix_free_stability_configuration.h>
#include <stability/analysis/matrix_free_stability_scan.h>
#include <stability/analysis/matrix_free_stability_solver.h>
#include <stability/analysis/spectrum_classifier.h>
#include <stability/eigensolvers/matrix_free_factorized_krylov_schur.h>
#include <stability/eigensolvers/transformations/euler_polynomial_factorization.h>
#include <stability/eigensolvers/transformations/matrix_free_complex_factor_solver_bundle.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <symmetry/linearization/projected_affine_inverse_provider.h>
#include <symmetry/linearization/projected_linear_operator.h>

#include "analytical_dense_operator.h"
#include "analytical_eigenproblem.h"
#include "analytical_real_affine_inverse_model.h"

namespace stability
{
namespace tests
{
namespace matrix_free_factorized_krylov_schur_test
{

inline std::size_t checks = 0;
inline std::size_t failures = 0;

inline void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template<class Problem>
std::vector<typename Problem::real_type> matrix_diagonal(
    const Problem& problem)
{
    std::vector<typename Problem::real_type> diagonal(
        problem.dimension());
    for(std::size_t index = 0; index < diagonal.size(); ++index)
        diagonal[index] = problem.matrix(index, index);
    return diagonal;
}

template<class VectorSpace, class Operator>
class coordinate_quotient_model
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    coordinate_quotient_model(
        VectorSpace& vector_space,
        const Operator& linear_operator,
        std::size_t gauge_index)
        : vector_space_(vector_space),
          linear_operator_(linear_operator),
          gauge_index_(gauge_index),
          host_(vector_space.get_default_size())
    {
        if(gauge_index_ >= host_.size())
            throw std::out_of_range(
                "coordinate quotient gauge index");
    }

    VectorSpace* get_vec_ops_ref()
    {
        return &vector_space_;
    }

    void project_current_tangent(
        const vector_type& source,
        vector_type& destination) const
    {
        vector_space_.get(
            source,
            host_.data(),
            host_.size());
        host_[gauge_index_] = scalar_type{};
        vector_space_.set(
            host_.data(),
            destination,
            host_.size());
    }

    void projected_jacobian_u(
        const vector_type& source,
        vector_type& destination) const
    {
        linear_operator_.apply(source, destination);
        project_current_tangent(destination, destination);
    }

private:
    VectorSpace& vector_space_;
    const Operator& linear_operator_;
    std::size_t gauge_index_;
    mutable std::vector<scalar_type> host_;
};

template<class Driver>
typename Driver::options_type recovery_options()
{
    typename Driver::options_type options;
    options.transformed.desired_eigenvalues = 2;
    options.transformed.krylov_dimension = 6;
    options.transformed.restart_dimension = 3;
    options.transformed.max_restarts = 4;
    options.transformed.absolute_tolerance = 1.0e-11;
    options.transformed.relative_tolerance = 1.0e-10;
    options.transformed.preserve_conjugate_pairs = true;
    options.transformed.target.kind =
        stability::eigensolvers::spectrum_target::largest_magnitude;
    options.transformed.orthogonalization.method =
        nmfd::solvers::krylov::orthogonalization_method::
            modified_gram_schmidt;
    options.transformed.orthogonalization.reorthogonalization =
        nmfd::solvers::krylov::reorthogonalization_policy::dgks;
    options.transformed.orthogonalization.max_passes = 2;
    options.recovery.absolute_residual_tolerance = 1.0e-9;
    options.recovery.relative_residual_tolerance = 1.0e-9;
    return options;
}

template<class Backend>
void test_complex_pair_recovery(const std::string& label)
{
    using real_type = double;
    using real_space_type =
        scfd_vector_operations<Backend, real_type>;
    using complex_scalar_type =
        common::scfd_backend_ext::complex_t<
            Backend,
            real_type>;
    using complex_space_type =
        scfd_vector_operations<
            Backend,
            complex_scalar_type>;
    using real_operator_type =
        analytical_dense_operator<
            real_space_type,
            real_type>;
    using model_type =
        analytical_real_affine_inverse_model<
            real_space_type>;
    using provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                real_space_type,
                model_type>;
    using factorization_types =
        stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                real_space_type,
                complex_space_type,
                real_operator_type,
                provider_type>;
    using factor_operator_type =
        typename factorization_types::factor_operator_type;
    using preconditioner_type =
        typename factorization_types::preconditioner_type;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_type>;
    using inner_solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;
    using factor_bundle_type =
        stability::eigensolvers::transformations::
            matrix_free_complex_factor_solver_bundle<
                factorization_types,
                inner_solver_type>;
    using dense_lapack_type =
        nmfd::operations::linalg::
            host_small_dense_lapack<real_type>;
    using driver_type =
        stability::eigensolvers::
            matrix_free_factorized_krylov_schur<
                factor_bundle_type,
                dense_lapack_type>;

    const auto problem =
        complex_pair_eigenproblem<real_type>();
    auto real_space =
        std::make_shared<real_space_type>(problem.dimension());
    auto complex_space =
        std::make_shared<complex_space_type>(problem.dimension());
    real_operator_type real_operator(*real_space, problem);
    auto model = std::make_shared<model_type>(
        *real_space,
        matrix_diagonal(problem));
    auto provider =
        std::make_shared<provider_type>(
            *real_space,
            *model);

    constexpr real_type step = 0.1;
    constexpr std::size_t repetitions = 3;
    const std::complex<real_type> mapped_target =
        std::pow(
            std::complex<real_type>(1.0, 0.0) +
                step*std::complex<real_type>(0.0, 2.0),
            repetitions);
    const std::complex<real_type> shift =
        mapped_target +
        std::complex<real_type>(0.04, 0.025);
    const auto factors =
        stability::eigensolvers::transformations::
            euler_denominator_factors(
                step,
                repetitions,
                shift);

    typename inner_solver_type::params inner_parameters;
    inner_parameters.basis_size = 3;
    inner_parameters.batch_size = 3;
    inner_parameters.preconditioner_side = 'L';
    inner_parameters.orthogonalization = "mgs";
    inner_parameters.reorthogonalization_policy = "dgks";
    inner_parameters.max_orthogonalization_passes = 2;
    inner_parameters.monitor.rel_tol = 1.0e-12;
    inner_parameters.monitor.abs_tol = 1.0e-13;
    inner_parameters.monitor.max_iters_num = 20;
    inner_parameters.monitor.divide_out_norms_by_rel_base = false;

    factor_bundle_type factor_bundle(
        real_space,
        complex_space,
        real_operator,
        provider,
        factors,
        inner_parameters);
    dense_lapack_type dense_lapack;
    driver_type driver(factor_bundle, dense_lapack);

    nmfd::detail::vector_wrap<
        real_space_type,
        true,
        true> initial(*real_space);
    const std::vector<real_type> host_initial{
        1.0,
        0.35,
        -0.2};
    real_space->set(
        host_initial.data(),
        *initial,
        host_initial.size());
    stability::eigensolvers::ritz_vector_storage<
        real_space_type> recovered_vectors(
            *real_space,
            problem.dimension());

    const auto result = driver.execute(
        *initial,
        recovery_options<driver_type>(),
        &recovered_vectors);

    require(
        result.transformed.succeeded(),
        label + " transformed Krylov-Schur status: " +
            result.transformed.diagnostic);
    require(
        result.recovered.succeeded(),
        label + " projected recovery status: " +
            result.recovered.diagnostic);
    require(
        result.succeeded(),
        label + " complete matrix-free recovery");
    require(
        result.transformed_solver_calls ==
            result.transformed.operator_calls &&
        result.transformed_solver_failures == 0 &&
        result.transformed.inner_solver_calls ==
            result.transformed_solver_calls,
        label + " transformed solve accounting");
    require(
        recovered_vectors.size() ==
            result.recovered.eigenpairs.size() &&
        result.recovered.projection_dimension >= 2,
        label + " physical Ritz-vector recovery");

    for(const std::complex<real_type> expected :
        {std::complex<real_type>(0.0, 2.0),
         std::complex<real_type>(0.0, -2.0)})
    {
        const auto nearest = std::min_element(
            result.recovered.eigenpairs.begin(),
            result.recovered.eigenpairs.end(),
            [expected](const auto& left, const auto& right)
            {
                return
                    std::abs(left.value - expected) <
                    std::abs(right.value - expected);
            });
        require(
            nearest != result.recovered.eigenpairs.end() &&
            std::abs(nearest->value - expected) <= 2.0e-9 &&
            nearest->relative_residual <= 2.0e-9,
            label + " recovered physical eigenvalue");
    }

    const auto statistics = factor_bundle.statistics();
    require(
        statistics.solve_calls ==
            result.transformed_solver_calls &&
        statistics.factor_solve_calls ==
            factors.size()*result.transformed_solver_calls &&
        statistics.failed_solves == 0,
        label + " factor bundle accounting");
    require(
        factor_bundle.complexified_operator().operator_calls() > 0 &&
        factor_bundle.complexified_operator().
            component_operator_calls() ==
            2*factor_bundle.complexified_operator().
                operator_calls(),
        label + " matrix-free real Jacobian complexification");
    require(
        provider->apply_calls() > 0 &&
        provider->failed_applications() == 0 &&
        model->apply_calls() == provider->apply_calls(),
        label + " real-only affine preconditioner boundary");

    auto rejecting_options =
        recovery_options<driver_type>();
    rejecting_options.recovery.absolute_residual_tolerance =
        real_type(0);
    rejecting_options.recovery.relative_residual_tolerance =
        real_type(0);
    const auto rejected = driver.execute(
        *initial,
        rejecting_options);
    require(
        !rejected.succeeded() &&
            rejected.recovered.diagnostic.find(
                "rejected physical Ritz values") !=
                std::string::npos,
        label + " rejected physical Ritz residual diagnostics: " +
            rejected.recovered.diagnostic);

    using stability_assembly_type =
        stability::analysis::matrix_free_stability_assembly<
            factorization_types,
            inner_solver_type,
            dense_lapack_type>;
    stability_assembly_type stability_assembly(
        real_space,
        complex_space,
        real_operator,
        provider,
        factors,
        inner_parameters,
        recovery_options<driver_type>());
    const auto structured_result =
        stability_assembly.execute(*initial);
    require(
        structured_result.succeeded(),
        label + " structured matrix-free stability status: " +
            structured_result.diagnostic);
    require(
        structured_result.inner_solver_calls > 0 &&
        structured_result.operator_calls >=
            structured_result.inner_solver_calls &&
        structured_result.effective_subspace_dimension >= 2,
        label + " structured matrix-free accounting");

    stability::analysis::spectrum_classifier<real_type>
        classifier;
    const auto classified =
        classifier.classify(structured_result);
    require(
        classified.succeeded() &&
        classified.unstable.real == 0 &&
        classified.unstable.complex_pairs == 0 &&
        classified.neutral_complex_pairs == 1,
        label + " matrix-free physical spectrum classification");

    using scan_type =
        stability::analysis::matrix_free_stability_scan<
            factorization_types,
            inner_solver_type,
            dense_lapack_type>;
    stability::analysis::matrix_free_stability_config<
        real_type> scan_config;
    scan_config.enabled = true;
    scan_config.transformation.type =
        stability::analysis::
            matrix_free_spectral_transformation::
                explicit_euler;
    scan_config.transformation.step = step;
    scan_config.transformation.repetitions = repetitions;
    scan_config.transformation.shifts = {
        shift,
        shift + std::complex<real_type>(-0.02, 0.03)};
    scan_config.outer.desired_eigenvalues = 2;
    scan_config.outer.krylov_dimension = 6;
    scan_config.outer.restart_dimension = 3;
    scan_config.outer.maximum_restarts = 4;
    scan_config.outer.absolute_tolerance = 1.0e-11;
    scan_config.outer.relative_tolerance = 1.0e-10;
    scan_config.outer.breakdown_relative_tolerance = 1.0e-12;
    scan_config.recovery.absolute_residual_tolerance = 1.0e-9;
    scan_config.recovery.relative_residual_tolerance = 1.0e-9;
    scan_config.inner_solver.basis_size = 3;
    scan_config.inner_solver.basis_retry_sizes = {5};
    scan_config.inner_solver.batch_size = 3;
    scan_config.inner_solver.preconditioner_side = 'R';
    scan_config.inner_solver.relative_tolerance = 1.0e-12;
    scan_config.inner_solver.absolute_tolerance = 1.0e-13;
    scan_config.inner_solver.maximum_iterations = 20;
    scan_config.inner_solver.divide_norms_by_relative_base = false;
    scan_config.aggregation.minimum_successful_scans = 2;
    scan_config.aggregation.require_all_scans = true;

    const auto scans =
        stability::analysis::make_matrix_free_stability_scans<
            scan_type>(scan_config);
    const auto scan_inner_parameters =
        stability::analysis::
            make_matrix_free_inner_solver_parameters<
                typename scan_type::inner_parameters_type>(
                    scan_config);
    require(
        scan_inner_parameters.preconditioner_side == 'R',
        label + " scan uses true-residual right preconditioning");
    require(
        scans.size() == 2 &&
            scans.front().inner_basis_retry_sizes ==
                std::vector<unsigned int>({5}),
        label + " scan preserves inner basis retry policy");
    scan_type scan_driver(
        real_space,
        complex_space,
        real_operator,
        provider,
        scans,
        scan_inner_parameters,
        stability::analysis::
            make_spectrum_scan_aggregation_options(scan_config));
    const auto scanned = scan_driver.execute(*initial);
    require(
        scanned.succeeded() &&
            scanned.coverage_complete &&
            scanned.scans_requested == 2 &&
            scanned.scans_succeeded == 2,
        label + " sequential spectral scan status: " +
            scanned.diagnostic);
    const auto scan_classified = classifier.classify(scanned);
    require(
        scan_classified.succeeded() &&
            scan_classified.neutral_complex_pairs == 1,
        label + " aggregated physical spectrum classification");
}

template<class Backend>
void test_projected_quotient_recovery(const std::string& label)
{
    using real_type = double;
    using real_space_type =
        scfd_vector_operations<Backend, real_type>;
    using complex_scalar_type =
        common::scfd_backend_ext::complex_t<
            Backend,
            real_type>;
    using complex_space_type =
        scfd_vector_operations<
            Backend,
            complex_scalar_type>;
    using base_operator_type =
        analytical_dense_operator<
            real_space_type,
            real_type>;
    using quotient_model_type =
        coordinate_quotient_model<
            real_space_type,
            base_operator_type>;
    using real_operator_type =
        symmetry::linearization::projected_linear_operator<
            real_space_type,
            quotient_model_type>;
    using affine_model_type =
        analytical_real_affine_inverse_model<
            real_space_type>;
    using base_provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                real_space_type,
                affine_model_type>;
    using provider_type =
        symmetry::linearization::
            projected_affine_inverse_provider<
                real_space_type,
                quotient_model_type,
                base_provider_type>;
    using factorization_types =
        stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                real_space_type,
                complex_space_type,
                real_operator_type,
                provider_type>;
    using factor_operator_type =
        typename factorization_types::factor_operator_type;
    using preconditioner_type =
        typename factorization_types::preconditioner_type;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_type>;
    using inner_solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;
    using factor_bundle_type =
        stability::eigensolvers::transformations::
            matrix_free_complex_factor_solver_bundle<
                factorization_types,
                inner_solver_type>;
    using dense_lapack_type =
        nmfd::operations::linalg::
            host_small_dense_lapack<real_type>;
    using stability_assembly_type =
        stability::analysis::matrix_free_stability_assembly<
            factorization_types,
            inner_solver_type,
            dense_lapack_type>;

    const auto problem = diagonal_eigenproblem<real_type>();
    auto real_space =
        std::make_shared<real_space_type>(problem.dimension());
    auto complex_space =
        std::make_shared<complex_space_type>(problem.dimension());
    base_operator_type base_operator(*real_space, problem);
    quotient_model_type quotient_model(
        *real_space,
        base_operator,
        problem.dimension() - 1);
    real_operator_type quotient_operator(
        &quotient_model,
        real_type(0));
    auto affine_model =
        std::make_shared<affine_model_type>(
            *real_space,
            matrix_diagonal(problem));
    auto base_provider =
        std::make_shared<base_provider_type>(
            *real_space,
            *affine_model);
    auto provider =
        std::make_shared<provider_type>(
            *real_space,
            quotient_model,
            base_provider,
            real_type(0));

    constexpr real_type step = 0.1;
    constexpr std::size_t repetitions = 3;
    const std::complex<real_type> mapped_target =
        std::pow(
            std::complex<real_type>(1.0, 0.0) +
                step*std::complex<real_type>(2.0, 0.0),
            repetitions);
    const auto factors =
        stability::eigensolvers::transformations::
            euler_denominator_factors(
                step,
                repetitions,
                mapped_target +
                    std::complex<real_type>(0.04, 0.025));

    typename inner_solver_type::params inner_parameters;
    inner_parameters.basis_size = 4;
    inner_parameters.batch_size = 4;
    inner_parameters.preconditioner_side = 'L';
    inner_parameters.orthogonalization = "mgs";
    inner_parameters.reorthogonalization_policy = "dgks";
    inner_parameters.max_orthogonalization_passes = 2;
    inner_parameters.monitor.rel_tol = 1.0e-12;
    inner_parameters.monitor.abs_tol = 1.0e-13;
    inner_parameters.monitor.max_iters_num = 24;
    inner_parameters.monitor.divide_out_norms_by_rel_base = false;

    using raw_driver_type =
        typename stability_assembly_type::solver_type::
            eigensolver_type;
    auto options = recovery_options<raw_driver_type>();
    options.transformed.krylov_dimension = 7;
    options.transformed.restart_dimension = 4;
    stability_assembly_type stability_assembly(
        real_space,
        complex_space,
        quotient_operator,
        provider,
        factors,
        inner_parameters,
        options);

    nmfd::detail::vector_wrap<
        real_space_type,
        true,
        true> initial(*real_space);
    const std::vector<real_type> host_initial{
        0.4,
        -0.7,
        1.0,
        0.8};
    real_space->set(
        host_initial.data(),
        *initial,
        host_initial.size());
    const auto result = stability_assembly.execute(*initial);
    require(
        result.succeeded(),
        label + " projected quotient stability status: " +
            result.diagnostic);

    const auto target = std::min_element(
        result.eigenpairs.begin(),
        result.eigenpairs.end(),
        [](const auto& left, const auto& right)
        {
            return
                std::abs(left.value - std::complex<real_type>(2.0, 0.0)) <
                std::abs(right.value - std::complex<real_type>(2.0, 0.0));
        });
    require(
        target != result.eigenpairs.end() &&
        std::abs(
            target->value -
            std::complex<real_type>(2.0, 0.0)) <= 2.0e-9,
        label + " projected quotient recovers physical unstable mode");
    const bool contains_identity_completion =
        std::any_of(
            result.eigenpairs.begin(),
            result.eigenpairs.end(),
            [](const auto& estimate)
            {
                return std::abs(
                    estimate.value -
                    std::complex<real_type>(1.0, 0.0)) <= 1.0e-8;
            });
    require(
        !contains_identity_completion,
        label + " projected stability excludes Newton gauge completion");

    stability::analysis::spectrum_classifier<real_type>
        classifier;
    const auto classified = classifier.classify(result);
    require(
        classified.succeeded() &&
        classified.unstable.real == 1,
        label + " projected quotient unstable dimension");
}

template<class Backend>
void test_inner_failure(const std::string& label)
{
    using real_type = double;
    using real_space_type =
        scfd_vector_operations<Backend, real_type>;
    using complex_scalar_type =
        common::scfd_backend_ext::complex_t<
            Backend,
            real_type>;
    using complex_space_type =
        scfd_vector_operations<
            Backend,
            complex_scalar_type>;
    using real_operator_type =
        analytical_dense_operator<
            real_space_type,
            real_type>;
    using model_type =
        analytical_real_affine_inverse_model<
            real_space_type>;
    using provider_type =
        stability::eigensolvers::transformations::
            nonlinear_operator_real_affine_inverse_provider<
                real_space_type,
                model_type>;
    using factorization_types =
        stability::eigensolvers::transformations::
            matrix_free_complex_factorization_types<
                real_space_type,
                complex_space_type,
                real_operator_type,
                provider_type>;
    using factor_operator_type =
        typename factorization_types::factor_operator_type;
    using preconditioner_type =
        typename factorization_types::preconditioner_type;
    using log_type = scfd::utils::log_std;
    using monitor_type =
        nmfd::solvers::monitor_krylov<
            complex_space_type,
            log_type>;
    using inner_solver_type =
        nmfd::solvers::gmres<
            complex_space_type,
            monitor_type,
            log_type,
            factor_operator_type,
            preconditioner_type>;
    using factor_bundle_type =
        stability::eigensolvers::transformations::
            matrix_free_complex_factor_solver_bundle<
                factorization_types,
                inner_solver_type>;
    using dense_lapack_type =
        nmfd::operations::linalg::
            host_small_dense_lapack<real_type>;
    using driver_type =
        stability::eigensolvers::
            matrix_free_factorized_krylov_schur<
                factor_bundle_type,
                dense_lapack_type>;

    const auto problem =
        complex_pair_eigenproblem<real_type>();
    auto real_space =
        std::make_shared<real_space_type>(problem.dimension());
    auto complex_space =
        std::make_shared<complex_space_type>(problem.dimension());
    real_operator_type real_operator(*real_space, problem);
    real_operator.fail_after(0);
    auto model = std::make_shared<model_type>(
        *real_space,
        matrix_diagonal(problem));
    auto provider =
        std::make_shared<provider_type>(
            *real_space,
            *model);
    const auto factors =
        stability::eigensolvers::transformations::
            euler_denominator_factors(
                real_type(0.1),
                std::size_t(3),
                std::complex<real_type>(0.9, 0.5));

    typename inner_solver_type::params inner_parameters;
    inner_parameters.basis_size = 3;
    inner_parameters.batch_size = 3;
    inner_parameters.monitor.rel_tol = 1.0e-10;
    inner_parameters.monitor.abs_tol = 1.0e-12;
    inner_parameters.monitor.max_iters_num = 8;
    inner_parameters.monitor.divide_out_norms_by_rel_base = false;

    factor_bundle_type factor_bundle(
        real_space,
        complex_space,
        real_operator,
        provider,
        factors,
        inner_parameters);
    dense_lapack_type dense_lapack;
    driver_type driver(factor_bundle, dense_lapack);
    nmfd::detail::vector_wrap<
        real_space_type,
        true,
        true> initial(*real_space);
    const std::vector<real_type> host_initial{
        1.0,
        0.25,
        -0.5};
    real_space->set(
        host_initial.data(),
        *initial,
        host_initial.size());

    const auto result = driver.execute(
        *initial,
        recovery_options<driver_type>());
    require(
        result.transformed.status ==
            stability::eigensolvers::eigensolver_status::
                inner_solver_failure,
        label + " inner failure status");
    require(
        result.transformed_solver_calls == 1 &&
        result.transformed_solver_failures == 1 &&
        result.transformed.inner_solver_calls == 1,
        label + " inner failure accounting");
    require(
        result.recovered.eigenpairs.empty() &&
        !result.succeeded(),
        label + " failed transform skips physical recovery");
}

template<class Backend>
void run_backend(const std::string& label)
{
    test_complex_pair_recovery<Backend>(label);
    test_projected_quotient_recovery<Backend>(label);
    test_inner_failure<Backend>(label);
}

inline int finish()
{
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

} // namespace matrix_free_factorized_krylov_schur_test
} // namespace tests
} // namespace stability

#endif
