#ifndef __STABILITY_TESTS_COMMON_PROJECTED_SPECTRUM_RECOVERY_TEST_SUITE_H__
#define __STABILITY_TESTS_COMMON_PROJECTED_SPECTRUM_RECOVERY_TEST_SUITE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <common/scfd_vector_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <stability/eigensolvers/projected_spectrum_recovery.h>

#include "analytical_dense_operator.h"
#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{
namespace projected_spectrum_recovery_test
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

inline void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message)
{
    require(
        std::abs(actual - expected) <= tolerance,
        message + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

template<class VectorSpace>
void set_coordinate_candidates(
    const VectorSpace& vector_space,
    std::size_t dimension,
    std::size_t candidate_count,
    stability::eigensolvers::ritz_vector_storage<VectorSpace>& candidates)
{
    using scalar_type = typename VectorSpace::scalar_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    nmfd::detail::vector_wrap<VectorSpace, true, true> coordinate(
        vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> zero(vector_space);
    vector_space.assign_scalar(scalar_type{}, *zero);

    std::vector<scalar_type> host(dimension, scalar_type{});
    for(std::size_t column = 0; column < candidate_count; ++column)
    {
        std::fill(host.begin(), host.end(), scalar_type{});
        host[column] = scalar_type(1);
        vector_space.set(host.data(), *coordinate, host.size());
        vector_space.assign(
            *coordinate,
            candidates.real(),
            static_cast<ordinal_type>(candidates.capacity()),
            static_cast<ordinal_type>(column));
        vector_space.assign(
            *zero,
            candidates.imaginary(),
            static_cast<ordinal_type>(candidates.capacity()),
            static_cast<ordinal_type>(column));
    }
    candidates.set_size(candidate_count);
}

template<class VectorSpace, class Operator>
double physical_residual(
    const VectorSpace& vector_space,
    Operator& matrix_operator,
    const stability::eigensolvers::ritz_vector_storage<VectorSpace>& vectors,
    std::size_t index,
    std::complex<double> eigenvalue)
{
    using scalar_type = typename VectorSpace::scalar_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    nmfd::detail::vector_wrap<VectorSpace, true, true> real(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> imaginary(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> applied_real(
        vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> applied_imaginary(
        vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> residual_real(
        vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> residual_imaginary(
        vector_space);

    vector_space.assign(
        vectors.real(),
        static_cast<ordinal_type>(vectors.capacity()),
        static_cast<ordinal_type>(index),
        *real);
    vector_space.assign(
        vectors.imaginary(),
        static_cast<ordinal_type>(vectors.capacity()),
        static_cast<ordinal_type>(index),
        *imaginary);
    require(matrix_operator.apply(*real, *applied_real), "validation A*Re(v)");
    require(
        matrix_operator.apply(*imaginary, *applied_imaginary),
        "validation A*Im(v)");

    vector_space.assign(*applied_real, *residual_real);
    vector_space.add_lin_comb(
        static_cast<scalar_type>(-eigenvalue.real()),
        *real,
        scalar_type(1),
        *residual_real);
    vector_space.add_lin_comb(
        static_cast<scalar_type>(eigenvalue.imag()),
        *imaginary,
        scalar_type(1),
        *residual_real);

    vector_space.assign(*applied_imaginary, *residual_imaginary);
    vector_space.add_lin_comb(
        static_cast<scalar_type>(-eigenvalue.imag()),
        *real,
        scalar_type(1),
        *residual_imaginary);
    vector_space.add_lin_comb(
        static_cast<scalar_type>(-eigenvalue.real()),
        *imaginary,
        scalar_type(1),
        *residual_imaginary);

    return std::sqrt(
        vector_space.norm_sq(*residual_real) +
        vector_space.norm_sq(*residual_imaginary));
}

template<class Backend>
void test_complex_pair_recovery(const std::string& label)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using recovery_type =
        stability::eigensolvers::projected_spectrum_recovery<
            vector_space_type,
            lapack_type>;

    const auto problem =
        stability::tests::complex_pair_eigenproblem<double>();
    vector_space_type vector_space(problem.dimension());
    operator_type matrix_operator(vector_space, problem);
    lapack_type lapack;
    recovery_type recovery(vector_space, lapack);

    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        transformed_vectors(vector_space, problem.dimension());
    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        physical_vectors(vector_space, problem.dimension());
    set_coordinate_candidates(
        vector_space,
        problem.dimension(),
        problem.dimension(),
        transformed_vectors);

    typename recovery_type::options_type options;
    options.absolute_residual_tolerance = 1.0e-12;
    options.relative_residual_tolerance = 1.0e-11;

    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        insufficient_vectors(vector_space, problem.dimension() - 1);
    const auto insufficient_result = recovery.execute(
        matrix_operator,
        transformed_vectors,
        options,
        &insufficient_vectors);
    require(
        insufficient_result.status ==
            stability::eigensolvers::eigensolver_status::invalid_input,
        label + " rejects insufficient physical-vector capacity");
    require(
        matrix_operator.operator_calls() == 0,
        label + " validates output capacity before operator application");

    const auto result = recovery.execute(
        matrix_operator,
        transformed_vectors,
        options,
        &physical_vectors);

    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " recovery status: " + result.diagnostic);
    require(
        result.projection_dimension == problem.dimension(),
        label + " projection dimension");
    require(
        result.original_operator_calls == problem.dimension(),
        label + " one original-operator call per basis vector");
    require(
        matrix_operator.operator_calls() == problem.dimension(),
        label + " residuals reuse stored basis images");
    require(
        result.eigenpairs.size() == problem.dimension(),
        label + " eigenpair count");
    require(
        result.eigenvalues.size() == result.eigenpairs.size(),
        label + " compatibility eigenvalue count");
    require(
        physical_vectors.size() == result.eigenpairs.size(),
        label + " physical vector count");
    require(result.all_converged(), label + " all eigenpairs converged");

    for(const auto& expected : problem.eigenpairs())
    {
        const auto nearest = std::min_element(
            result.eigenpairs.begin(),
            result.eigenpairs.end(),
            [&expected](const auto& left, const auto& right)
            {
                return std::abs(left.value - expected.value) <
                    std::abs(right.value - expected.value);
            });
        require(
            nearest != result.eigenpairs.end() &&
                std::abs(nearest->value - expected.value) <= 1.0e-12,
            label + " expected eigenvalue");
    }

    for(std::size_t index = 0; index < result.eigenpairs.size(); ++index)
    {
        const auto& estimate = result.eigenpairs[index];
        require(estimate.converged, label + " per-pair convergence");
        require(
            estimate.residual <= 1.0e-12,
            label + " absolute physical residual");
        require(
            estimate.relative_residual <= 1.0e-12,
            label + " relative physical residual");
        const double independently_computed = physical_residual(
            vector_space,
            matrix_operator,
            physical_vectors,
            index,
            estimate.value);
        require_close(
            estimate.residual,
            independently_computed,
            1.0e-12,
            label + " reported physical residual");
    }
}

template<class Backend>
void test_noninvariant_subspace(const std::string& label)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using recovery_type =
        stability::eigensolvers::projected_spectrum_recovery<
            vector_space_type,
            lapack_type>;

    const stability::tests::analytical_eigenproblem<double> problem(
        "noninvariant_one_dimensional_trial_space",
        2,
        {1.0, 0.25, 0.0, 2.0},
        {});
    vector_space_type vector_space(problem.dimension());
    operator_type matrix_operator(vector_space, problem);
    lapack_type lapack;
    recovery_type recovery(vector_space, lapack);

    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        transformed_vectors(vector_space, 1);
    set_coordinate_candidates(
        vector_space,
        problem.dimension(),
        1,
        transformed_vectors);

    typename recovery_type::options_type options;
    options.absolute_residual_tolerance = 1.0e-13;
    options.relative_residual_tolerance = 1.0e-13;
    const auto result =
        recovery.execute(matrix_operator, transformed_vectors, options);

    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::success,
        label + " noninvariant recovery completed");
    require(
        result.eigenpairs.size() == 1,
        label + " noninvariant eigenpair count");
    require_close(
        result.eigenpairs.front().value.real(),
        1.0,
        1.0e-14,
        label + " projected eigenvalue");
    require_close(
        result.eigenpairs.front().residual,
        0.25,
        1.0e-14,
        label + " out-of-subspace residual");
    require(
        result.eigenpairs.front().relative_residual > 0.1,
        label + " out-of-subspace relative residual");
    require(
        !result.eigenpairs.front().converged && !result.all_converged(),
        label + " rejects a noninvariant trial subspace");
    require(
        result.original_operator_calls == 1 &&
            matrix_operator.operator_calls() == 1,
        label + " noninvariant operator-call count");
}

template<class Backend>
void test_operator_failure(const std::string& label)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using operator_type =
        stability::tests::analytical_dense_operator<
            vector_space_type,
            double>;
    using lapack_type =
        nmfd::operations::linalg::host_small_dense_lapack<double>;
    using recovery_type =
        stability::eigensolvers::projected_spectrum_recovery<
            vector_space_type,
            lapack_type>;

    const auto problem =
        stability::tests::symmetric_eigenproblem<double>();
    vector_space_type vector_space(problem.dimension());
    operator_type matrix_operator(vector_space, problem);
    matrix_operator.fail_after(0);
    lapack_type lapack;
    recovery_type recovery(vector_space, lapack);
    stability::eigensolvers::ritz_vector_storage<vector_space_type>
        transformed_vectors(vector_space, problem.dimension());
    set_coordinate_candidates(
        vector_space,
        problem.dimension(),
        problem.dimension(),
        transformed_vectors);

    const auto result =
        recovery.execute(matrix_operator, transformed_vectors);
    require(
        result.status ==
            stability::eigensolvers::eigensolver_status::operator_failure,
        label + " operator failure propagation");
    require(
        result.original_operator_calls == 1,
        label + " failed operator-call accounting");
}

template<class Backend>
void run_backend(const std::string& label)
{
    test_complex_pair_recovery<Backend>(label);
    test_noninvariant_subspace<Backend>(label);
    test_operator_failure<Backend>(label);
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

} // namespace projected_spectrum_recovery_test
} // namespace tests
} // namespace stability

#endif
