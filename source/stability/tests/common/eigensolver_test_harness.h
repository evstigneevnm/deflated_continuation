#ifndef __STABILITY_TESTS_COMMON_EIGENSOLVER_TEST_HARNESS_H__
#define __STABILITY_TESTS_COMMON_EIGENSOLVER_TEST_HARNESS_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{

enum class eigensolver_status
{
    success,
    no_convergence,
    operator_failure,
    numerical_breakdown,
    invalid_input
};

inline const char* eigensolver_status_name(eigensolver_status status)
{
    switch(status)
    {
    case eigensolver_status::success:
        return "success";
    case eigensolver_status::no_convergence:
        return "no_convergence";
    case eigensolver_status::operator_failure:
        return "operator_failure";
    case eigensolver_status::numerical_breakdown:
        return "numerical_breakdown";
    case eigensolver_status::invalid_input:
        return "invalid_input";
    }
    return "unknown";
}

template<class Real>
struct computed_eigenpair
{
    using complex_type = std::complex<Real>;

    complex_type value{};
    std::vector<complex_type> right_eigenvector;
    Real reported_residual = std::numeric_limits<Real>::quiet_NaN();
};

template<class Real>
struct eigensolver_result
{
    eigensolver_status status = eigensolver_status::invalid_input;
    std::vector<computed_eigenpair<Real>> eigenpairs;
    std::size_t iterations = 0;
    std::size_t restarts = 0;
    std::size_t operator_calls = 0;
    std::string diagnostic;
};

template<class Real>
struct eigensolver_test_tolerances
{
    Real eigenvalue_absolute = Real(100)*std::numeric_limits<Real>::epsilon();
    Real eigenvalue_relative = Real(1000)*std::numeric_limits<Real>::epsilon();
    Real residual_relative = Real(1000)*std::numeric_limits<Real>::epsilon();
    Real eigenvector_alignment = Real(1000)*std::numeric_limits<Real>::epsilon();
    bool require_eigenvectors = true;
};

struct eigensolver_validation_report
{
    std::size_t checks = 0;
    std::vector<std::string> failures;

    bool passed() const
    {
        return failures.empty();
    }

    void require(bool condition, std::string message)
    {
        ++checks;
        if(!condition)
            failures.emplace_back(std::move(message));
    }
};

namespace detail
{

template<class Real>
bool finite(const std::complex<Real>& value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

template<class Real>
Real vector_norm(const std::vector<std::complex<Real>>& vector)
{
    Real norm_sq = Real{};
    for(const auto& value : vector)
        norm_sq += std::norm(value);
    using std::sqrt;
    return sqrt(norm_sq);
}

template<class Real>
std::complex<Real> inner_product(
    const std::vector<std::complex<Real>>& left,
    const std::vector<std::complex<Real>>& right)
{
    std::complex<Real> result{};
    for(std::size_t i = 0; i < left.size(); ++i)
        result += std::conj(left[i])*right[i];
    return result;
}

template<class Real>
Real relative_residual(
    const analytical_eigenproblem<Real>& problem,
    const computed_eigenpair<Real>& pair)
{
    const auto applied = problem.apply(pair.right_eigenvector);
    std::vector<std::complex<Real>> residual(applied.size());
    for(std::size_t i = 0; i < applied.size(); ++i)
        residual[i] = applied[i] - pair.value*pair.right_eigenvector[i];

    const Real vector_norm_value = vector_norm(pair.right_eigenvector);
    const Real scale =
        (problem.frobenius_norm() + std::abs(pair.value))*vector_norm_value;
    const Real denominator = std::max(scale, std::numeric_limits<Real>::min());
    return vector_norm(residual)/denominator;
}

template<class Real>
std::vector<std::size_t> minimum_eigenvalue_assignment(
    const std::vector<analytical_eigenpair<Real>>& expected,
    const std::vector<computed_eigenpair<Real>>& computed)
{
    const std::size_t count = expected.size();
    std::vector<std::size_t> permutation(count);
    std::iota(permutation.begin(), permutation.end(), std::size_t(0));

    if(count <= 9)
    {
        std::vector<std::size_t> best = permutation;
        Real best_cost = std::numeric_limits<Real>::infinity();
        do
        {
            Real cost = Real{};
            for(std::size_t i = 0; i < count; ++i)
                cost += std::abs(expected[i].value - computed[permutation[i]].value);
            if(cost < best_cost)
            {
                best_cost = cost;
                best = permutation;
            }
        }
        while(std::next_permutation(permutation.begin(), permutation.end()));
        return best;
    }

    std::vector<bool> used(count, false);
    for(std::size_t i = 0; i < count; ++i)
    {
        Real best_error = std::numeric_limits<Real>::infinity();
        std::size_t best_index = count;
        for(std::size_t j = 0; j < count; ++j)
        {
            if(used[j])
                continue;
            const Real error = std::abs(expected[i].value - computed[j].value);
            if(error < best_error)
            {
                best_error = error;
                best_index = j;
            }
        }
        permutation[i] = best_index;
        used[best_index] = true;
    }
    return permutation;
}

template<class Value>
std::string value_string(const Value& value)
{
    std::ostringstream stream;
    stream.precision(17);
    stream << value;
    return stream.str();
}

} // namespace detail

template<class Real>
eigensolver_validation_report validate_eigensolver_result(
    const analytical_eigenproblem<Real>& problem,
    const eigensolver_result<Real>& result,
    const eigensolver_test_tolerances<Real>& tolerances = {})
{
    eigensolver_validation_report report;
    report.require(
        result.status == eigensolver_status::success,
        problem.name() + ": solver status is " + eigensolver_status_name(result.status));
    report.require(
        result.eigenpairs.size() == problem.eigenpairs().size(),
        problem.name() + ": computed eigenpair count does not match the reference");

    if(result.eigenpairs.size() != problem.eigenpairs().size())
        return report;

    const auto assignment =
        detail::minimum_eigenvalue_assignment(problem.eigenpairs(), result.eigenpairs);

    for(std::size_t expected_index = 0;
        expected_index < problem.eigenpairs().size();
        ++expected_index)
    {
        const auto& expected = problem.eigenpairs()[expected_index];
        const auto& computed = result.eigenpairs[assignment[expected_index]];
        const std::string prefix =
            problem.name() + ": eigenpair " + std::to_string(expected_index);

        report.require(
            detail::finite(computed.value),
            prefix + " has a non-finite eigenvalue");

        const Real eigenvalue_error = std::abs(expected.value - computed.value);
        const Real eigenvalue_tolerance =
            tolerances.eigenvalue_absolute +
            tolerances.eigenvalue_relative*std::max(Real(1), std::abs(expected.value));
        report.require(
            eigenvalue_error <= eigenvalue_tolerance,
            prefix + " eigenvalue error " + detail::value_string(eigenvalue_error) +
                " exceeds " + detail::value_string(eigenvalue_tolerance));

        if(!tolerances.require_eigenvectors)
            continue;

        report.require(
            computed.right_eigenvector.size() == problem.dimension(),
            prefix + " eigenvector size does not match the problem dimension");
        if(computed.right_eigenvector.size() != problem.dimension())
            continue;

        bool vector_is_finite = true;
        for(const auto& value : computed.right_eigenvector)
            vector_is_finite = vector_is_finite && detail::finite(value);
        report.require(vector_is_finite, prefix + " has a non-finite eigenvector");

        const Real computed_norm = detail::vector_norm(computed.right_eigenvector);
        report.require(
            computed_norm > std::numeric_limits<Real>::min(),
            prefix + " has a zero eigenvector");
        if(!(computed_norm > std::numeric_limits<Real>::min()))
            continue;

        const Real residual = detail::relative_residual(problem, computed);
        report.require(
            residual <= tolerances.residual_relative,
            prefix + " relative residual " + detail::value_string(residual) +
                " exceeds " + detail::value_string(tolerances.residual_relative));

        if(!expected.right_eigenvector.empty())
        {
            const Real expected_norm = detail::vector_norm(expected.right_eigenvector);
            const Real alignment =
                std::abs(detail::inner_product(
                    expected.right_eigenvector,
                    computed.right_eigenvector))/(expected_norm*computed_norm);
            report.require(
                Real(1) - std::min(Real(1), alignment) <=
                    tolerances.eigenvector_alignment,
                prefix + " eigenvector alignment " + detail::value_string(alignment) +
                    " is below tolerance");
        }

        if(std::isfinite(computed.reported_residual))
        {
            const Real residual_difference = std::abs(computed.reported_residual - residual);
            const Real residual_scale =
                std::max({Real(1), residual, std::abs(computed.reported_residual)});
            report.require(
                residual_difference <=
                    Real(10)*tolerances.residual_relative*residual_scale,
                prefix + " reported residual is inconsistent with the recomputed residual");
        }
    }

    report.require(
        result.operator_calls >= result.iterations,
        problem.name() + ": operator call count is smaller than iteration count");
    return report;
}

template<class Real>
eigensolver_result<Real> exact_reference_result(
    const analytical_eigenproblem<Real>& problem)
{
    eigensolver_result<Real> result;
    result.status = eigensolver_status::success;
    result.operator_calls = problem.eigenpairs().size();
    result.iterations = problem.eigenpairs().size();
    result.eigenpairs.reserve(problem.eigenpairs().size());
    for(const auto& reference : problem.eigenpairs())
    {
        computed_eigenpair<Real> pair;
        pair.value = reference.value;
        pair.right_eigenvector = reference.right_eigenvector;
        pair.reported_residual = Real{};
        result.eigenpairs.emplace_back(std::move(pair));
    }
    return result;
}

} // namespace tests
} // namespace stability

#endif
