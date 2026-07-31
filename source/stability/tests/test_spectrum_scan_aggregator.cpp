#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <stability/analysis/spectrum_classifier.h>
#include <stability/analysis/spectrum_scan_aggregator.h>

namespace
{

using result_type =
    stability::eigensolvers::eigensolver_result<double>;
using estimate_type =
    stability::eigensolvers::eigenpair_estimate<double>;

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

estimate_type estimate(
    std::complex<double> value,
    double relative_residual)
{
    estimate_type result;
    result.value = value;
    result.residual = relative_residual;
    result.relative_residual = relative_residual;
    result.converged = true;
    return result;
}

result_type success(
    std::initializer_list<estimate_type> eigenpairs,
    std::size_t iterations,
    std::size_t calls)
{
    result_type result;
    result.status =
        stability::eigensolvers::eigensolver_status::success;
    result.eigenpairs.assign(
        eigenpairs.begin(),
        eigenpairs.end());
    result.iterations = iterations;
    result.operator_calls = calls;
    result.inner_solver_calls = 2*calls;
    result.effective_subspace_dimension = eigenpairs.size();
    return result;
}

result_type failure()
{
    result_type result;
    result.status =
        stability::eigensolvers::eigensolver_status::
            inner_solver_failure;
    result.diagnostic = "expected scan failure";
    return result;
}

void test_complete_aggregation()
{
    using aggregator_type =
        stability::analysis::spectrum_scan_aggregator<double>;
    aggregator_type aggregator(2);
    aggregator.add(
        success(
            {
                estimate({2.0, 0.0}, 1.0e-6),
                estimate({-1.0, 3.0}, 2.0e-8),
                estimate({-1.0, -3.0}, 2.0e-8)
            },
            4,
            7),
        "first");
    aggregator.add(
        success(
            {
                estimate({2.0 + 2.0e-10, 0.0}, 1.0e-12),
                estimate({-4.0, 0.0}, 3.0e-10)
            },
            5,
            11),
        "second");

    const auto result = aggregator.finish();
    require(result.succeeded(), "complete scan succeeds");
    require(result.coverage_complete, "complete scan coverage");
    require(
        result.scans_requested == 2 &&
            result.scans_succeeded == 2,
        "complete scan accounting");
    require(
        result.eigenpairs.size() == 4,
        "near-duplicate eigenvalue is merged");
    require(
        result.iterations == 9 &&
            result.operator_calls == 18 &&
            result.inner_solver_calls == 36,
        "scan work counters are accumulated");

    const auto duplicate = std::find_if(
        result.eigenpairs.begin(),
        result.eigenpairs.end(),
        [](const estimate_type& value)
        {
            return std::abs(value.value.real() - 2.0) < 1.0e-6;
        });
    require(
        duplicate != result.eigenpairs.end() &&
            duplicate->relative_residual == 1.0e-12,
        "lower-residual duplicate is retained");
}

void test_partial_coverage_policy()
{
    using options_type =
        stability::analysis::
            spectrum_scan_aggregation_options<double>;
    using aggregator_type =
        stability::analysis::spectrum_scan_aggregator<double>;

    options_type strict_options;
    strict_options.require_all_scans = true;
    aggregator_type strict(2, strict_options);
    strict.add(
        success(
            {
                estimate({1.0, 0.0}, 1.0e-12),
                estimate({-2.0, 0.0}, 1.0e-12)
            },
            1,
            1),
        "success");
    strict.add(failure(), "failure");
    const auto strict_result = strict.finish();
    require(
        !strict_result.succeeded(),
        "strict aggregation rejects a failed scan");

    options_type permissive_options;
    permissive_options.require_all_scans = false;
    aggregator_type permissive(2, permissive_options);
    permissive.add(
        success(
            {
                estimate({1.0, 0.0}, 1.0e-12),
                estimate({-2.0, 0.0}, 1.0e-12)
            },
            1,
            1),
        "success");
    permissive.add(failure(), "failure");
    auto partial_result = permissive.finish();
    require(
        partial_result.succeeded() &&
            !partial_result.coverage_complete,
        "permissive aggregation reports partial coverage");

    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    classifier_type classifier;
    const auto rejected = classifier.classify(partial_result);
    require(
        !rejected.classification_complete() &&
            rejected.diagnostic.find("1/2") != std::string::npos,
        "classifier rejects partial coverage by default");

    auto classifier_options = classifier.options();
    classifier_options.require_complete_scan_coverage = false;
    classifier.set_options(classifier_options);
    const auto accepted =
        classifier.classify(std::move(partial_result));
    require(
        accepted.classification_complete() &&
            accepted.unstable.real == 1,
        "classifier can explicitly accept partial coverage");
}

void test_minimum_aggregated_spectrum()
{
    using options_type =
        stability::analysis::
            spectrum_scan_aggregation_options<double>;
    using aggregator_type =
        stability::analysis::spectrum_scan_aggregator<double>;

    options_type options;
    options.minimum_eigenpairs = 2;
    aggregator_type aggregator(2, options);
    aggregator.add(
        success(
            {estimate({1.0, 0.0}, 1.0e-12)},
            1,
            1),
        "first");
    aggregator.add(
        success(
            {estimate({1.0 + 1.0e-10, 0.0}, 1.0e-13)},
            1,
            1),
        "second");
    const auto result = aggregator.finish();
    require(
        !result.succeeded() &&
            result.coverage_complete &&
            result.eigenpairs.size() == 1,
        "aggregate rejects too few physical eigenpairs");
}

void test_geometric_multiplicity_is_preserved()
{
    using options_type =
        stability::analysis::
            spectrum_scan_aggregation_options<double>;
    using aggregator_type =
        stability::analysis::spectrum_scan_aggregator<double>;

    options_type options;
    options.minimum_eigenpairs = 3;
    aggregator_type aggregator(2, options);
    aggregator.add(
        success(
            {
                estimate({2.0, 0.0}, 1.0e-9),
                estimate({2.0, 0.0}, 2.0e-9),
                estimate({-1.0, 0.0}, 1.0e-9)
            },
            1,
            1),
        "first");
    aggregator.add(
        success(
            {
                estimate({2.0 + 1.0e-10, 0.0}, 1.0e-12),
                estimate({2.0 - 1.0e-10, 0.0}, 2.0e-12),
                estimate({-1.0, 0.0}, 1.0e-12)
            },
            1,
            1),
        "second");
    const auto result = aggregator.finish();
    const auto repeated = std::count_if(
        result.eigenpairs.begin(),
        result.eigenpairs.end(),
        [](const estimate_type& value)
        {
            return std::abs(value.value.real() - 2.0) < 1.0e-6;
        });
    require(
        result.succeeded() &&
            result.eigenpairs.size() == 3 &&
            repeated == 2,
        "within-scan eigenvalue multiplicity is preserved");
}

} // namespace

int main()
{
    test_complete_aggregation();
    test_partial_coverage_policy();
    test_minimum_aggregated_spectrum();
    test_geometric_multiplicity_is_preserved();
    std::cout
        << "Spectrum scan aggregation checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
