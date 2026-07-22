#include <cstdlib>
#include <iostream>
#include <string>

#include <continuation/predictor_chart_validation.h>

namespace
{

int checks = 0;
int failures = 0;

void record_failure(const std::string& message)
{
    ++failures;
    std::cerr << "FAIL " << message << std::endl;
}

void require_true(const bool value, const std::string& label)
{
    ++checks;
    if(!value)
    {
        record_failure(label);
    }
}

void require_false(const bool value, const std::string& label)
{
    require_true(!value, label);
}

void require_close(const double value, const double expected, const double tolerance, const std::string& label)
{
    ++checks;
    const double error = value > expected ? value - expected : expected - value;
    if(!(error <= tolerance))
    {
        record_failure(
            label +
            " value=" + std::to_string(value) +
            " expected=" + std::to_string(expected) +
            " error=" + std::to_string(error) +
            " tolerance=" + std::to_string(tolerance));
    }
}

void test_good_predictor_passes()
{
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, 0.019, 1.0e-4);
    require_true(
        result.decision == continuation::predictor_chart_decision::accept,
        "good predictor is accepted");
    require_false(result.needs_repair(), "good predictor does not need repair");
    require_false(result.charted_non_positive, "good predictor has positive charted progress");
    require_false(result.progress_jump_warning, "good predictor has no progress jump warning");
    require_false(result.displacement_warning, "good predictor has no displacement warning");
}

void test_backward_charted_progress_rejected()
{
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, -0.01, 0.03);
    require_true(
        result.decision == continuation::predictor_chart_decision::reject_chart,
        "backward charted progress rejects chart");
    require_true(result.charted_non_positive, "backward charted progress is detected");
    require_true(result.needs_repair(), "backward charted progress needs repair");
}

void test_huge_progress_jump_rejected()
{
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, 2.0, 0.01);
    require_true(
        result.decision == continuation::predictor_chart_decision::reject_chart,
        "huge progress jump rejects chart");
    require_true(result.progress_jump_warning, "large progress jump emits warning");
    require_true(result.progress_jump_reject, "huge progress jump is marked for repair");
    require_true(result.needs_repair(), "huge progress jump needs repair");
    require_close(result.progress_ratio, 100.0, 1.0e-12, "progress ratio");
}

void test_large_displacement_rejected()
{
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, 0.02, 3.0);
    require_true(
        result.decision == continuation::predictor_chart_decision::reject_chart,
        "large displacement rejects chart");
    require_true(result.displacement_warning, "large displacement emits warning");
    require_true(result.displacement_reject, "huge displacement is marked for repair");
    require_true(result.needs_repair(), "huge displacement needs repair");
    require_close(result.displacement_ratio, 150.0, 1.0e-12, "displacement ratio");
}

void test_warning_only_weak_progress()
{
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, 0.003, 0.1);
    require_true(
        result.decision == continuation::predictor_chart_decision::accept_with_warning,
        "weak progress is accepted with warning");
    require_true(result.charted_weak_warning, "weak progress emits warning");
    require_false(result.charted_weak_reject, "moderately weak progress is warning-only");
    require_false(result.needs_repair(), "moderately weak progress does not need repair");
}

void test_bad_raw_tangent_is_distinct_from_bad_chart()
{
    const auto result = continuation::validate_predictor_chart(0.02, -0.02, -0.01, 0.0);
    require_true(
        result.decision == continuation::predictor_chart_decision::reject_tangent,
        "backward raw progress rejects tangent");
    require_true(result.rejected(), "bad raw tangent is rejected");
}

void test_disabled_policy_is_diagnostic_only()
{
    continuation::predictor_chart_policy<double> policy;
    policy.enabled = false;
    const auto result = continuation::validate_predictor_chart(0.02, 0.02, -0.01, 3.0, policy);
    require_true(
        result.decision == continuation::predictor_chart_decision::accept,
        "disabled policy accepts bad chart");
    require_true(result.needs_repair(), "disabled policy still computes diagnostics");
}

void test_invalid_policy_is_rejected()
{
    continuation::predictor_chart_policy<double> policy;
    policy.minimum_progress_ratio = 0.3;
    policy.weak_progress_warning_ratio = 0.2;
    bool threw = false;
    try
    {
        continuation::validate_predictor_chart_policy(policy);
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }
    require_true(threw, "invalid predictor chart policy is rejected");
}

} // namespace

int main()
{
    test_good_predictor_passes();
    test_backward_charted_progress_rejected();
    test_huge_progress_jump_rejected();
    test_large_displacement_rejected();
    test_warning_only_weak_progress();
    test_bad_raw_tangent_is_distinct_from_bad_chart();
    test_disabled_policy_is_diagnostic_only();
    test_invalid_policy_is_rejected();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
