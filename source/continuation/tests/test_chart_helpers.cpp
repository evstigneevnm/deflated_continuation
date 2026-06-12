#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <continuation/chart_helpers.h>

namespace
{

struct vector_ops
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void assign(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }
};

struct test_log
{
    std::vector<std::string> contexts;

    void info(const std::string&)
    {
    }

    template<class... Args>
    void info_f(const char*, Args...)
    {
    }
};

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

void require_vector(
    const std::vector<double>& value,
    const std::vector<double>& expected,
    const std::string& label)
{
    ++checks;
    if(value.size() != expected.size())
    {
        record_failure(label + " size mismatch");
        return;
    }
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        if(value[i] != expected[i])
        {
            record_failure(
                label +
                " mismatch at " + std::to_string(i) +
                " value=" + std::to_string(value[i]) +
                " expected=" + std::to_string(expected[i]));
            return;
        }
    }
}

struct plain_operator
{
};

struct hook_operator
{
    int begin_calls = 0;
    int predictor_calls = 0;
    int corrector_calls = 0;
    int arclength_calls = 0;
    int arclength_tangent_calls = 0;
    int log_calls = 0;

    void begin_continuation_chart(
        const std::vector<double>& x0,
        const double& lambda0,
        const std::vector<double>& x0_s,
        const double& lambda0_s)
    {
        ++begin_calls;
        last_scalar = x0[0] + lambda0 + x0_s[0] + lambda0_s;
    }

    void stabilize_predictor_for_continuation(
        const std::vector<double>& x0,
        const double& lambda0,
        const std::vector<double>& x0_s,
        const double& lambda0_s,
        const std::vector<double>& x_predictor,
        const double& lambda_predictor,
        std::vector<double>& x_trial,
        double& lambda_trial)
    {
        ++predictor_calls;
        x_trial = x_predictor;
        x_trial[0] += x0[0];
        x_trial[1] += x0_s[1];
        lambda_trial = lambda_predictor + lambda0 + lambda0_s;
    }

    void stabilize_corrector_trial(
        const std::vector<double>& reference,
        const double& reference_lambda,
        std::vector<double>& trial,
        double& trial_lambda)
    {
        ++corrector_calls;
        trial[0] += reference[0];
        trial[1] += reference[1];
        trial_lambda += reference_lambda;
    }

    void stabilize_for_arclength(
        const std::vector<double>& reference,
        const std::vector<double>& source,
        std::vector<double>& destination)
    {
        ++arclength_calls;
        destination = source;
        destination[0] += reference[0];
        destination[1] += reference[1];
    }

    void stabilize_tangent_for_arclength(
        const std::vector<double>& reference,
        const std::vector<double>& tangent,
        std::vector<double>& destination)
    {
        ++arclength_tangent_calls;
        destination = tangent;
        destination[0] += reference[0];
        destination[1] += reference[1];
    }

    void log_continuation_chart(test_log* log, const char* context)
    {
        ++log_calls;
        log->contexts.push_back(context);
    }

    double last_scalar = 0.0;
};

struct legacy_operator
{
    int project_calls = 0;
    int arclength_calls = 0;
    int log_calls = 0;

    void project_relative_to(const std::vector<double>& reference, std::vector<double>& trial)
    {
        ++project_calls;
        trial[0] -= reference[0];
        trial[1] -= reference[1];
    }

    void stabilize_for_arclength(
        const std::vector<double>& reference,
        const std::vector<double>& source,
        std::vector<double>& destination)
    {
        ++arclength_calls;
        destination = source;
        destination[0] = reference[0] - source[0];
        destination[1] = reference[1] - source[1];
    }

    void log_projection_diagnostics(test_log* log, const char* context)
    {
        ++log_calls;
        log->contexts.push_back(context);
    }
};

void test_plain_fallback()
{
    vector_ops ops;
    test_log log;
    plain_operator op;

    std::vector<double> x0{1.0, 2.0};
    std::vector<double> x0_s{0.5, -0.25};
    std::vector<double> predictor{3.0, 4.0};
    std::vector<double> trial;
    double lambda0 = 5.0;
    double lambda0_s = 0.25;
    double lambda_predictor = 6.0;
    double lambda_trial = 0.0;

    continuation::chart::begin_continuation_chart(&ops, &log, &op, x0, lambda0, x0_s, lambda0_s);
    continuation::chart::stabilize_predictor_for_continuation(
        &ops,
        &log,
        &op,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        predictor,
        lambda_predictor,
        trial,
        lambda_trial);
    require_vector(trial, predictor, "plain predictor fallback copies vector");
    require_close(lambda_trial, lambda_predictor, 0.0, "plain predictor fallback copies lambda");

    continuation::chart::stabilize_corrector_trial(&ops, &log, &op, x0, lambda0, trial, lambda_trial);
    require_vector(trial, predictor, "plain corrector fallback leaves vector unchanged");
    require_close(lambda_trial, lambda_predictor, 0.0, "plain corrector fallback leaves lambda unchanged");

    std::vector<double> charted;
    continuation::chart::stabilize_for_arclength(&ops, &op, x0, predictor, charted);
    require_vector(charted, predictor, "plain arclength fallback copies vector");

    std::vector<double> charted_tangent;
    continuation::chart::stabilize_tangent_for_arclength(&ops, &op, x0, x0_s, charted_tangent);
    require_vector(charted_tangent, x0_s, "plain arclength tangent fallback copies vector");

    continuation::chart::log_continuation_chart(&log, &op, "plain");
    require_true(log.contexts.empty(), "plain log fallback is no-op");
}

void test_new_hooks_take_priority()
{
    vector_ops ops;
    test_log log;
    hook_operator op;

    std::vector<double> x0{1.0, 2.0};
    std::vector<double> x0_s{0.5, -0.25};
    std::vector<double> predictor{3.0, 4.0};
    std::vector<double> trial;
    double lambda0 = 5.0;
    double lambda0_s = 0.25;
    double lambda_predictor = 6.0;
    double lambda_trial = 0.0;

    continuation::chart::begin_continuation_chart(&ops, &log, &op, x0, lambda0, x0_s, lambda0_s);
    require_true(op.begin_calls == 1, "begin hook called");
    require_close(op.last_scalar, 6.75, 0.0, "begin hook receives chart data");

    continuation::chart::stabilize_predictor_for_continuation(
        &ops,
        &log,
        &op,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        predictor,
        lambda_predictor,
        trial,
        lambda_trial);
    require_true(op.predictor_calls == 1, "predictor hook called");
    require_vector(trial, {4.0, 3.75}, "predictor hook owns vector output");
    require_close(lambda_trial, 11.25, 0.0, "predictor hook owns lambda output");

    continuation::chart::stabilize_corrector_trial(&ops, &log, &op, x0, lambda0, trial, lambda_trial);
    require_true(op.corrector_calls == 1, "corrector hook called");
    require_vector(trial, {5.0, 5.75}, "corrector hook owns vector update");
    require_close(lambda_trial, 16.25, 0.0, "corrector hook owns lambda update");

    std::vector<double> charted;
    continuation::chart::stabilize_for_arclength(&ops, &op, x0, predictor, charted);
    require_true(op.arclength_calls == 1, "arclength hook called");
    require_vector(charted, {4.0, 6.0}, "arclength hook owns charted vector");

    std::vector<double> charted_tangent;
    continuation::chart::stabilize_tangent_for_arclength(&ops, &op, x0, x0_s, charted_tangent);
    require_true(op.arclength_tangent_calls == 1, "arclength tangent hook called");
    require_vector(charted_tangent, {1.5, 1.75}, "arclength tangent hook owns charted vector");

    continuation::chart::log_continuation_chart(&log, &op, "hook");
    require_true(op.log_calls == 1, "chart log hook called");
    require_true(log.contexts.size() == 1 && log.contexts[0] == "hook", "chart log hook receives context");
}

void test_legacy_projection_compatibility()
{
    vector_ops ops;
    test_log log;
    legacy_operator op;

    std::vector<double> x0{1.0, 2.0};
    std::vector<double> x0_s{0.5, -0.25};
    std::vector<double> predictor{3.0, 4.0};
    std::vector<double> trial;
    double lambda0 = 5.0;
    double lambda0_s = 0.25;
    double lambda_predictor = 6.0;
    double lambda_trial = 0.0;

    continuation::chart::stabilize_predictor_for_continuation(
        &ops,
        &log,
        &op,
        x0,
        lambda0,
        x0_s,
        lambda0_s,
        predictor,
        lambda_predictor,
        trial,
        lambda_trial);
    require_vector(trial, predictor, "legacy predictor fallback stays generic");
    require_true(op.project_calls == 0, "legacy project_relative_to is not used for predictor fallback");

    continuation::chart::stabilize_corrector_trial(&ops, &log, &op, x0, lambda0, trial, lambda_trial);
    require_true(op.project_calls == 1, "legacy project_relative_to used for corrector fallback");
    require_vector(trial, {2.0, 2.0}, "legacy corrector projection result");

    std::vector<double> charted;
    continuation::chart::stabilize_for_arclength(&ops, &op, x0, predictor, charted);
    require_true(op.arclength_calls == 1, "legacy arclength hook called");
    require_vector(charted, {-2.0, -2.0}, "legacy arclength hook result");

    std::vector<double> charted_tangent;
    continuation::chart::stabilize_tangent_for_arclength(&ops, &op, x0, x0_s, charted_tangent);
    require_vector(charted_tangent, x0_s, "legacy arclength tangent fallback copies vector");

    continuation::chart::log_continuation_chart(&log, &op, "legacy");
    require_true(op.log_calls == 1, "legacy projection diagnostics used for chart log fallback");
    require_true(log.contexts.size() == 1 && log.contexts[0] == "legacy", "legacy log receives context");
}

} // namespace

int main()
{
    test_plain_fallback();
    test_new_hooks_take_priority();
    test_legacy_projection_compatibility();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
