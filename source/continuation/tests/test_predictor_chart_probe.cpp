#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <continuation/predictor_chart_probe.h>

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

    void add_mul(const double value, const vector_type& source, vector_type& destination) const
    {
        for(std::size_t i = 0; i < source.size(); ++i)
        {
            destination[i] += value*source[i];
        }
    }

    void assign_mul(
        const double left_value,
        const vector_type& left,
        const double right_value,
        const vector_type& right,
        vector_type& destination) const
    {
        destination.resize(left.size());
        for(std::size_t i = 0; i < left.size(); ++i)
        {
            destination[i] = left_value*left[i] + right_value*right[i];
        }
    }

    double scalar_prod(const vector_type& left, const vector_type& right) const
    {
        double result = 0.0;
        for(std::size_t i = 0; i < left.size(); ++i)
        {
            result += left[i]*right[i];
        }
        return result;
    }

    double norm_l2(const vector_type& value) const
    {
        return std::sqrt(scalar_prod(value, value));
    }
};

struct test_log
{
};

struct plain_operator
{
};

struct jumping_chart_operator
{
    int begin_calls = 0;
    int stabilize_calls = 0;

    void begin_continuation_chart(
        const std::vector<double>&,
        const double&,
        const std::vector<double>&,
        const double&)
    {
        ++begin_calls;
    }

    void stabilize_predictor_for_continuation(
        const std::vector<double>&,
        const double&,
        const std::vector<double>&,
        const double&,
        const std::vector<double>& predictor,
        const double& predictor_lambda,
        std::vector<double>& trial,
        double& trial_lambda)
    {
        ++stabilize_calls;
        trial = predictor;
        trial[1] += 11.0;
        trial_lambda = predictor_lambda;
    }
};

int checks = 0;
int failures = 0;

void require_true(const bool value, const std::string& label)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

void require_close(const double value, const double expected, const double tolerance, const std::string& label)
{
    ++checks;
    if(std::abs(value - expected) > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label << " value=" << value << " expected=" << expected << std::endl;
    }
}

template<class Operator>
continuation::predictor_chart_probe_result<double> run_probe(Operator& op)
{
    vector_ops ops;
    test_log log;
    std::vector<double> x0{1.0, 2.0};
    std::vector<double> tangent{0.6, 0.0};
    std::vector<double> raw;
    std::vector<double> charted;
    std::vector<double> work;
    continuation::predictor_chart_policy<double> policy;
    return continuation::probe_predictor_chart(
        &ops,
        &log,
        &op,
        x0,
        3.0,
        tangent,
        0.8,
        0.1,
        raw,
        charted,
        work,
        policy);
}

void test_plain_operator_is_copy_only()
{
    plain_operator op;
    static_assert(!continuation::chart::has_continuation_chart<
        plain_operator,
        std::vector<double>,
        double>::value,
        "plain operator must not advertise a continuation chart");
    const auto result = run_probe(op);
    require_close(result.raw_tangent_progress, 0.1, 1.0e-14, "plain raw progress");
    require_close(result.charted_tangent_progress, 0.1, 1.0e-14, "plain charted progress");
    require_close(result.chart_displacement, 0.0, 1.0e-14, "plain chart displacement");
    require_true(
        result.validation.decision == continuation::predictor_chart_decision::accept,
        "plain operator predictor accepted");
}

void test_chart_jump_is_rejected()
{
    jumping_chart_operator op;
    static_assert(continuation::chart::has_continuation_chart<
        jumping_chart_operator,
        std::vector<double>,
        double>::value,
        "hook operator must advertise a continuation chart");
    const auto result = run_probe(op);
    require_true(op.begin_calls == 1, "chart begin called once");
    require_true(op.stabilize_calls == 1, "chart stabilizer called once");
    require_true(result.validation.displacement_reject, "large chart displacement detected");
    require_true(
        result.validation.decision == continuation::predictor_chart_decision::reject_chart,
        "large chart displacement rejects chart");
}

} // namespace

int main()
{
    test_plain_operator_is_copy_only();
    test_chart_jump_is_rejected();
    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
