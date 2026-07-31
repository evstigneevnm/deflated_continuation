#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <continuation/observational_knot_sample.h>

namespace
{

int checks = 0;
int failures = 0;

void require_true(const bool value, const std::string& label)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << '\n';
    }
}

void test_requested_sample_is_observational()
{
    const std::vector<double> left{1.0, 2.0};
    const std::vector<double> right{3.0, 4.0};
    const auto left_before = left;
    const auto right_before = right;
    std::vector<double> sample(2, 0.0);
    bool relocation_called = false;

    const auto result = continuation::sample_knot_observationally(
        2.0,
        2.0,
        1.0,
        left,
        3.0,
        right,
        sample,
        [](double parameter, double parameter_left,
           const std::vector<double>& value_left, double parameter_right,
           const std::vector<double>& value_right,
           std::vector<double>& output)
        {
            const double weight =
                (parameter - parameter_left)/
                (parameter_right - parameter_left);
            for(std::size_t index = 0; index < output.size(); ++index)
            {
                output[index] =
                    (1.0 - weight)*value_left[index] +
                    weight*value_right[index];
            }
            return true;
        },
        [&relocation_called](
            double, double, const std::vector<double>&, double,
            const std::vector<double>&, double&, std::vector<double>&)
        {
            relocation_called = true;
            return false;
        });

    require_true(result.sampled(), "requested sample succeeds");
    require_true(!result.relocated(), "requested sample is not relocated");
    require_true(result.sampled_parameter == 2.0, "requested sample parameter");
    require_true(sample == std::vector<double>({2.0, 3.0}), "requested sample value");
    require_true(left == left_before, "left accepted state is unchanged");
    require_true(right == right_before, "right accepted state is unchanged");
    require_true(!relocation_called, "relocation is skipped after direct success");
}

void test_relocated_sample_is_observational()
{
    const std::vector<double> left{5.0};
    const std::vector<double> right{7.0};
    const auto left_before = left;
    const auto right_before = right;
    std::vector<double> sample(1, 0.0);

    const auto result = continuation::sample_knot_observationally(
        6.0,
        6.0,
        5.0,
        left,
        7.0,
        right,
        sample,
        [](double, double, const std::vector<double>&, double,
           const std::vector<double>&, std::vector<double>&)
        {
            return false;
        },
        [](double, double, const std::vector<double>&, double,
           const std::vector<double>&, double& parameter,
           std::vector<double>& output)
        {
            parameter = 6.1;
            output = {6.1};
            return true;
        });

    require_true(result.sampled(), "relocated sample succeeds");
    require_true(result.relocated(), "relocated sample is marked");
    require_true(result.sampled_parameter == 6.1, "relocated sample parameter");
    require_true(sample == std::vector<double>({6.1}), "relocated sample value");
    require_true(left == left_before, "relocation leaves left state unchanged");
    require_true(right == right_before, "relocation leaves right state unchanged");
}

void test_failure_and_non_crossing()
{
    const std::vector<double> left{1.0};
    const std::vector<double> right{2.0};
    std::vector<double> sample(1, 0.0);
    int callbacks = 0;

    const auto outside = continuation::sample_knot_observationally(
        3.0,
        3.0,
        1.0,
        left,
        2.0,
        right,
        sample,
        [&callbacks](double, double, const std::vector<double>&, double,
                     const std::vector<double>&, std::vector<double>&)
        {
            ++callbacks;
            return true;
        },
        [&callbacks](double, double, const std::vector<double>&, double,
                     const std::vector<double>&, double&,
                     std::vector<double>&)
        {
            ++callbacks;
            return true;
        });
    require_true(!outside.crossed(), "outside knot is ignored");
    require_true(callbacks == 0, "outside knot invokes no callbacks");

    const auto failed = continuation::sample_knot_observationally(
        1.5,
        1.5,
        1.0,
        left,
        2.0,
        right,
        sample,
        [](double, double, const std::vector<double>&, double,
           const std::vector<double>&, std::vector<double>&)
        {
            return false;
        },
        [](double, double, const std::vector<double>&, double,
           const std::vector<double>&, double&, std::vector<double>&)
        {
            return false;
        });
    require_true(failed.crossed(), "failed knot was crossed");
    require_true(!failed.sampled(), "failed knot has no sample");
    require_true(
        failed.status == continuation::observational_knot_sample_status::failed,
        "failed knot status");
}

} // namespace

int main()
{
    test_requested_sample_is_observational();
    test_relocated_sample_is_observational();
    test_failure_and_non_crossing();

    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}
