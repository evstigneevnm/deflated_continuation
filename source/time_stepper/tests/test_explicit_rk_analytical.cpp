#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <scfd/utils/log_std.h>

#include <time_stepper/tests/common/legacy_fixed_step_runner.h>
#include <time_stepper/tests/common/test_context.h>

namespace
{

template<class VectorOperations>
struct exponential_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type,
        const vector_type& state,
        const scalar_type rate,
        vector_type& output) const
    {
        output(0) = rate*state(0);
    }
};

template<class VectorOperations>
struct nonautonomous_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type time,
        const vector_type& state,
        const scalar_type,
        vector_type& output) const
    {
        output(0) = time*state(0);
    }
};

long double observed_order(
    const long double coarse_error,
    const long double fine_error)
{
    return std::log(coarse_error/fine_error)/std::log(2.0L);
}

} // namespace

int main()
{
    using scalar_type = double;
    using vector_operations_type = scfd_serial_cpu_vector_operations<scalar_type>;
    using log_type = scfd::utils::log_std;
    using time_steppers::tests::integrate_legacy_explicit_fixed;

    time_steppers::tests::test_context test;
    vector_operations_type vector_operations(1);
    log_type log;
    log.set_verbosity(0);
    exponential_problem<vector_operations_type> exponential;

    struct method_expectation
    {
        const char* name;
        unsigned int order;
    };
    const std::vector<method_expectation> methods = {
        {"EE", 1},
        {"HE", 2},
        {"RK33SSP", 3},
        {"RK43SSP", 3},
        {"RKDP45", 5},
        {"RK64SSP", 4}
    };

    constexpr scalar_type initial_time = 0;
    constexpr scalar_type final_time = 1;
    constexpr scalar_type rate = -0.7;
    const scalar_type exact = std::exp(rate*final_time);
    for(const auto& method: methods)
    {
        const auto coarse = integrate_legacy_explicit_fixed<1>(
            vector_operations,
            exponential,
            log,
            method.name,
            std::array<scalar_type, 1>{{1}},
            rate,
            initial_time,
            final_time,
            10);
        const auto fine = integrate_legacy_explicit_fixed<1>(
            vector_operations,
            exponential,
            log,
            method.name,
            std::array<scalar_type, 1>{{1}},
            rate,
            initial_time,
            final_time,
            20);
        const long double coarse_error = std::abs(static_cast<long double>(coarse[0])-exact);
        const long double fine_error = std::abs(static_cast<long double>(fine[0])-exact);
        const long double order = observed_order(coarse_error, fine_error);
        test.check(
            order >= static_cast<long double>(method.order)-0.2L,
            std::string(method.name) + " observed order=" + std::to_string(static_cast<double>(order)));
        test.check(
            fine_error < coarse_error,
            std::string(method.name) + " error decreases under refinement");
    }

    nonautonomous_problem<vector_operations_type> nonautonomous;
    const auto nonautonomous_coarse = integrate_legacy_explicit_fixed<1>(
        vector_operations,
        nonautonomous,
        log,
        "RKDP45",
        std::array<scalar_type, 1>{{1}},
        0,
        initial_time,
        final_time,
        8);
    const auto nonautonomous_fine = integrate_legacy_explicit_fixed<1>(
        vector_operations,
        nonautonomous,
        log,
        "RKDP45",
        std::array<scalar_type, 1>{{1}},
        0,
        initial_time,
        final_time,
        16);
    const scalar_type nonautonomous_exact = std::exp(0.5*final_time*final_time);
    const long double nonautonomous_order = observed_order(
        std::abs(static_cast<long double>(nonautonomous_coarse[0])-nonautonomous_exact),
        std::abs(static_cast<long double>(nonautonomous_fine[0])-nonautonomous_exact));
    test.check(
        nonautonomous_order >= 4.5L,
        "RKDP45 nonautonomous observed order=" +
            std::to_string(static_cast<double>(nonautonomous_order)));

    std::cout << "Explicit RK analytical checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
