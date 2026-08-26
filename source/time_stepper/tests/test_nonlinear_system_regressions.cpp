#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <scfd/utils/log_std.h>

#include <time_stepper/tests/common/legacy_fixed_step_runner.h>
#include <time_stepper/tests/common/legacy_implicit_fixed_step_runner.h>
#include <time_stepper/tests/common/nonlinear_benchmark_problems.h>
#include <time_stepper/tests/common/test_context.h>

namespace
{

template<std::size_t Dimension>
long double distance(
    const std::array<double, Dimension>& left,
    const std::array<double, Dimension>& right)
{
    long double result = 0;
    for(std::size_t i = 0; i < Dimension; ++i)
    {
        const long double difference =
            static_cast<long double>(left[i])-static_cast<long double>(right[i]);
        result += difference*difference;
    }
    return std::sqrt(result);
}

template<std::size_t Dimension>
long double observed_self_convergence_order(
    const std::array<double, Dimension>& coarse,
    const std::array<double, Dimension>& fine,
    const std::array<double, Dimension>& finer)
{
    return std::log(distance(coarse, fine)/distance(fine, finer))/std::log(2.0L);
}

template<std::size_t Dimension>
bool all_finite(const std::array<double, Dimension>& value)
{
    for(const auto entry: value)
    {
        if(!std::isfinite(entry))
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    using scalar_type = double;
    using log_type = scfd::utils::log_std;
    using time_steppers::tests::integrate_legacy_explicit_fixed;
    using time_steppers::tests::integrate_legacy_implicit_fixed;

    time_steppers::tests::test_context test;
    log_type log;
    log.set_verbosity(0);

    using vector_operations_3d = scfd_serial_cpu_vector_operations<scalar_type>;
    vector_operations_3d operations_3d(3);
    time_steppers::tests::lorenz_problem<vector_operations_3d> lorenz;
    typename vector_operations_3d::vector_type state_3d;
    typename vector_operations_3d::vector_type rate_3d;
    operations_3d.init_vectors(state_3d, rate_3d);
    operations_3d.start_use_vectors(state_3d, rate_3d);
    state_3d(0) = 1;
    state_3d(1) = 1;
    state_3d(2) = 1;
    lorenz.F(0, state_3d, 0, rate_3d);
    test.check(
        std::abs(rate_3d(0)) <= 32*std::numeric_limits<scalar_type>::epsilon() &&
        std::abs(rate_3d(1)-26) <= 32*std::numeric_limits<scalar_type>::epsilon() &&
        std::abs(rate_3d(2)+scalar_type(5)/scalar_type(3)) <=
            32*std::numeric_limits<scalar_type>::epsilon(),
        "Lorenz benchmark RHS matches its analytical value");
    const auto lorenz_coarse = integrate_legacy_explicit_fixed<3>(
        operations_3d, lorenz, log, "RKDP45", {{1, 1, 1}}, 0, 0, 0.25, 10);
    const auto lorenz_fine = integrate_legacy_explicit_fixed<3>(
        operations_3d, lorenz, log, "RKDP45", {{1, 1, 1}}, 0, 0, 0.25, 20);
    const auto lorenz_finer = integrate_legacy_explicit_fixed<3>(
        operations_3d, lorenz, log, "RKDP45", {{1, 1, 1}}, 0, 0, 0.25, 40);
    const long double lorenz_order = observed_self_convergence_order(
        lorenz_coarse, lorenz_fine, lorenz_finer);
    test.check(all_finite(lorenz_finer), "Lorenz trajectory remains finite");
    test.check(
        lorenz_order >= 4.2L,
        "Lorenz RKDP45 self-convergence order=" +
            std::to_string(static_cast<double>(lorenz_order)));

    time_steppers::tests::rossler_problem<vector_operations_3d> rossler;
    state_3d(0) = 1;
    state_3d(1) = 0;
    state_3d(2) = 0;
    rossler.F(0, state_3d, 0, rate_3d);
    test.check(
        std::abs(rate_3d(0)) <= 32*std::numeric_limits<scalar_type>::epsilon() &&
        std::abs(rate_3d(1)-1) <= 32*std::numeric_limits<scalar_type>::epsilon() &&
        std::abs(rate_3d(2)-scalar_type(0.2)) <=
            32*std::numeric_limits<scalar_type>::epsilon(),
        "Rossler benchmark RHS matches its analytical value");
    const auto rossler_coarse = integrate_legacy_explicit_fixed<3>(
        operations_3d, rossler, log, "RKDP45", {{1, 0, 0}}, 0, 0, 1, 8);
    const auto rossler_fine = integrate_legacy_explicit_fixed<3>(
        operations_3d, rossler, log, "RKDP45", {{1, 0, 0}}, 0, 0, 1, 16);
    const auto rossler_finer = integrate_legacy_explicit_fixed<3>(
        operations_3d, rossler, log, "RKDP45", {{1, 0, 0}}, 0, 0, 1, 32);
    const long double rossler_order = observed_self_convergence_order(
        rossler_coarse, rossler_fine, rossler_finer);
    test.check(all_finite(rossler_finer), "Rossler trajectory remains finite");
    test.check(
        rossler_order >= 4.2L,
        "Rossler RKDP45 self-convergence order=" +
            std::to_string(static_cast<double>(rossler_order)));

    using vector_operations_2d = scfd_serial_cpu_vector_operations<scalar_type>;
    vector_operations_2d operations_2d(2);
    time_steppers::tests::van_der_pol_problem<vector_operations_2d> van_der_pol;
    constexpr scalar_type mu = 5;
    typename vector_operations_2d::vector_type state_2d;
    typename vector_operations_2d::vector_type rate_2d;
    operations_2d.init_vectors(state_2d, rate_2d);
    operations_2d.start_use_vectors(state_2d, rate_2d);
    state_2d(0) = 2;
    state_2d(1) = 0;
    van_der_pol.F(0, state_2d, mu, rate_2d);
    test.check(
        std::abs(rate_2d(0)) <= 32*std::numeric_limits<scalar_type>::epsilon() &&
        std::abs(rate_2d(1)+2) <= 32*std::numeric_limits<scalar_type>::epsilon(),
        "Van der Pol benchmark RHS matches its analytical value");
    const auto vdp_explicit_coarse = integrate_legacy_explicit_fixed<2>(
        operations_2d, van_der_pol, log, "RKDP45", {{2, 0}}, mu, 0, 0.5, 40);
    const auto vdp_explicit_fine = integrate_legacy_explicit_fixed<2>(
        operations_2d, van_der_pol, log, "RKDP45", {{2, 0}}, mu, 0, 0.5, 80);
    const auto vdp_explicit_finer = integrate_legacy_explicit_fixed<2>(
        operations_2d, van_der_pol, log, "RKDP45", {{2, 0}}, mu, 0, 0.5, 160);
    const long double vdp_explicit_order = observed_self_convergence_order(
        vdp_explicit_coarse, vdp_explicit_fine, vdp_explicit_finer);
    test.check(all_finite(vdp_explicit_finer), "Van der Pol explicit trajectory remains finite");
    test.check(
        vdp_explicit_order >= 4.2L,
        "Van der Pol RKDP45 self-convergence order=" +
            std::to_string(static_cast<double>(vdp_explicit_order)));

    typename time_steppers::tests::van_der_pol_problem<vector_operations_2d>::linear_operator
        vdp_linear_operator(&van_der_pol);
    const auto vdp_implicit_coarse = integrate_legacy_implicit_fixed<2>(
        operations_2d,
        van_der_pol,
        vdp_linear_operator,
        log,
        "SDIRK3A3",
        {{2, 0}},
        mu,
        0,
        0.5,
        20);
    const auto vdp_implicit_fine = integrate_legacy_implicit_fixed<2>(
        operations_2d,
        van_der_pol,
        vdp_linear_operator,
        log,
        "SDIRK3A3",
        {{2, 0}},
        mu,
        0,
        0.5,
        40);
    const auto vdp_implicit_finer = integrate_legacy_implicit_fixed<2>(
        operations_2d,
        van_der_pol,
        vdp_linear_operator,
        log,
        "SDIRK3A3",
        {{2, 0}},
        mu,
        0,
        0.5,
        80);
    const long double vdp_implicit_order = observed_self_convergence_order(
        vdp_implicit_coarse, vdp_implicit_fine, vdp_implicit_finer);
    test.check(all_finite(vdp_implicit_finer), "Van der Pol implicit trajectory remains finite");
    test.check(
        vdp_implicit_order >= 2.5L,
        "Van der Pol SDIRK3A3 self-convergence order=" +
            std::to_string(static_cast<double>(vdp_implicit_order)));

    operations_2d.stop_use_vectors(state_2d, rate_2d);
    operations_2d.free_vectors(state_2d, rate_2d);
    operations_3d.stop_use_vectors(state_3d, rate_3d);
    operations_3d.free_vectors(state_3d, rate_3d);

    std::cout << "Nonlinear timestepper checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
