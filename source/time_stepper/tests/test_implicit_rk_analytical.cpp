#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <common/scfd_serial_cpu_vector_operations.h>
#include <scfd/utils/log_std.h>

#include <time_stepper/tests/common/legacy_implicit_fixed_step_runner.h>
#include <time_stepper/tests/common/test_context.h>

namespace
{

template<class VectorOperations>
class scalar_linear_problem
{
public:
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

    void set_linearization_point(const vector_type&, const scalar_type rate)
    {
        jacobian_ = rate;
    }

    class linear_operator
    {
    public:
        explicit linear_operator(const scalar_linear_problem* problem): problem_(problem)
        {}

        void set_aE_plus_bA(const std::pair<scalar_type, scalar_type>& coefficients)
        {
            coefficients_ = coefficients;
        }

        bool solve(const vector_type& right_hand_side, vector_type& solution) const
        {
            const scalar_type denominator =
                coefficients_.first+coefficients_.second*problem_->jacobian_;
            solution(0) = right_hand_side(0)/denominator;
            return std::isfinite(solution(0));
        }

    private:
        const scalar_linear_problem* problem_;
        std::pair<scalar_type, scalar_type> coefficients_{1, 0};
    };

private:
    scalar_type jacobian_ = 0;
};

template<class VectorOperations>
class scalar_cubic_problem
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void F(
        const scalar_type,
        const vector_type& state,
        const scalar_type,
        vector_type& output) const
    {
        output(0) = -state(0)*state(0)*state(0);
    }

    void set_linearization_point(const vector_type& state, const scalar_type)
    {
        jacobian_ = -scalar_type(3)*state(0)*state(0);
    }

    class linear_operator
    {
    public:
        explicit linear_operator(const scalar_cubic_problem* problem): problem_(problem)
        {}

        void set_aE_plus_bA(const std::pair<scalar_type, scalar_type>& coefficients)
        {
            coefficients_ = coefficients;
        }

        bool solve(const vector_type& right_hand_side, vector_type& solution) const
        {
            const scalar_type denominator =
                coefficients_.first+coefficients_.second*problem_->jacobian_;
            solution(0) = right_hand_side(0)/denominator;
            return std::isfinite(solution(0));
        }

    private:
        const scalar_cubic_problem* problem_;
        std::pair<scalar_type, scalar_type> coefficients_{1, 0};
    };

private:
    scalar_type jacobian_ = 0;
};

long double observed_order(const long double coarse_error, const long double fine_error)
{
    return std::log(coarse_error/fine_error)/std::log(2.0L);
}

} // namespace

int main()
{
    using scalar_type = double;
    using vector_operations_type = scfd_serial_cpu_vector_operations<scalar_type>;
    using log_type = scfd::utils::log_std;
    using time_steppers::tests::integrate_legacy_implicit_fixed;

    time_steppers::tests::test_context test;
    vector_operations_type vector_operations(1);
    log_type log;
    log.set_verbosity(0);

    scalar_linear_problem<vector_operations_type> linear_problem;
    typename scalar_linear_problem<vector_operations_type>::linear_operator linear_operator(
        &linear_problem);
    struct method_expectation
    {
        const char* name;
        unsigned int order;
    };
    const std::vector<method_expectation> methods = {
        {"IE", 1},
        {"IM", 2},
        {"CN", 2},
        {"SDIRK2A1", 2},
        {"ESDIRK3A2", 2},
        {"SDIRK3A3", 3}
    };

    constexpr scalar_type rate = -0.8;
    constexpr scalar_type final_time = 1;
    const scalar_type exact = std::exp(rate*final_time);
    for(const auto& method: methods)
    {
        const auto coarse = integrate_legacy_implicit_fixed<1>(
            vector_operations,
            linear_problem,
            linear_operator,
            log,
            method.name,
            std::array<scalar_type, 1>{{1}},
            rate,
            0,
            final_time,
            10);
        const auto fine = integrate_legacy_implicit_fixed<1>(
            vector_operations,
            linear_problem,
            linear_operator,
            log,
            method.name,
            std::array<scalar_type, 1>{{1}},
            rate,
            0,
            final_time,
            20);
        const long double coarse_error = std::abs(static_cast<long double>(coarse[0])-exact);
        const long double fine_error = std::abs(static_cast<long double>(fine[0])-exact);
        const long double order = observed_order(coarse_error, fine_error);
        test.check(
            order >= static_cast<long double>(method.order)-0.25L,
            std::string(method.name) + " observed order=" +
                std::to_string(static_cast<double>(order)));
        test.check(
            fine_error < coarse_error,
            std::string(method.name) + " error decreases under refinement");
    }

    scalar_cubic_problem<vector_operations_type> cubic_problem;
    typename scalar_cubic_problem<vector_operations_type>::linear_operator cubic_operator(
        &cubic_problem);
    const auto cubic_coarse = integrate_legacy_implicit_fixed<1>(
        vector_operations,
        cubic_problem,
        cubic_operator,
        log,
        "SDIRK3A3",
        std::array<scalar_type, 1>{{1}},
        0,
        0,
        final_time,
        12);
    const auto cubic_fine = integrate_legacy_implicit_fixed<1>(
        vector_operations,
        cubic_problem,
        cubic_operator,
        log,
        "SDIRK3A3",
        std::array<scalar_type, 1>{{1}},
        0,
        0,
        final_time,
        24);
    const scalar_type cubic_exact = 1/std::sqrt(3.0);
    const long double cubic_order = observed_order(
        std::abs(static_cast<long double>(cubic_coarse[0])-cubic_exact),
        std::abs(static_cast<long double>(cubic_fine[0])-cubic_exact));
    test.check(
        cubic_order >= 2.7L,
        "SDIRK3A3 nonlinear observed order=" +
            std::to_string(static_cast<double>(cubic_order)));

    std::cout << "Implicit RK analytical checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
