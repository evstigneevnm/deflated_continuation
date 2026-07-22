#include <cmath>
#include <iostream>
#include <stdexcept>

#include <continuation/corrector_retry_policy.h>
#include <continuation/predictor_adaptive.h>
#include <continuation/progress_monitor.h>

namespace
{

void require(const bool condition, const char* message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

bool close(const double left, const double right)
{
    return std::abs(left-right) <= 1.0e-14;
}

struct unused_vector_operations
{
    using scalar_type = double;
    using vector_type = int;
};

struct unused_log
{
    template<class... Args>
    void info_f(const char*, Args...)
    {
    }
};

void test_legacy_success_growth_mapping()
{
    continuation::predictor_adaptive<unused_vector_operations, unused_log> predictor(
        nullptr,
        nullptr,
        0.01,
        0.02,
        0.8,
        0.0777,
        12);
    predictor.set_verbose(false);

    for(unsigned int step = 0; step < 5; ++step)
    {
        predictor.begin_step();
        predictor.accept_step(false);
    }
    require(close(predictor.get_ds(), 0.01),
            "legacy dS grew before six clean continuation steps");

    predictor.begin_step();
    predictor.accept_step(false);
    require(close(predictor.get_ds(), 0.0125),
            "legacy dS did not preserve its historical 1.25 growth factor");
}

void test_bounded_monotone_retries()
{
    continuation::corrector_retry_policy<double> policy;
    policy.maximum_retries = 3;
    policy.failure_reduction_factor = 0.5;
    policy.minimum_step_ratio = 0.1;
    policy.successes_before_growth = 2;
    policy.success_growth_factor = 2.0;

    continuation::adaptive_step_controller<double> controller(0.1, 0.4, policy);
    controller.begin_step();
    require(controller.retry_after_failure() == continuation::step_retry_result::retry,
            "first retry was rejected");
    require(close(controller.step(), 0.05), "first retry did not reduce dS monotonically");
    require(controller.retry_after_failure() == continuation::step_retry_result::retry,
            "second retry was rejected");
    require(close(controller.step(), 0.025), "second retry produced the wrong dS");
    require(controller.retry_after_failure() == continuation::step_retry_result::retry,
            "third retry was rejected");
    require(close(controller.step(), 0.0125), "third retry produced the wrong dS");
    require(controller.retry_after_failure() == continuation::step_retry_result::retry_limit,
            "retry limit was not enforced");

    controller.accept_step(true);
    controller.begin_step();
    require(close(controller.step(), 0.0125), "a recovered step reset dS to its initial value");

    controller.accept_step(false);
    controller.begin_step();
    controller.accept_step(false);
    require(close(controller.step(), 0.025), "first-attempt success streak did not grow dS");

    controller.reset_semicurve();
    require(close(controller.step(), 0.1), "semicurve reset did not restore initial dS");
}

void test_minimum_step()
{
    continuation::corrector_retry_policy<double> policy;
    policy.maximum_retries = 20;
    policy.failure_reduction_factor = 0.5;
    policy.minimum_step_size = 0.03;

    continuation::adaptive_step_controller<double> controller(0.1, 0.4, policy);
    controller.begin_step();
    require(controller.retry_after_failure() == continuation::step_retry_result::retry,
            "minimum-step test rejected first retry");
    require(close(controller.step(), 0.05), "minimum-step test produced wrong first dS");
    require(controller.retry_after_failure() == continuation::step_retry_result::retry,
            "minimum-step test rejected the floor retry");
    require(close(controller.step(), 0.03), "minimum dS floor was not applied");
    require(controller.retry_after_failure() == continuation::step_retry_result::minimum_step,
            "minimum dS did not terminate retries");
}

void test_event_refinement_reduces_future_steps()
{
    continuation::corrector_retry_policy<double> policy;
    policy.maximum_retries = 1;
    policy.minimum_step_size = 0.01;

    continuation::adaptive_step_controller<double> controller(0.1, 0.4, policy);
    require(controller.reduce_next_step(0.25) == continuation::step_retry_result::retry,
            "branch-event refinement rejected a valid reduction");
    require(close(controller.step(), 0.025),
            "branch-event refinement produced the wrong dS");
    require(controller.retries() == 0,
            "branch-event refinement consumed the nonlinear retry budget");
    require(controller.reduce_next_step(0.25) == continuation::step_retry_result::retry,
            "branch-event refinement did not reach its step floor");
    require(close(controller.step(), 0.01),
            "branch-event refinement did not clamp to the minimum dS");
    require(controller.reduce_next_step(0.25) == continuation::step_retry_result::minimum_step,
            "branch-event refinement did not report its minimum dS");
}

void test_bracket_retry_sets_interior_step()
{
    continuation::corrector_retry_policy<double> policy;
    policy.maximum_retries = 2;
    policy.minimum_step_size = 0.01;

    continuation::adaptive_step_controller<double> controller(0.1, 0.4, policy);
    controller.begin_step();
    require(controller.retry_at_step(0.04) == continuation::step_retry_result::retry,
            "event bracket rejected an interior step");
    require(close(controller.step(), 0.04),
            "event bracket did not set its requested step");
    require(controller.retry_at_step(0.07) == continuation::step_retry_result::retry,
            "event bracket could not move upward inside the bracket");
    require(close(controller.step(), 0.07),
            "event bracket upward trial used the wrong step");
    require(controller.retry_at_step(0.08) == continuation::step_retry_result::retry_limit,
            "event bracket did not share the nonlinear retry limit");
}

void test_progress_monitor()
{
    continuation::progress_monitor_policy<double> policy;
    policy.enabled = true;
    policy.window_size = 3;
    policy.minimum_window_progress_ratio = 0.5;

    continuation::progress_monitor<double> monitor;
    monitor.configure(policy, 0.1);
    require(!monitor.observe(0.01), "progress monitor stopped before its window was full");
    require(!monitor.observe(0.01), "progress monitor stopped before its window was full");
    require(monitor.observe(0.01), "progress monitor did not detect a stalled window");

    monitor.reset();
    require(!monitor.observe(0.02), "moving window stopped too early");
    require(!monitor.observe(0.02), "moving window stopped too early");
    require(!monitor.observe(0.02), "adequate accumulated progress was marked as stalled");
}

} // namespace

int main()
{
    try
    {
        test_bounded_monotone_retries();
        test_minimum_step();
        test_event_refinement_reduces_future_steps();
        test_bracket_retry_sets_interior_step();
        test_progress_monitor();
        test_legacy_success_growth_mapping();
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
