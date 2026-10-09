#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef TEST_VECTOR_BACKEND_CUDA
#include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
#elif defined(TEST_VECTOR_BACKEND_OMP)
#include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
#else
#include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
#endif
#include <common/scfd_vector_operations.h>
#include <time_stepper/integration/time_integrator.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

using operations_type = scfd_vector_operations<backend_type, double>;
using vector_type = operations_type::vector_type;
using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
using integration_status = nmfd::time_steppers::integration_status;
using step_status = nmfd::time_steppers::single_step_status;
using adaptation_status = nmfd::time_steppers::adaptation_status;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

double read(const operations_type& operations, const vector_type& state)
{
    const auto view = operations.view(state);
    return view(0);
}

struct linear_growth
{
    operations_type& operations;
    std::size_t calls = 0;

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.assign(in, out);
    }
};

struct forced_growth
{
    operations_type& operations;
    std::size_t calls = 0;
    double time = 0;

    void set_time(double value)
    {
        time = value;
    }

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.assign(in, out);
        operations.add_mul_scalar(time, 1., out);
    }
};

struct quadratic_growth
{
    operations_type& operations;
    std::size_t calls = 0;

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.mul_pointwise(1., in, 1., in, out);
    }
};

struct limited_rate
{
    operations_type& operations;
    double time = 0;

    void set_time(double value)
    {
        time = value;
    }

    void apply(const vector_type&, vector_type& out)
    {
        operations.assign_scalar(time > .75 ? std::numeric_limits<double>::infinity() : 1., out);
    }
};

template<class Problem>
void check_convergence(
    operations_type& operations, Problem& problem, double end_time, double coarse_step, double exact, const char* name)
{
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations);
    initial.start_use();
    result.start_use();
    operations.assign_scalar(1., *initial);
    double errors[3]{};
    for (int level = 0; level < 3; ++level)
    {
        adaptation_type adaptation({coarse_step / (1 << level)});
        using step_type =
            nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, Problem, adaptation_type>;
        step_type step(operations, problem, adaptation, {"DOP853"});
        nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
        problem.calls = 0;
        integrator.set_time_interval(0, end_time);
        integrator.apply(*initial, *result);
        require(integrator.get_status() == integration_status::completed && integrator.get_final_time() == end_time,
            "DOP853 analytical integration failed");
        require(problem.calls == step.table().size() * integrator.get_steps(),
            "Fixed DOP853 evaluates thirteen derivatives per step, without interpolation stages");
        errors[level] = std::abs(read(operations, *result) - exact);
        require(std::isfinite(errors[level]) && errors[level] > 64 * std::numeric_limits<double>::epsilon(),
            "DOP853 convergence measurements must stay above the roundoff floor");
    }
    for (int level = 0; level < 2; ++level)
    {
        const auto order = std::log2(errors[level] / errors[level + 1]);
        require(order > 7.3 && order < 8.7, "DOP853 analytical refinement must approach eighth order");
        std::cout << "DOP853 " << name << ": errors " << errors[level] << " -> " << errors[level + 1]
                  << ", observed order " << order << '\n';
    }
}

void check_lifecycle(operations_type& operations)
{
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations);
    initial.start_use();
    result.start_use();
    operations.assign_scalar(1., *initial);
    linear_growth problem{operations};
    adaptation_type adaptation({.5});
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, linear_growth, adaptation_type>;
    step_type step(operations, problem, adaptation, {"DOP853"});
    step.apply(*initial, *result);
    require(step.get_status() == step_status::converged && problem.calls == 13 && step.dense_output_order() == 0,
        "Fixed DOP853 prepares a candidate without native interpolation or an error estimate");
    bool refused = false;
    try
    {
        step.error_estimate();
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    require(refused, "Fixed DOP853 must not expose a fabricated embedded error vector");
    refused = false;
    try
    {
        step.apply(*initial, *result);
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    require(refused, "Pending DOP853 candidates must be finalized before reuse");
    step.finalize(adaptation_status::accepted, .5, *result);

    nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
    integrator.set_time_interval(0, .75);
    for (int run = 0; run < 2; ++run)
    {
        integrator.apply(*initial, *result);
        require(integrator.get_status() == integration_status::completed && integrator.get_steps() == 2 &&
                    integrator.get_final_time() == .75 && std::abs(read(operations, *result) - std::exp(.75)) < 1e-8,
            "DOP853 restart and clipped forward endpoint");
    }
    operations.assign_scalar(std::exp(.75), *initial);
    integrator.set_time_interval(.75, 0);
    integrator.apply(*initial, *result);
    require(integrator.get_status() == integration_status::completed && integrator.get_final_time() == 0 &&
                std::abs(read(operations, *result) - 1) < 1e-8,
        "DOP853 backward integration");
    operations.assign_scalar(1., *initial);
    integrator.set_time_interval(0, .75);
    integrator.apply(*initial, *initial);
    require(integrator.get_status() == integration_status::completed &&
                std::abs(read(operations, *initial) - std::exp(.75)) < 1e-8,
        "DOP853 aliased integration");

    operations.assign_scalar(1., *initial);
    nmfd::time_steppers::integration::time_integrator<operations_type, step_type> budgeted(operations, step, {1});
    budgeted.apply(*initial, *result);
    require(budgeted.get_status() == integration_status::attempt_limit_reached && budgeted.get_final_time() == .5 &&
                std::abs(read(operations, *result) - std::exp(.5)) < 1e-8,
        "DOP853 step budget preserves the accepted state");

    using adaptive_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
    adaptive_type adaptive(operations);
    nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, linear_growth, adaptive_type> unsupported(
        operations, problem, adaptive, {"DOP853"});
    operations.assign_scalar(77., *result);
    const auto calls = problem.calls;
    unsupported.apply(*initial, *result);
    require(unsupported.get_status() == step_status::error_estimate_unavailable && read(operations, *result) == 77 &&
                problem.calls == calls,
        "Adaptive DOP853 must fail before evaluating or publishing a candidate");

    step.reset();
    step.set_time(1e30);
    step.set_target_time(2e30);
    step.apply(*initial, *result);
    require(step.get_status() == step_status::step_size_underflow && read(operations, *result) == 77,
        "DOP853 detects floating-point time stagnation without changing output");

    limited_rate bad{operations};
    using failing_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, limited_rate, adaptation_type>;
    failing_step failed(operations, bad, adaptation, {"DOP853"});
    nmfd::time_steppers::integration::time_integrator<operations_type, failing_step> failure(operations, failed);
    failure.apply(*initial, *initial);
    require(failure.get_status() == integration_status::step_failure &&
                failed.get_status() == step_status::failed_nonfinite && failure.get_final_time() == .5 &&
                failure.get_steps() == 1 && std::abs(read(operations, *initial) - 1.5) < 1e-14,
        "DOP853 nonfinite stages preserve the last accepted state, including aliased output");
}

int main()
try
{
    backend_type::init_device();
    operations_type operations(1);
    linear_growth linear{operations};
    forced_growth forced{operations};
    quadratic_growth nonlinear{operations};
    check_convergence(operations, linear, 2, 1, std::exp(2.), "u'=u");
    check_convergence(operations, forced, 2, 1, 2 * std::exp(2.) - 3, "u'=u+t");
    check_convergence(operations, nonlinear, .5, .25, 2, "u'=u^2");
    check_lifecycle(operations);
    std::cout << "Fixed DOP853 analytical order, endpoint, restart, capability and failure tests: PASS\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
