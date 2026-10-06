#include <algorithm>
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
#include <time_stepper/runge_kutta/explicit_time_step.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_integrator.h>

using operations_type = scfd_vector_operations<backend_type, double>;
using vector_type = operations_type::vector_type;
using constant_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
using adaptive_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template<class T>
void test_error_scaling()
{
    using vector_operations_type = scfd_vector_operations<backend_type, T>;
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<vector_operations_type>;
    using status_type = nmfd::time_steppers::adaptation_status;
    vector_operations_type operations(3);
    nmfd::detail::vector_wrap<vector_operations_type> previous(operations), candidate(operations), error(operations);
    previous.start_use();
    candidate.start_use();
    error.start_use();
    const scfd::static_vec::vec<T, 3> x{0, -4, T(.25)}, y{0, 1, -8};
    scfd::static_vec::vec<T, 3> e, actual;
    operations.set(x.d, *previous);
    operations.set(y.d, *candidate);
    typename adaptation_type::params p;
    p.absolute_tolerance = T(.02);
    p.relative_tolerance = T(.01);
    adaptation_type adaptation(operations, p);
    const T tolerance = 64 * std::numeric_limits<T>::epsilon();
    T next = 0;
    for (const T ratio : {T(.25), T(4)})
    {
        T expected = 0;
        for (int i = 0; i < 3; ++i)
        {
            const T scale =
                std::max(p.absolute_tolerance, p.relative_tolerance * std::max(std::abs(x[i]), std::abs(y[i])));
            e[i] = -ratio * scale;
            expected = std::max(expected, std::abs(e[i]) / scale);
        }
        operations.set(e.d, *error);
        const nmfd::time_steppers::detail::scaled_error_mapping<T> mapping{p.absolute_tolerance / p.relative_tolerance};
        const T measured =
            operations.transform_reduce_max(mapping, *error, *previous, *candidate) / p.relative_tolerance;
        require(std::abs(measured - expected) <= tolerance * expected,
            "Matlab scaling: absolute/previous/candidate weights");
        adaptation.reset();
        const auto decision = adaptation.assess(0, T(.1), *previous, *candidate, next, 2, &*error);
        require(decision == (ratio > 1 ? status_type::rejected : status_type::accepted), "Scaled error acceptance");
        require(std::abs(next - T(.1) * T(.8) / std::sqrt(expected)) < tolerance, "Scaled error next step");
        operations.get(*error, actual.d);
        for (int i = 0; i < 3; ++i)
        {
            require(actual[i] == e[i], "Scaling modified error");
        }
    }
    operations.get(*previous, actual.d);
    for (int i = 0; i < 3; ++i)
    {
        require(actual[i] == x[i], "Scaling modified previous state");
    }
    operations.get(*candidate, actual.d);
    for (int i = 0; i < 3; ++i)
    {
        require(actual[i] == y[i], "Scaling modified candidate state");
    }

    // A non-finite component must not disappear in the backend maximum reduction.
    for (int field = 0; field < 3; ++field)
    {
        for (const int position : {0, 2})
        {
            for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
            {
                auto hx = x, hy = y, he = e;
                (field == 0 ? he : field == 1 ? hx : hy)[position] = invalid;
                operations.set(hx.d, *previous);
                operations.set(hy.d, *candidate);
                operations.set(he.d, *error);
                adaptation.reset();
                require(adaptation.assess(0, T(.1), *previous, *candidate, next, 2, &*error) == status_type::rejected,
                    "Non-finite error/state must reject");
                require(std::isfinite(next) && next == T(.1) * T(.5), "Non-finite error reduces step");
            }
        }
    }
    operations.set(x.d, *previous);
    operations.set(y.d, *candidate);
    operations.assign_scalar(0, *error);
    adaptation.reset();
    require(
        adaptation.assess(0, T(.1), *previous, *candidate, next, 2, &*error) == status_type::accepted && next == T(.5),
        "Zero error accepts with bounded growth");
    require(adaptation.assess(0, T(.1), *previous, *candidate, next, 2) == status_type::failed &&
                adaptation.assess(0, T(.1), *previous, *candidate, next, 0, &*error) == status_type::failed,
        "Missing error/order must fail");
    for (const T invalid : {T(0), T(-1), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
    {
        for (int field = 0; field < 2; ++field)
        {
            auto invalid_params = p;
            (field == 0 ? invalid_params.absolute_tolerance : invalid_params.relative_tolerance) = invalid;
            bool refused = false;
            try
            {
                adaptation_type invalid_adaptation(operations, invalid_params);
            }
            catch (const std::invalid_argument&)
            {
                refused = true;
            }
            require(refused, "Invalid adaptation tolerance must fail");
        }
    }
}

// Count trajectory storage, excluding caller vectors and backend reduction scratch.
struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable std::size_t allocations = 0, live_vectors = 0;

    void start_use_vector(vector_type& x) const override
    {
        const bool allocate = x.is_free();
        operations_type::start_use_vector(x);
        if (allocate)
        {
            ++allocations;
            ++live_vectors;
        }
    }

    void free_vector(vector_type& x) const override
    {
        if (!x.is_free())
        {
            --live_vectors;
        }
        operations_type::free_vector(x);
    }
};

// Non-autonomous exact solution u(t)=2*exp(t)-t-1, u(0)=1.
struct forced_growth
{
    operations_type& operations;
    double time = 0;

    void set_time(double t)
    {
        time = t;
    }

    void apply(const vector_type& in, vector_type& out)
    {
        operations.assign(in, out);
        operations.add_mul_scalar(time, 1., out);
    }
};

struct constant_rate
{
    operations_type& operations;

    void apply(const vector_type&, vector_type& out)
    {
        operations.assign_scalar(1., out);
    }
};

struct limited_rate
{
    operations_type& operations;
    double time = 0;

    void set_time(double t)
    {
        time = t;
    }

    void apply(const vector_type&, vector_type& out)
    {
        operations.assign_scalar(time > .08 ? std::numeric_limits<double>::infinity() : 1., out);
    }
};

struct stop_at_section
{
    operations_type& operations;
    double start = 0;
    double* end = nullptr;
    nmfd::time_steppers::integration_status status = nmfd::time_steppers::integration_status::running;

    void set_time_interval(const double& t, double& next)
    {
        start = t;
        end = &next;
    }

    bool apply(const vector_type& in, vector_type& out)
    {
        if (start == 0)
        {
            status = nmfd::time_steppers::integration_status::running;
        }
        if (*end < .35)
        {
            return false;
        }
        *end = .35;
        operations.assign(in, out);
        operations.add_mul_scalar(*end - start, 1., out);
        status = nmfd::time_steppers::integration_status::stopped_by_external_operation;
        return true;
    }

    nmfd::time_steppers::integration_status get_status() const
    {
        return status;
    }
};

struct failed_callback
{
    operations_type& operations;
    double fail_after = 0, start = 0;

    void set_time_interval(const double& time, double&)
    {
        start = time;
    }

    bool apply(const vector_type&, vector_type& out)
    {
        if (start < fail_after)
        {
            return false;
        }
        operations.assign_scalar(42., out);
        return true;
    }

    nmfd::time_steppers::integration_status get_status() const
    {
        return start < fail_after ? nmfd::time_steppers::integration_status::running
                                  : nmfd::time_steppers::integration_status::external_operation_failure;
    }
};

void test_storage(const vector_type& initial, vector_type& result)
{
    counting_operations operations(1);
    forced_growth problem{operations};
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<counting_operations>;
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, adaptation_type>;
    {
        adaptation_type adaptation(operations);
        require(operations.allocations == 0, "Adaptation must not allocate trajectory vectors");
        step_type step(operations, problem, adaptation);
        require(operations.live_vectors == step.table().size() + 2, "RK storage: derivatives, work, embedded error");
        nmfd::time_steppers::integration::time_integrator<counting_operations, step_type> integrator(operations, step);
        require(operations.live_vectors == 10, "DOPRI54 integration must own 10 vectors, not 14");
        integrator.set_time_interval(0, .25);
        for (int run = 0; run < 2; ++run)
        {
            integrator.apply(initial, result);
            require(
                integrator.get_status() == nmfd::time_steppers::integration_status::completed, "Counted integration");
            require(operations.allocations == 10, "No vector allocations during integration or restart");
        }
    }
    require(operations.live_vectors == 0, "Embedded RK storage release");
    {
        using fixed_step =
            nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, constant_type>;
        constant_type constant({.1});
        fixed_step step(operations, problem, constant, {"EE"});
        require(operations.live_vectors == 2, "Unembedded RK must not allocate an error vector");
        nmfd::time_steppers::integration::time_integrator<counting_operations, fixed_step> integrator(operations, step);
        integrator.set_time_interval(0, .2);
        integrator.apply(initial, result);
        require(operations.live_vectors == 3 && operations.allocations == 13, "Unembedded integration storage");
        bool refused = false;
        try
        {
            step.error_estimate();
        }
        catch (const std::logic_error&)
        {
            refused = true;
        }
        require(refused, "Unembedded error access must be rejected");
    }
    require(operations.live_vectors == 0, "Unembedded RK storage release");
}

void test_bs32_error(operations_type& operations, const vector_type& initial, vector_type& result)
{
    forced_growth problem{operations};
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, constant_type>;
    double errors[2]{};
    for (int level = 0; level < 2; ++level)
    {
        const double h = .1 / (level + 1);
        constant_type adaptation({h});
        step_type step(operations, problem, adaptation, {"BS32"});
        step.apply(initial, result);
        require(step.get_status() == nmfd::time_steppers::single_step_status::converged, "BS32 fixed step");
        double value = 0;
        operations.get(result, &value);
        require(std::abs(value - (1 + h + h * h + h * h * h / 3)) < 2e-14, "BS32 primary update for u'=u+t, u(0)=1");
        operations.get(step.error_estimate(), &value);
        require(std::abs(value + h * h * h * (1 + h) / 24) < 2e-14,
            "BS32 signed embedded state error includes the step-size factor exactly once");
        errors[level] = std::abs(value);
        step.finalize(nmfd::time_steppers::adaptation_status::accepted, h, result);
    }
    require(errors[0] / errors[1] > 8 && errors[0] / errors[1] < 8.5, "BS32 embedded state error scales as h^3");

    adaptive_type::params p;
    p.initial_step = .5;
    p.relative_tolerance = 1e-9;
    p.absolute_tolerance = 1e-11;
    adaptive_type adaptation(operations, p);
    using adaptive_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptive_type>;
    adaptive_step step(operations, problem, adaptation, {"BS32"});
    step.apply(initial, result);
    require(step.get_status() == nmfd::time_steppers::single_step_status::converged && step.get_attempts() > 1,
        "BS32 adaptive rejection converges from the original state");
    double value = 0;
    operations.get(initial, &value);
    require(value == 1, "BS32 rejection preserves the input state");
    step.finalize(nmfd::time_steppers::adaptation_status::accepted, step.get_dt(), result);
}

static_assert(nmfd::time_steppers::runge_kutta::detail::has_apply<constant_rate, vector_type>::value);
static_assert(!nmfd::time_steppers::runge_kutta::detail::has_set_time<constant_rate, double>::value);
static_assert(!nmfd::time_steppers::runge_kutta::detail::has_apply<int, vector_type>::value);

int main()
try
{
    backend_type::init_device();
    test_error_scaling<float>();
    test_error_scaling<double>();
    operations_type operations(1);
    nmfd::detail::vector_wrap<operations_type> initial_wrap(operations), result_wrap(operations),
        coarse_wrap(operations);
    initial_wrap.start_use();
    result_wrap.start_use();
    coarse_wrap.start_use();
    auto& initial = *initial_wrap;
    auto& result = *result_wrap;
    auto& coarse = *coarse_wrap;
    operations.assign_scalar(1., initial);
    test_storage(initial, result);
    test_bs32_error(operations, initial, result);
    forced_growth problem{operations};
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, constant_type>;
    const auto exact = 2 * std::exp(.5) - 1.5;
    for (const auto* name : {"EE", "HE", "BS32", "RK33SSP", "RK43SSP", "RK64SSP", "DOPRI54"})
    {
        double errors[2]{};
        for (int level = 0; level < 2; ++level)
        {
            constant_type adaptation({.1 / (level + 1)});
            step_type step(operations, problem, adaptation, {name});
            nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
            integrator.set_time_interval(0, .5);
            integrator.apply(initial, result);
            double value = 0;
            operations.get(result, &value);
            require(
                integrator.get_status() == nmfd::time_steppers::integration_status::completed, "Forward integration");
            errors[level] = std::abs(value - exact);
        }
        const auto order = nmfd::time_steppers::runge_kutta::make_butcher_table(name).order();
        require(errors[0] / errors[1] > std::pow(2., order - .6), "Non-autonomous convergence order");
    }
    constant_type constant({.1});
    step_type step(operations, problem, constant);
    nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
    integrator.set_time_interval(.5, 0);
    operations.assign_scalar(exact, coarse);
    integrator.apply(coarse, result);
    double value = 0;
    operations.get(result, &value);
    require(std::abs(value - 1) < 1e-8 && integrator.get_final_time() == 0, "Backward integration");
    integrator.set_time_interval(0, .23);
    integrator.apply(initial, initial);
    operations.get(initial, &value);
    require(integrator.get_final_time() == .23 && std::abs(value - (2 * std::exp(.23) - 1.23)) < 1e-8,
        "Aliased integration/clipped endpoint");
    integrator.set_time_interval(.23, .23);
    integrator.apply(initial, result);
    double unchanged = 0;
    operations.get(result, &unchanged);
    require(unchanged == value && integrator.get_steps() == 0 &&
                integrator.get_status() == nmfd::time_steppers::integration_status::completed,
        "Zero-length integration");
    operations.assign_scalar(1., initial);

    // A pending result and its embedded error must be explicitly finalized.
    step.reset();
    step.set_time(0);
    step.set_target_time(1);
    step.apply(initial, result);
    require(step.get_status() == nmfd::time_steppers::single_step_status::converged, "Pending step");
    operations.get(step.error_estimate(), &value);
    const auto first_error = std::abs(value);
    bool refused = false;
    try
    {
        step.apply(initial, result);
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    require(refused, "No overwrite of a pending step");
    step.finalize(nmfd::time_steppers::adaptation_status::accepted, .1, result);
    constant_type half({.05});
    step_type half_step(operations, problem, half);
    half_step.set_target_time(1);
    half_step.apply(initial, coarse);
    operations.get(half_step.error_estimate(), &value);
    require(first_error / std::abs(value) > 25 && first_error / std::abs(value) < 40,
        "DP embedded STATE error scales as h^5, not h^4 or h^6");
    half_step.finalize(nmfd::time_steppers::adaptation_status::accepted, .05, coarse);

    adaptive_type::params p;
    p.initial_step = 1;
    p.relative_tolerance = 1e-12;
    p.absolute_tolerance = 1e-14;
    adaptive_type adaptation(operations, p);
    using adaptive_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptive_type>;
    adaptive_step adaptive(operations, problem, adaptation);
    adaptive.set_target_time(1);
    adaptive.apply(initial, result);
    require(adaptive.get_status() == nmfd::time_steppers::single_step_status::converged && adaptive.get_attempts() > 1,
        "Adaptive rejection must retry from the original state");
    operations.get(initial, &value);
    require(value == 1, "Rejection modified input");
    adaptive.finalize(nmfd::time_steppers::adaptation_status::accepted, adaptive.get_dt(), result);
    adaptive_step no_error(operations, problem, adaptation, {"EE"});
    operations.assign_scalar(77., result);
    no_error.apply(initial, result);
    operations.get(result, &value);
    require(value == 77 && no_error.get_status() == nmfd::time_steppers::single_step_status::error_estimate_unavailable,
        "Do not adapt an unembedded method using a zero error vector");
    adaptive_step limited(operations, problem, adaptation, {"DOPRI54", 1});
    limited.reset();
    limited.apply(initial, result);
    require(
        limited.get_status() == nmfd::time_steppers::single_step_status::attempt_limit_reached, "Bounded rejection");
    operations.get(result, &value);
    require(value == 77, "Failed attempt must not write output");

    limited_rate nonfinite{operations};
    p.initial_step = .1;
    adaptive_type recovery(operations, p);
    nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, limited_rate, adaptive_type> recover(
        operations, nonfinite, recovery);
    recover.apply(initial, result);
    require(recover.get_status() == nmfd::time_steppers::single_step_status::converged && recover.get_dt() <= .08,
        "Nonfinite-stage recovery");
    recover.finalize(nmfd::time_steppers::adaptation_status::accepted, recover.get_dt(), result);

    constant_rate rate{operations};
    using rate_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, constant_rate, constant_type>;
    rate_step autonomous(operations, rate, constant);
    stop_at_section stop{operations};
    nmfd::time_steppers::integration::time_integrator<operations_type, rate_step, stop_at_section> stopped(
        operations, autonomous, {}, &stop);
    operations.assign_scalar(0., initial);
    for (int run = 0; run < 2; ++run)
    {
        stopped.apply(initial, result);
        operations.get(result, &value);
        require(stopped.get_status() == nmfd::time_steppers::integration_status::stopped_by_external_operation &&
                    stopped.get_final_time() == .35 && std::abs(value - .35) < 1e-14,
            "Refined callback commit/reset");
    }
    failed_callback failure{operations};
    nmfd::time_steppers::integration::time_integrator<operations_type, rate_step, failed_callback> failed(
        operations, autonomous, {}, &failure);
    failed.apply(initial, result);
    operations.get(result, &value);
    require(failed.get_status() == nmfd::time_steppers::integration_status::external_operation_failure &&
                failed.get_final_time() == 0 && value == 0,
        "Failed callback rollback");
    failure.fail_after = .1;
    failed.apply(initial, initial);
    operations.get(initial, &value);
    require(failed.get_status() == nmfd::time_steppers::integration_status::external_operation_failure &&
                failed.get_final_time() == .1 && std::abs(value - .1) < 1e-14,
        "Aliased callback failure must preserve the last accepted state");
    operations.assign_scalar(0., initial);
    nmfd::time_steppers::integration::time_integrator<operations_type, rate_step> budgeted(operations, autonomous, {1});
    budgeted.apply(initial, result);
    operations.get(result, &value);
    require(budgeted.get_status() == nmfd::time_steppers::integration_status::attempt_limit_reached &&
                budgeted.get_final_time() == .1 && std::abs(value - .1) < 1e-14,
        "Step budget preserves committed state");
    using failing_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, limited_rate, constant_type>;
    failing_step fail_later(operations, nonfinite, half);
    nmfd::time_steppers::integration::time_integrator<operations_type, failing_step> numerical_failure(
        operations, fail_later);
    numerical_failure.apply(initial, result);
    operations.get(result, &value);
    require(numerical_failure.get_status() == nmfd::time_steppers::integration_status::step_failure &&
                numerical_failure.get_final_time() == .05 && std::abs(value - .05) < 1e-14,
        "Numerical failure must preserve the last accepted state");
    autonomous.reset();
    autonomous.set_time(1e30);
    autonomous.set_target_time(2e30);
    autonomous.apply(initial, result);
    require(autonomous.get_status() == nmfd::time_steppers::single_step_status::step_size_underflow,
        "Floating-point stagnation");
    refused = false;
    try
    {
        rate_step invalid(operations, rate, constant, {"IE"});
    }
    catch (const std::invalid_argument&)
    {
        refused = true;
    }
    require(refused, "Implicit table must not enter explicit kernel");
    std::cout << "Explicit RK order, storage, lifecycle, rejection, callback and failure tests: PASS\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
