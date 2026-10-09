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
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

using adaptation_status = nmfd::time_steppers::adaptation_status;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template<class T>
void check_assessment()
{
    using operations_type = scfd_vector_operations<backend_type, T>;
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;
    operations_type operations(3);
    nmfd::detail::vector_wrap<operations_type> previous(operations), candidate(operations), error(operations),
        error3(operations);
    previous.start_use();
    candidate.start_use();
    error.start_use();
    error3.start_use();
    const scfd::static_vec::vec<T, 3> x{0, -4, T(.25)}, y{0, 1, -8};
    const scfd::static_vec::vec<T, 3> ratios{T(-.25), T(.5), T(-.75)}, ratios3{1, -2, 4};
    typename adaptation_type::params p;
    p.absolute_tolerance = T(.02);
    p.relative_tolerance = T(.01);
    adaptation_type adaptation(operations, p);
    scfd::static_vec::vec<T, 3> e, e3;
    T sum = 0, sum3 = 0;
    for (int i = 0; i < 3; ++i)
    {
        const auto scale = p.absolute_tolerance + p.relative_tolerance * std::max(std::abs(x[i]), std::abs(y[i]));
        e[i] = ratios[i] * scale;
        e3[i] = ratios3[i] * scale;
        sum += ratios[i] * ratios[i];
        sum3 += ratios3[i] * ratios3[i];
    }
    operations.set(x.d, *previous);
    operations.set(y.d, *candidate);
    operations.set(e.d, *error);
    operations.set(e3.d, *error3);
    const auto rms = std::sqrt(sum / 3);
    const auto tolerance = T(128) * std::numeric_limits<T>::epsilon();
    T next = 0;
    for (const unsigned int order : {2u, 3u, 5u, 8u})
    {
        adaptation.reset();
        require(adaptation.assess(0, T(.1), *previous, *candidate, next, order, &*error) == adaptation_status::accepted,
            "SciPy additive RMS assessment accepts a small state error");
        const auto expected = T(.1) * T(.9) * std::pow(rms, -T(1) / order);
        require(std::abs(next - expected) < tolerance, "SciPy power-law exponent and additive scaling");
    }
    adaptation.reset();
    const auto combined = sum / std::sqrt(T(3) * (sum + T(.01) * sum3));
    require(
        adaptation.assess(0, T(.1), *previous, *candidate, next, 8, &*error, &*error3) == adaptation_status::accepted,
        "SciPy DOP853 assessment accepts both state-error estimates");
    require(std::abs(next - T(.1) * T(.9) * std::pow(combined, -T(1) / 8)) < tolerance,
        "DOP853 combined assessment does not multiply state errors by dt twice");
    for (int field = 0; field < 4; ++field)
    {
        const auto view = field == 0   ? operations.view(*previous)
                          : field == 1 ? operations.view(*candidate)
                          : field == 2 ? operations.view(*error)
                                       : operations.view(*error3);
        const auto& expected = field == 0 ? x : field == 1 ? y : field == 2 ? e : e3;
        for (int i = 0; i < 3; ++i)
        {
            require(view(i) == expected[i], "SciPy assessment preserves all supplied vectors");
        }
    }
    for (int field = 0; field < 4; ++field)
    {
        for (int index = 0; index < 3; ++index)
        {
            for (const T invalid : {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
            {
                auto hx = x, hy = y, he = e, he3 = e3;
                (field == 0 ? hx : field == 1 ? hy : field == 2 ? he : he3)[index] = invalid;
                operations.set(hx.d, *previous);
                operations.set(hy.d, *candidate);
                operations.set(he.d, *error);
                operations.set(he3.d, *error3);
                adaptation.reset();
                require(adaptation.assess(0, T(.1), *previous, *candidate, next, 8, &*error, &*error3) ==
                                adaptation_status::rejected &&
                            std::abs(next - T(.02)) < tolerance,
                    "Nonfinite states and either estimate reject with bounded shrinkage");
            }
        }
    }
}

template<class T>
void check_control()
{
    using operations_type = scfd_vector_operations<backend_type, T>;
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;
    operations_type operations(3);
    nmfd::detail::vector_wrap<operations_type> state(operations), error(operations), error3(operations);
    state.start_use();
    error.start_use();
    error3.start_use();
    operations.assign_scalar(0, *state);
    typename adaptation_type::params p;
    p.absolute_tolerance = 1;
    p.relative_tolerance = T(.1);
    p.initial_step = T(.1);
    adaptation_type adaptation(operations, p);
    const auto tolerance = T(128) * std::numeric_limits<T>::epsilon();
    T next = 0;
    operations.assign_scalar(1, *error);
    require(adaptation.assess(0, T(.1), *state, *state, next, 5, &*error) == adaptation_status::rejected &&
                std::abs(next - T(.09)) < tolerance,
        "SciPy rejects the exact unit-error boundary");
    adaptation.update(adaptation_status::rejected, 0, T(.1), *state);
    operations.assign_scalar(0, *error);
    require(adaptation.assess(0, T(.09), *state, *state, next, 5, &*error) == adaptation_status::accepted &&
                std::abs(next - T(.09)) < tolerance,
        "SciPy forbids growth immediately after rejection");
    adaptation.update(adaptation_status::accepted, T(.09), T(.09), *state);
    require(adaptation.assess(T(.09), T(.09), *state, *state, next, 5, &*error) == adaptation_status::accepted &&
                std::abs(next - T(.9)) < tolerance,
        "Zero error restores growth up to factor ten after commitment");
    adaptation.update(adaptation_status::accepted_modified, T(.1), T(.01), *state);
    require(std::abs(adaptation.get_dt() - T(.01)) < tolerance, "Modified endpoints limit the next step");

    adaptation.reset();
    operations.assign_scalar(32, *error);
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        const auto used = adaptation.get_dt();
        require(adaptation.assess(0, used, *state, *state, next, 5, &*error) == adaptation_status::rejected &&
                    std::abs(next - used * T(.45)) < tolerance,
            "Repeated SciPy rejection uses the error, not automatic halving");
        adaptation.update(adaptation_status::rejected, 0, used, *state);
    }
    adaptation.reset();
    operations.assign_scalar(256, *error);
    require(adaptation.assess(0, T(.1), *state, *state, next, 8, &*error) == adaptation_status::rejected &&
                std::abs(next - T(.045)) < tolerance,
        "Eighth-power assessment uses exponent one eighth");
    adaptation.reset();
    operations.assign_scalar(T(1e6), *error);
    require(adaptation.assess(0, T(-.1), *state, *state, next, 3, &*error) == adaptation_status::rejected &&
                std::abs(next - T(.02)) < tolerance,
        "Shrinkage is bounded by factor .2 and returns a positive magnitude");
    adaptation.reset();
    require(adaptation.reject_step(T(-.1)) == adaptation_status::rejected &&
                std::abs(adaptation.get_dt() - T(.02)) < tolerance,
        "Failed numerical attempts use the independent recovery path");
    adaptation.reset();
    require(adaptation.assess(0, T(.1), *state, *state, next, 5) == adaptation_status::failed &&
                adaptation.assess(0, T(.1), *state, *state, next, 0, &*error) == adaptation_status::failed &&
                adaptation.assess(0, T(.1), *state, *state, next, 5, &*error, &*error3) == adaptation_status::failed,
        "Missing estimates, missing order, and incorrect dual-estimator order fail");
    for (const T invalid : {T(0), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
    {
        require(adaptation.assess(0, invalid, *state, *state, next, 5, &*error) == adaptation_status::failed,
            "Invalid attempted step fails without candidate assessment");
    }

    p.minimum_step = T(.05);
    adaptation_type limited(operations, p);
    require(limited.reject_step(T(.1)) == adaptation_status::failed,
        "An inadmissibly small retry fails rather than clamping and looping");
    operations.assign_scalar(0, *error);
    require(limited.assess(0, T(.01), *state, *state, next, 5, &*error) == adaptation_status::accepted,
        "A clipped terminal step below the minimum can be accepted when accurate");
    p.maximum_step = T(.2);
    adaptation_type bounded(operations, p);
    bounded.assess(0, T(.1), *state, *state, next, 5, &*error);
    require(next == p.maximum_step, "Accepted growth respects the maximum step");
    require(
        bounded.initialize(0, *state, T(.001)) == p.minimum_step && bounded.initialize(0, *state, 1) == p.maximum_step,
        "Initial proposals respect configured bounds");
    p.relative_tolerance = std::numeric_limits<T>::epsilon();
    adaptation_type precise(operations, p);
    require(precise.relative_tolerance() == T(100) * std::numeric_limits<T>::epsilon(),
        "Effective relative tolerance respects scalar precision");

    for (int field = 0; field < 5; ++field)
    {
        for (const T invalid : {T(0), T(-1), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity()})
        {
            typename adaptation_type::params bad;
            T* fields[] = {&bad.relative_tolerance, &bad.absolute_tolerance, &bad.initial_step, &bad.minimum_step,
                &bad.maximum_step};
            *fields[field] = invalid;
            bool refused = false;
            try
            {
                adaptation_type invalid_adaptation(operations, bad);
            }
            catch (const std::invalid_argument&)
            {
                refused = true;
            }
            require(refused, "Invalid SciPy adaptation parameters are rejected");
        }
    }
}

using operations_type = scfd_vector_operations<backend_type, double>;

struct forced_growth
{
    operations_type& operations;
    double time = 0;

    void set_time(double value)
    {
        time = value;
    }

    void apply(const operations_type::vector_type& in, operations_type::vector_type& out)
    {
        operations.assign(in, out);
        operations.add_mul_scalar(time, 1., out);
    }
};

void check_integration()
{
    using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;
    operations_type operations(1);
    forced_growth problem{operations};
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations);
    initial.start_use();
    result.start_use();
    operations.assign_scalar(1., *initial);
    adaptation_type::params p;
    p.initial_step = 1;
    p.relative_tolerance = 1e-10;
    p.absolute_tolerance = 1e-12;
    adaptation_type adaptation(operations, p);
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptation_type>;
    for (const auto* method : {"RK23", "RK45", "DOP853"})
    {
        step_type step(operations, problem, adaptation, {method});
        step.apply(*initial, *result);
        require(step.get_status() == nmfd::time_steppers::single_step_status::converged && step.get_attempts() > 1,
            "SciPy adaptation retries an initially excessive RK step");
        step.finalize(adaptation_status::accepted, step.get_dt(), *result);
        nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(operations, step);
        integrator.set_time_interval(0, .5);
        for (int run = 0; run < 2; ++run)
        {
            integrator.apply(*initial, *result);
            const auto view = operations.view(*result);
            require(integrator.get_status() == nmfd::time_steppers::integration_status::completed &&
                        integrator.get_final_time() == .5 && std::abs(view(0) - (2 * std::exp(.5) - 1.5)) < 2e-8,
                "SciPy-controlled RK analytical convergence and restart");
        }
        operations.assign_scalar(2 * std::exp(.5) - 1.5, *initial);
        integrator.set_time_interval(.5, 0);
        integrator.apply(*initial, *result);
        {
            const auto view = operations.view(*result);
            require(integrator.get_status() == nmfd::time_steppers::integration_status::completed &&
                        integrator.get_final_time() == 0 && std::abs(view(0) - 1) < 2e-8,
                "SciPy-controlled RK backward integration");
        }
        operations.assign_scalar(1., *initial);
    }
}

int main()
try
{
    backend_type::init_device();
    check_assessment<float>();
    check_assessment<double>();
    check_control<float>();
    check_control<double>();
    check_integration();
    std::cout << "SciPy RMS, dual-estimator, rejection, bounds, precision, and RK integration tests: PASS\n";
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
