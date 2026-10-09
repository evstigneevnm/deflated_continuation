#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
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
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

using operations_type = scfd_vector_operations<backend_type, double>;
using vector_type = operations_type::vector_type;
using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
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

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0;

    void start_use_vector(vector_type& state) const override
    {
        if (state.is_free())
        {
            ++allocations;
            ++live;
        }
        operations_type::start_use_vector(state);
    }

    void free_vector(vector_type& state) const override
    {
        if (!state.is_free())
        {
            --live;
        }
        operations_type::free_vector(state);
    }
};

struct exponential_growth
{
    operations_type& operations;
    unsigned int calls = 0;

    static double exact(double time)
    {
        return std::exp(time);
    }

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.assign(in, out);
    }
};

struct forced_growth
{
    operations_type& operations;
    unsigned int calls = 0, fail_call = 0;
    double time = 0;

    static double exact(double time)
    {
        return 2 * std::exp(time) - time - 1;
    }

    void set_time(double value)
    {
        time = value;
    }

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.assign(in, out);
        operations.add_mul_scalar(time, 1., out);
        if (calls == fail_call)
        {
            operations.assign_scalar(std::numeric_limits<double>::infinity(), out);
        }
    }
};

struct quadratic_growth
{
    operations_type& operations;
    unsigned int calls = 0;

    static double exact(double time)
    {
        return 1 / (1 - time);
    }

    void apply(const vector_type& in, vector_type& out)
    {
        ++calls;
        operations.mul_pointwise(1., in, 1., in, out);
    }
};

struct sample_row
{
    unsigned int problem;
    double start, dt, theta, value;
};

template<class Problem>
void check_order(operations_type& operations, double start, double coarse_step,
    unsigned int problem_id, std::vector<sample_row>& rows)
{
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations), sample(operations);
    initial.start_use();
    result.start_use();
    sample.start_use();
    operations.assign_scalar(Problem::exact(start), *initial);
    double errors[3]{};
    for (int level = 0; level < 3; ++level)
    {
        counting_operations counted(1);
        Problem problem{counted};
        const auto dt = coarse_step / (1 << level);
        adaptation_type adaptation({std::abs(dt)});
        using step_type =
            nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, Problem, adaptation_type, true>;
        {
            step_type step(counted, problem, adaptation, {"DOP853"});
            require(step.dense_output_order() == 7 && counted.live == 18,
                "DOP853 dense output adds only the initial state and three derivatives");
            const auto allocated = counted.allocations;
            step.set_time(start);
            step.set_target_time(start + coarse_step);
            step.apply(*initial, *result);
            require(step.get_status() == step_status::converged && problem.calls == 16,
                "Native DOP853 evaluates three dense-only stages after its thirteen ordinary derivatives");
            const auto dense = step.get_continuous_integration();
            for (const double theta : {0., .01, .13, .37, .73, .99, 1.})
            {
                const auto time = dense.evaluate(theta, *sample);
                const auto value = read(operations, *sample);
                require(time == start + theta * step.get_dt(), "Dense physical time");
                rows.push_back({problem_id, start, step.get_dt(), theta, value});
                if (theta == 0 || theta == 1)
                {
                    require(value == read(operations, theta == 0 ? *initial : *result), "Exact dense endpoints");
                }
                else
                {
                    errors[level] = std::max(errors[level], std::abs(value - Problem::exact(time)));
                }
            }
            require(problem.calls == 16 && counted.allocations == allocated,
                "Dense evaluation must not allocate or evaluate the RHS");
        }
        require(counted.live == 0, "Dense vector ownership must release every buffer");
        require(errors[level] > 32 * std::numeric_limits<double>::epsilon(),
            "Dense-order measurements must stay above roundoff");
    }
    for (int level = 0; level < 2; ++level)
    {
        const auto order = std::log2(errors[level] / errors[level + 1]);
        std::cout << "DOP853 dense problem " << problem_id << ", h=" << coarse_step
                  << ": errors " << errors[level] << " -> " << errors[level + 1] << ", local order " << order << '\n';
        require(order > 7.2 && order < 10, "Seventh-order dense output must approach local eighth-order accuracy");
    }
}

template<class Dense>
void require_expired(const Dense& dense, vector_type& sample)
{
    bool refused = false;
    try
    {
        dense.evaluate(.5, sample);
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    require(refused, "Dense views must expire when their pending step is finalized or replaced");
}

struct retry_adaptation : adaptation_type
{
    double dt = .5;
    unsigned int rejections = 0, assessments = 0;
    bool reject_first_trial = false;

    double initialize(double, const vector_type&, double proposed = -1)
    {
        reset();
        if (proposed > 0)
        {
            dt = proposed;
        }
        return dt;
    }

    void reset()
    {
        dt = .5;
        rejections = assessments = 0;
    }

    double get_dt() const
    {
        return dt;
    }

    adaptation_status assess(double, double, const vector_type&, const vector_type&,
        double& next_dt, unsigned int = 0, const vector_type* = nullptr)
    {
        ++assessments;
        if (reject_first_trial && assessments == 1)
        {
            dt /= 2;
            next_dt = dt;
            return adaptation_status::rejected;
        }
        next_dt = dt;
        return adaptation_status::accepted;
    }

    adaptation_status reject_step(double attempted)
    {
        dt = std::abs(attempted) / 2;
        ++rejections;
        return adaptation_status::rejected;
    }
};

void check_lifecycle(operations_type& operations)
{
    nmfd::detail::vector_wrap<operations_type> initial(operations), result(operations), sample(operations);
    initial.start_use();
    result.start_use();
    sample.start_use();
    operations.assign_scalar(1., *initial);
    counting_operations counted(1);
    forced_growth problem{counted};
    adaptation_type adaptation({.5});
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, adaptation_type, true>;
    using plain_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, adaptation_type>;
    {
        plain_step step(counted, problem, adaptation, {"DOP853"});
        require(counted.live == 14 && step.dense_output_order() == 0, "Disabled dense output removes all four vectors");
        step.apply(*initial, *result);
        require(problem.calls == 13, "Disabled dense output does not evaluate extra stages");
    }
    require(counted.live == 0, "Disabled dense storage release");
    step_type step(counted, problem, adaptation, {"DOP853"});
    step.apply(*initial, *initial);
    const auto first = step.get_continuous_integration();
    first.evaluate(0, *sample);
    require(read(operations, *sample) == 1, "Aliased apply preserves the dense initial state");
    first.evaluate(1, *sample);
    require(read(operations, *sample) == read(operations, *initial), "Aliased apply preserves the dense endpoint");
    require(problem.time == .5, "Extra interpolation stages must restore the problem endpoint time");
    for (const double theta : {-.1, 1.1, std::numeric_limits<double>::quiet_NaN()})
    {
        operations.assign_scalar(77., *sample);
        bool refused = false;
        try
        {
            first.evaluate(theta, *sample);
        }
        catch (const std::invalid_argument&)
        {
            refused = true;
        }
        require(refused && read(operations, *sample) == 77, "Invalid interpolation must preserve output");
    }
    first.evaluate(.37, *result);
    step.finalize(adaptation_status::accepted_modified, .185, *result);
    require_expired(first, *sample);
    step.set_time(.185);
    step.set_target_time(.2);
    step.apply(*result, *initial);
    const auto second = step.get_continuous_integration();
    second.evaluate(1, *sample);
    require(std::abs(.185 + step.get_dt() - .2) < 1e-16 && read(operations, *sample) == read(operations, *initial),
        "Clipped steps preserve exact dense endpoints after external replacement");
    step.reset();
    require_expired(second, *sample);
    step.set_time(0);
    operations.assign_scalar(1., *initial);
    step.apply(*initial, *result);
    require_expired(first, *sample);
    require_expired(second, *sample);
    step.finalize(adaptation_status::accepted, step.get_dt(), *result);

    for (const unsigned int fail_call : {14u, 15u, 16u})
    {
        for (const bool aliased : {false, true})
        {
            forced_growth bad{operations, 0, fail_call};
            using failing_step =
                nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptation_type, true>;
            failing_step failed(operations, bad, adaptation, {"DOP853"});
            operations.assign_scalar(1., *initial);
            operations.assign_scalar(77., *result);
            failed.apply(*initial, aliased ? *initial : *result);
            require(failed.get_status() == step_status::failed_nonfinite && read(operations, *initial) == 1 &&
                        read(operations, *result) == 77,
                "Failed dense stages must preserve both the input and caller output");
            bool refused = false;
            try
            {
                failed.get_continuous_integration();
            }
            catch (const std::logic_error&)
            {
                refused = true;
            }
            require(refused, "Failed dense preparation must not publish an interpolant");
        }
    }
    forced_growth transient{operations, 0, 14};
    retry_adaptation recovery;
    using recovering_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, retry_adaptation, true>;
    recovering_step recovered(operations, transient, recovery, {"DOP853"});
    recovered.apply(*initial, *result);
    require(recovered.get_status() == step_status::converged && recovered.get_attempts() == 2 &&
                recovery.rejections == 1 && transient.calls == 30 && recovered.get_dt() == .25,
        "Dense-stage recovery must recompute from the original state with a smaller step");
    recovered.get_continuous_integration().evaluate(0, *sample);
    require(read(operations, *sample) == 1, "Retried dense output must retain the original initial state");
    recovered.finalize(adaptation_status::accepted, .25, *result);
    recovered.reset();
    transient.calls = 0;
    recovering_step budgeted(operations, transient, recovery, {"DOP853", 1});
    operations.assign_scalar(77., *result);
    budgeted.apply(*initial, *result);
    require(budgeted.get_status() == step_status::attempt_limit_reached && read(operations, *result) == 77 &&
                recovery.rejections == 1,
        "Dense-stage recovery must respect the numerical attempt budget");
    forced_growth valid{operations};
    retry_adaptation reject_first;
    reject_first.reject_first_trial = true;
    recovering_step rejected(operations, valid, reject_first, {"DOP853"});
    rejected.apply(*initial, *result);
    require(rejected.get_status() == step_status::converged && rejected.get_attempts() == 2 &&
                reject_first.assessments == 2 && valid.calls == 29 && rejected.get_dt() == .25,
        "Numerically rejected candidates must not evaluate the dense-only stages");
}

int main(int argc, char** argv)
try
{
    require(argc <= 2, "Usage: test_explicit_dop853_dense_output [samples.txt]");
    backend_type::init_device();
    operations_type operations(1);
    std::vector<sample_row> rows;
    check_order<exponential_growth>(operations, .4, .8, 0, rows);
    check_order<exponential_growth>(operations, .4, -.8, 0, rows);
    check_order<forced_growth>(operations, .4, .8, 1, rows);
    check_order<forced_growth>(operations, .4, -.8, 1, rows);
    check_order<quadratic_growth>(operations, 0, .4, 2, rows);
    check_order<quadratic_growth>(operations, 0, -.2, 2, rows);
    check_lifecycle(operations);
    if (argc == 2)
    {
        std::ofstream file(argv[1]);
        file << "# problem start dt theta value\n" << std::setprecision(17);
        for (const auto& row : rows)
        {
            file << row.problem << ' ' << row.start << ' ' << row.dt << ' ' << row.theta << ' ' << row.value << '\n';
        }
        require(bool(file), "Failed to save dense samples");
    }
    std::cout << "DOP853 dense order, memory, lifecycle and transactional failure tests: PASS\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
