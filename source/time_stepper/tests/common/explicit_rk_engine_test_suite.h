#ifndef TIME_STEPPER_TESTS_COMMON_EXPLICIT_RK_ENGINE_TEST_SUITE_H
#define TIME_STEPPER_TESTS_COMMON_EXPLICIT_RK_ENGINE_TEST_SUITE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common/scfd_vector_operations.h>
#include <scfd/utils/device_tag.h>

#include <time_stepper/detail/butcher_tables.h>
#include <time_stepper/runge_kutta/explicit_rk_step.h>
#include <time_stepper/semidiscrete/problem_traits.h>
#include <time_stepper/tests/common/test_context.h>

namespace time_steppers
{
namespace tests
{
namespace explicit_rk_engine_detail
{

template<class Backend, class Scalar>
class counting_vector_operations: public scfd_vector_operations<Backend, Scalar>
{
public:
    using base_type = scfd_vector_operations<Backend, Scalar>;
    using vector_type = typename base_type::vector_type;

    explicit counting_vector_operations(const std::size_t size): base_type(size)
    {
    }

    void start_use_vector(vector_type& vector) const override
    {
        ++start_use_calls_;
        base_type::start_use_vector(vector);
    }

    std::size_t start_use_calls() const
    {
        return start_use_calls_;
    }

private:
    mutable std::size_t start_use_calls_ = 0;
};

template<class VectorOperations>
class owned_vectors
{
public:
    using vector_type = typename VectorOperations::vector_type;

    owned_vectors(VectorOperations& vector_operations, const std::size_t count):
        vector_operations_(&vector_operations),
        vectors_(count)
    {
        try
        {
            for(auto& vector: vectors_)
            {
                vector_operations_->init_vector(vector);
                vector_operations_->start_use_vector(vector);
                ++started_;
            }
        }
        catch(...)
        {
            release();
            throw;
        }
    }

    owned_vectors(const owned_vectors&) = delete;
    owned_vectors& operator=(const owned_vectors&) = delete;

    ~owned_vectors()
    {
        release();
    }

    vector_type& operator[](const std::size_t index)
    {
        return vectors_.at(index);
    }

    const vector_type& operator[](const std::size_t index) const
    {
        return vectors_.at(index);
    }

private:
    void release() noexcept
    {
        for(std::size_t index = 0; index < started_; ++index)
        {
            try
            {
                vector_operations_->stop_use_vector(vectors_[index]);
                vector_operations_->free_vector(vectors_[index]);
            }
            catch(...)
            {
            }
        }
        started_ = 0;
    }

    VectorOperations* vector_operations_;
    std::vector<vector_type> vectors_;
    std::size_t started_ = 0;
};

struct exponential_equation
{
    template<class Scalar, class Ordinal>
    __DEVICE_TAG__ static Scalar evaluate(
        const Ordinal index,
        const Scalar* state,
        const Scalar,
        const Scalar rate)
    {
        return rate*state[index];
    }
};

struct time_forcing_equation
{
    template<class Scalar, class Ordinal>
    __DEVICE_TAG__ static Scalar evaluate(
        const Ordinal,
        const Scalar*,
        const Scalar time,
        const Scalar)
    {
        return time;
    }
};

struct lorenz_equation
{
    template<class Scalar, class Ordinal>
    __DEVICE_TAG__ static Scalar evaluate(
        const Ordinal index,
        const Scalar* state,
        const Scalar,
        const Scalar)
    {
        if(index == Ordinal(0))
        {
            return Scalar(10)*(state[1]-state[0]);
        }
        if(index == Ordinal(1))
        {
            return state[0]*(Scalar(28)-state[2])-state[1];
        }
        return state[0]*state[1]-(Scalar(8)/Scalar(3))*state[2];
    }
};

struct rossler_equation
{
    template<class Scalar, class Ordinal>
    __DEVICE_TAG__ static Scalar evaluate(
        const Ordinal index,
        const Scalar* state,
        const Scalar,
        const Scalar)
    {
        if(index == Ordinal(0))
        {
            return -state[1]-state[2];
        }
        if(index == Ordinal(1))
        {
            return state[0]+Scalar(0.2)*state[1];
        }
        return Scalar(0.2)+state[2]*(state[0]-Scalar(5.7));
    }
};

struct van_der_pol_equation
{
    template<class Scalar, class Ordinal>
    __DEVICE_TAG__ static Scalar evaluate(
        const Ordinal index,
        const Scalar* state,
        const Scalar,
        const Scalar mu)
    {
        if(index == Ordinal(0))
        {
            return state[1];
        }
        return mu*(Scalar(1)-state[0]*state[0])*state[1]-state[0];
    }
};

template<
    class VectorOperations,
    class Equation,
    class MassMatrix = semidiscrete::identity_mass_matrix>
class device_semidiscrete_problem
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using parameter_type = scalar_type;
    using mass_matrix_type = MassMatrix;
    using ordinal_type = typename VectorOperations::ordinal_type;
    using for_each_type = typename VectorOperations::for_each_type;

    explicit device_semidiscrete_problem(
        VectorOperations& vector_operations,
        const bool record_stage_times = false):
        vector_operations_(&vector_operations),
        record_stage_times_(record_stage_times)
    {
    }

    void mass_action(
        const scalar_type,
        const vector_type&,
        const vector_type& state_rate,
        const parameter_type&,
        vector_type& output)
    {
        vector_operations_->assign(state_rate, output);
    }

    void residual(
        const scalar_type time,
        const vector_type& state,
        const parameter_type parameter,
        vector_type& output)
    {
        if(record_stage_times_)
        {
            observed_stage_times_.push_back(time);
        }
        const auto state_pointer = state.raw_ptr();
        auto output_pointer = output.raw_ptr();
        const ordinal_type size = static_cast<ordinal_type>(state.size());
        for_each_type for_each;
        for_each([=] __DEVICE_TAG__ (const ordinal_type index)
        {
            output_pointer[index] =
                -Equation::template evaluate<scalar_type>(
                    index,
                    state_pointer,
                    time,
                    parameter);
        }, size);
        for_each.wait();
    }

    const std::vector<scalar_type>& observed_stage_times() const
    {
        return observed_stage_times_;
    }

private:
    VectorOperations* vector_operations_;
    bool record_stage_times_;
    std::vector<scalar_type> observed_stage_times_;
};

template<class VectorOperations>
void write_vector(
    VectorOperations& vector_operations,
    typename VectorOperations::vector_type& vector,
    const std::vector<typename VectorOperations::scalar_type>& values)
{
    vector_operations.set(values.data(), vector, values.size());
}

template<class VectorOperations>
std::vector<typename VectorOperations::scalar_type> read_vector(
    VectorOperations& vector_operations,
    const typename VectorOperations::vector_type& vector)
{
    std::vector<typename VectorOperations::scalar_type> values(
        vector_operations.get_size(vector));
    vector_operations.get(vector, values.data(), values.size());
    return values;
}

template<class Scalar>
std::pair<Scalar, Scalar> reference_exponential_step(
    const std::string& method,
    const Scalar initial,
    const Scalar rate,
    const Scalar step_size)
{
    auto tableau = detail::butcher_tables().set_table_by_name(method);
    std::vector<Scalar> stage_rates(tableau.get_size(), Scalar(0));
    for(std::size_t stage = 0; stage < tableau.get_size(); ++stage)
    {
        Scalar stage_state = initial;
        for(std::size_t previous = 0; previous < stage; ++previous)
        {
            stage_state += step_size*
                tableau.template get_A<Scalar>(stage, previous)*stage_rates[previous];
        }
        stage_rates[stage] = rate*stage_state;
    }

    Scalar next = initial;
    Scalar error = 0;
    for(std::size_t stage = 0; stage < tableau.get_size(); ++stage)
    {
        next += step_size*tableau.template get_b<Scalar>(stage)*stage_rates[stage];
        if(tableau.is_embedded())
        {
            error += step_size*
                tableau.template get_err_b<Scalar>(stage)*stage_rates[stage];
        }
    }
    return {next, error};
}

template<class VectorOperations, class Problem>
std::vector<typename VectorOperations::scalar_type> integrate(
    VectorOperations& vector_operations,
    Problem& problem,
    const std::string& method,
    const std::vector<typename VectorOperations::scalar_type>& initial,
    const typename Problem::parameter_type parameter,
    const typename Problem::scalar_type initial_time,
    const typename Problem::scalar_type final_time,
    const std::size_t step_count,
    test_context& test)
{
    using scalar_type = typename VectorOperations::scalar_type;
    owned_vectors<VectorOperations> vectors(vector_operations, 3);
    auto& current = vectors[0];
    auto& next = vectors[1];
    auto& error = vectors[2];
    write_vector(vector_operations, current, initial);

    runge_kutta::explicit_rk_step<VectorOperations> step(vector_operations, method);
    const scalar_type step_size =
        (final_time-initial_time)/static_cast<scalar_type>(step_count);
    scalar_type time = initial_time;
    for(std::size_t index = 0; index < step_count; ++index)
    {
        const auto result = step.advance(
            problem,
            time,
            step_size,
            current,
            parameter,
            next,
            error);
        test.check(static_cast<bool>(result), method + " fixed-step advance succeeds");
        if(!result)
        {
            return read_vector(vector_operations, current);
        }
        vector_operations.assign(next, current);
        time += step_size;
    }
    return read_vector(vector_operations, current);
}

inline long double vector_distance(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    long double sum = 0;
    for(std::size_t index = 0; index < left.size(); ++index)
    {
        const long double difference =
            static_cast<long double>(left[index])-static_cast<long double>(right[index]);
        sum += difference*difference;
    }
    return std::sqrt(sum);
}

inline long double observed_order(
    const long double coarse_error,
    const long double fine_error)
{
    return std::log(coarse_error/fine_error)/std::log(2.0L);
}

inline long double observed_self_order(
    const std::vector<double>& coarse,
    const std::vector<double>& fine,
    const std::vector<double>& finer)
{
    return observed_order(vector_distance(coarse, fine), vector_distance(fine, finer));
}

template<class VectorOperations>
class fail_at_stage_provider
{
public:
    explicit fail_at_stage_provider(const std::size_t failed_stage):
        failed_stage_(failed_stage)
    {
    }

    template<class Operations, class Problem>
    bool evaluate(
        Operations& vector_operations,
        Problem& problem,
        const runge_kutta::stage_context<typename Problem::scalar_type>& context,
        const typename Problem::vector_type& state,
        const typename Problem::parameter_type& parameter,
        typename Problem::vector_type& rate)
    {
        if(context.stage_index == failed_stage_)
        {
            return false;
        }
        return identity_provider_.evaluate(
            vector_operations,
            problem,
            context,
            state,
            parameter,
            rate);
    }

private:
    std::size_t failed_stage_;
    semidiscrete::identity_mass_rate_provider identity_provider_;
};

template<class Backend>
int run_explicit_rk_engine_tests(const std::string& backend_name)
{
    using scalar_type = double;
    using vector_operations_type = counting_vector_operations<Backend, scalar_type>;
    using exponential_problem_type =
        device_semidiscrete_problem<vector_operations_type, exponential_equation>;
    using singular_problem_type = device_semidiscrete_problem<
        vector_operations_type,
        exponential_equation,
        semidiscrete::singular_mass_matrix>;

    static_assert(
        semidiscrete::is_explicit_identity_mass_problem_v<exponential_problem_type>,
        "The analytical ODE fixture must satisfy the explicit identity-mass contract.");
    static_assert(
        !semidiscrete::is_explicit_identity_mass_problem_v<singular_problem_type>,
        "A singular-mass DAE must not enter the identity-mass explicit RK path.");

    test_context test;
    const scalar_type tolerance = scalar_type(5.0e-12);
    const std::vector<std::pair<std::string, unsigned int>> methods = {
        {"EE", 1},
        {"HE", 2},
        {"RK33SSP", 3},
        {"RK43SSP", 3},
        {"RKDP45", 5},
        {"RK64SSP", 4}};

    {
        vector_operations_type vector_operations(1);
        exponential_problem_type problem(vector_operations);
        owned_vectors<vector_operations_type> vectors(vector_operations, 3);
        write_vector(vector_operations, vectors[0], {scalar_type(1.25)});

        for(const auto& method: methods)
        {
            write_vector(vector_operations, vectors[0], {scalar_type(1.25)});
            const auto before_constructor = vector_operations.start_use_calls();
            runge_kutta::explicit_rk_step<vector_operations_type> step(
                vector_operations,
                method.first);
            const auto after_constructor = vector_operations.start_use_calls();
            const scalar_type step_size = scalar_type(0.125);
            const scalar_type rate = scalar_type(-0.7);
            const auto result = step.advance(
                problem,
                scalar_type(0.3),
                step_size,
                vectors[0],
                rate,
                vectors[1],
                vectors[2]);
            const auto expected = reference_exponential_step(
                method.first,
                scalar_type(1.25),
                rate,
                step_size);
            const auto next = read_vector(vector_operations, vectors[1]);
            const auto error = read_vector(vector_operations, vectors[2]);
            test.check(static_cast<bool>(result), method.first + " one-step result succeeds");
            test.check(
                result.rate_evaluations == step.stage_count(),
                method.first + " evaluates each stage exactly once");
            test.check(
                std::abs(next[0]-expected.first) <= tolerance,
                method.first + " one-step endpoint agrees with tableau reference");
            test.check(
                std::abs(error[0]-expected.second) <= tolerance,
                method.first + " embedded error agrees with tableau reference");
            test.check(
                after_constructor-before_constructor == step.workspace_allocation_count() &&
                step.workspace_allocation_count() == step.stage_count()+1,
                method.first + " allocates one stage state and one rate per stage");

            const auto before_second_step = vector_operations.start_use_calls();
            const auto second_result = step.advance(
                problem,
                scalar_type(0.425),
                step_size,
                vectors[1],
                rate,
                vectors[0],
                vectors[2]);
            test.check(static_cast<bool>(second_result), method.first + " repeated advance succeeds");
            test.check(
                vector_operations.start_use_calls() == before_second_step,
                method.first + " repeated advance performs no vector allocation");
        }

        bool rejected_implicit = false;
        try
        {
            runge_kutta::explicit_rk_step<vector_operations_type> invalid_step(
                vector_operations,
                "IE");
        }
        catch(const std::invalid_argument&)
        {
            rejected_implicit = true;
        }
        test.check(rejected_implicit, "explicit engine rejects an implicit tableau");
    }

    for(const auto& method: methods)
    {
        vector_operations_type vector_operations(1);
        exponential_problem_type problem(vector_operations);
        const auto coarse = integrate(
            vector_operations, problem, method.first, {1}, scalar_type(-0.7), 0, 1, 10, test);
        const auto fine = integrate(
            vector_operations, problem, method.first, {1}, scalar_type(-0.7), 0, 1, 20, test);
        const scalar_type exact = std::exp(scalar_type(-0.7));
        const long double order = observed_order(
            std::abs(static_cast<long double>(coarse[0])-exact),
            std::abs(static_cast<long double>(fine[0])-exact));
        test.check(
            std::isfinite(static_cast<double>(order)) &&
                order >= static_cast<long double>(method.second)-0.2L,
            method.first + " global order on " + backend_name + " is " +
                std::to_string(static_cast<double>(order)));
    }

    {
        using problem_type =
            device_semidiscrete_problem<vector_operations_type, time_forcing_equation>;
        vector_operations_type vector_operations(1);
        problem_type problem(vector_operations, true);
        owned_vectors<vector_operations_type> vectors(vector_operations, 3);
        write_vector(vector_operations, vectors[0], {scalar_type(0)});
        runge_kutta::explicit_rk_step<vector_operations_type> step(
            vector_operations,
            "RKDP45");
        const auto result = step.advance(
            problem, 0, 1, vectors[0], 0, vectors[1], vectors[2]);
        const auto next = read_vector(vector_operations, vectors[1]);
        auto tableau = detail::butcher_tables().set_table_by_name("RKDP45");
        const auto& observed_times = problem.observed_stage_times();
        test.check(static_cast<bool>(result), "nonautonomous RKDP45 step succeeds");
        test.check(
            observed_times.size() == tableau.get_size(),
            "nonautonomous problem observes every stage time");
        for(std::size_t stage = 0; stage < observed_times.size(); ++stage)
        {
            test.check(
                std::abs(observed_times[stage]-tableau.get_c<scalar_type>(stage)) <= tolerance,
                "RKDP45 stage time " + std::to_string(stage) + " uses c_i");
        }
        test.check(
            std::abs(next[0]-scalar_type(0.5)) <= tolerance,
            "nonautonomous y'=t step integrates the stage times correctly");
    }

    {
        vector_operations_type vector_operations(1);
        exponential_problem_type problem(vector_operations);
        owned_vectors<vector_operations_type> vectors(vector_operations, 3);
        write_vector(vector_operations, vectors[0], {scalar_type(1)});
        runge_kutta::explicit_rk_step<vector_operations_type> step(vector_operations, "EE");
        const auto result = step.advance(
            problem, 0, scalar_type(0.1), vectors[0], scalar_type(-2), vectors[1], vectors[2]);
        const auto next = read_vector(vector_operations, vectors[1]);
        test.check(static_cast<bool>(result), "sign-convention Euler step succeeds");
        test.check(
            std::abs(next[0]-scalar_type(0.8)) <= tolerance,
            "M*u_dot+R=0 is converted to u_dot=-R exactly once");
    }

    {
        vector_operations_type vector_operations(1);
        exponential_problem_type problem(vector_operations);
        owned_vectors<vector_operations_type> vectors(vector_operations, 5);
        write_vector(vector_operations, vectors[0], {scalar_type(1)});
        runge_kutta::explicit_rk_step<vector_operations_type> step(vector_operations, "HE");
        const scalar_type coarse_step = scalar_type(0.2);
        const scalar_type fine_step = scalar_type(0.1);
        const auto coarse_result = step.advance(
            problem, 0, coarse_step, vectors[0], scalar_type(1), vectors[1], vectors[2]);
        const auto fine_result = step.advance(
            problem, 0, fine_step, vectors[0], scalar_type(1), vectors[3], vectors[4]);
        const scalar_type coarse_error = read_vector(vector_operations, vectors[2])[0];
        const scalar_type fine_error = read_vector(vector_operations, vectors[4])[0];
        test.check(
            coarse_result.error_estimate_available && fine_result.error_estimate_available,
            "Heun-Euler reports an embedded estimate");
        test.check(
            std::abs(coarse_error-scalar_type(0.5)*coarse_step*coarse_step) <= tolerance,
            "embedded estimate includes the step-size factor");
        test.check(
            std::abs(coarse_error/fine_error-scalar_type(4)) <= scalar_type(1.0e-10),
            "Heun-Euler embedded estimate scales quadratically");
    }

    {
        vector_operations_type vector_operations(1);
        exponential_problem_type problem(vector_operations);
        owned_vectors<vector_operations_type> vectors(vector_operations, 3);
        write_vector(vector_operations, vectors[0], {scalar_type(1)});
        write_vector(vector_operations, vectors[1], {scalar_type(17)});
        write_vector(vector_operations, vectors[2], {scalar_type(19)});
        using provider_type = fail_at_stage_provider<vector_operations_type>;
        runge_kutta::explicit_rk_step<vector_operations_type, provider_type> step(
            vector_operations,
            "HE",
            provider_type(1));
        const auto result = step.advance(
            problem, 0, scalar_type(0.1), vectors[0], scalar_type(1), vectors[1], vectors[2]);
        test.check(
            !result && result.status == runge_kutta::step_status::rate_evaluation_failure &&
                result.failed_stage == 1 && result.rate_evaluations == 1,
            "rate-provider failure identifies the failed stage");
        test.check(
            read_vector(vector_operations, vectors[1])[0] == scalar_type(17) &&
                read_vector(vector_operations, vectors[2])[0] == scalar_type(19),
            "failed stage leaves endpoint and error outputs unchanged");
    }

    {
        vector_operations_type vector_operations(3);
        device_semidiscrete_problem<vector_operations_type, lorenz_equation> problem(vector_operations);
        const auto coarse = integrate(
            vector_operations, problem, "RKDP45", {1, 1, 1}, 0, 0, scalar_type(0.25), 10, test);
        const auto fine = integrate(
            vector_operations, problem, "RKDP45", {1, 1, 1}, 0, 0, scalar_type(0.25), 20, test);
        const auto finer = integrate(
            vector_operations, problem, "RKDP45", {1, 1, 1}, 0, 0, scalar_type(0.25), 40, test);
        const long double order = observed_self_order(coarse, fine, finer);
        test.check(
            std::isfinite(static_cast<double>(order)) && order >= 4.2L,
            "Lorenz RKDP45 self-convergence order=" +
                std::to_string(static_cast<double>(order)));
    }

    {
        vector_operations_type vector_operations(3);
        device_semidiscrete_problem<vector_operations_type, rossler_equation> problem(vector_operations);
        const auto coarse = integrate(
            vector_operations, problem, "RKDP45", {1, 0, 0}, 0, 0, 1, 8, test);
        const auto fine = integrate(
            vector_operations, problem, "RKDP45", {1, 0, 0}, 0, 0, 1, 16, test);
        const auto finer = integrate(
            vector_operations, problem, "RKDP45", {1, 0, 0}, 0, 0, 1, 32, test);
        const long double order = observed_self_order(coarse, fine, finer);
        test.check(
            std::isfinite(static_cast<double>(order)) && order >= 4.2L,
            "Rossler RKDP45 self-convergence order=" +
                std::to_string(static_cast<double>(order)));
    }

    {
        vector_operations_type vector_operations(2);
        device_semidiscrete_problem<vector_operations_type, van_der_pol_equation> problem(vector_operations);
        const scalar_type mu = 5;
        const auto coarse = integrate(
            vector_operations, problem, "RKDP45", {2, 0}, mu, 0, scalar_type(0.5), 40, test);
        const auto fine = integrate(
            vector_operations, problem, "RKDP45", {2, 0}, mu, 0, scalar_type(0.5), 80, test);
        const auto finer = integrate(
            vector_operations, problem, "RKDP45", {2, 0}, mu, 0, scalar_type(0.5), 160, test);
        const long double order = observed_self_order(coarse, fine, finer);
        test.check(
            std::isfinite(static_cast<double>(order)) && order >= 4.2L,
            "Van der Pol RKDP45 self-convergence order=" +
                std::to_string(static_cast<double>(order)));
    }

    std::cout << "Explicit RK engine " << backend_name << " checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}

} // namespace explicit_rk_engine_detail

using explicit_rk_engine_detail::run_explicit_rk_engine_tests;

} // namespace tests
} // namespace time_steppers

#endif
