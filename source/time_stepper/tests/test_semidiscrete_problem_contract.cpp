#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

#include <time_stepper/semidiscrete/autonomous_spatial_problem.h>
#include <time_stepper/semidiscrete/problem_traits.h>
#include <time_stepper/semidiscrete/residual_assembly.h>

namespace
{

struct opaque_vector
{
    std::array<double, 2> values{};
};

struct opaque_vector_operations
{
    using scalar_type = double;
    using vector_type = opaque_vector;

    void assign(const vector_type& source, vector_type& destination)
    {
        destination = source;
    }

    void add_mul(const scalar_type alpha, const vector_type& source, vector_type& destination)
    {
        for(std::size_t i = 0; i < destination.values.size(); ++i)
        {
            destination.values[i] += alpha*source.values[i];
        }
    }
};

bool close(const opaque_vector& actual, const opaque_vector& expected, const double tolerance = 1.0e-13)
{
    for(std::size_t i = 0; i < actual.values.size(); ++i)
    {
        if(std::abs(actual.values[i] - expected.values[i]) > tolerance)
        {
            return false;
        }
    }
    return true;
}

struct test_context
{
    int checks = 0;
    int failures = 0;

    void check(const bool condition, const std::string& message)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }
};

struct split_spatial_operator
{
    void F(const opaque_vector& state, const double parameter, opaque_vector& output)
    {
        linear_residual(state, parameter, output);
        opaque_vector nonlinear;
        nonlinear_residual(state, parameter, nonlinear);
        for(std::size_t i = 0; i < output.values.size(); ++i)
        {
            output.values[i] += nonlinear.values[i];
        }
    }

    void linear_residual(const opaque_vector& state, const double parameter, opaque_vector& output)
    {
        output.values = {parameter*state.values[0], -2*state.values[1]};
    }

    void nonlinear_residual(const opaque_vector& state, const double, opaque_vector& output)
    {
        output.values = {state.values[0]*state.values[1], state.values[0]*state.values[0]};
    }
};

struct convex_concave_spatial_operator
{
    void F(const opaque_vector& state, const double parameter, opaque_vector& output)
    {
        convex_residual(state, parameter, output);
        opaque_vector concave;
        concave_residual(state, parameter, concave);
        for(std::size_t i = 0; i < output.values.size(); ++i)
        {
            output.values[i] += concave.values[i];
        }
    }

    void convex_residual(const opaque_vector& state, const double, opaque_vector& output)
    {
        output.values = {4*state.values[0], 4*state.values[1]};
    }

    void concave_residual(const opaque_vector& state, const double, opaque_vector& output)
    {
        output.values = {-state.values[0], -state.values[1]};
    }
};

struct convex_implicit_concave_explicit
{
    static void implicit_residual(
        convex_concave_spatial_operator& spatial_operator,
        const opaque_vector& state,
        const double parameter,
        opaque_vector& output)
    {
        spatial_operator.convex_residual(state, parameter, output);
    }

    static void explicit_residual(
        convex_concave_spatial_operator& spatial_operator,
        const opaque_vector& state,
        const double parameter,
        opaque_vector& output)
    {
        spatial_operator.concave_residual(state, parameter, output);
    }
};

struct regular_mass_problem
{
    using scalar_type = double;
    using vector_type = opaque_vector;
    using parameter_type = double;
    using mass_matrix_type = time_steppers::semidiscrete::regular_mass_matrix;

    void mass_action(
        const double,
        const vector_type&,
        const vector_type& state_rate,
        const double&,
        vector_type& output)
    {
        output.values = {2*state_rate.values[0], 3*state_rate.values[1]};
    }

    void residual(
        const double time,
        const vector_type& state,
        const double& parameter,
        vector_type& output)
    {
        output.values = {state.values[0] + parameter*time, state.values[1] - time};
    }
};

struct singular_mass_dae_problem
{
    using scalar_type = double;
    using vector_type = opaque_vector;
    using parameter_type = double;
    using mass_matrix_type = time_steppers::semidiscrete::singular_mass_matrix;

    void mass_action(
        const double,
        const vector_type&,
        const vector_type& state_rate,
        const double&,
        vector_type& output)
    {
        output.values = {state_rate.values[0], 0};
    }

    void residual(
        const double,
        const vector_type& state,
        const double&,
        vector_type& output)
    {
        output.values = {state.values[0] + state.values[1], state.values[0] - state.values[1]};
    }
};

struct incomplete_split_problem: regular_mass_problem
{
    void implicit_residual(const double, const opaque_vector&, const double&, opaque_vector&)
    {}
};

} // namespace

int main()
{
    namespace semidiscrete = time_steppers::semidiscrete;
    using split_problem = semidiscrete::autonomous_identity_mass_split_problem<
        opaque_vector_operations,
        split_spatial_operator>;
    using convex_split_problem = semidiscrete::autonomous_identity_mass_split_problem<
        opaque_vector_operations,
        convex_concave_spatial_operator,
        convex_implicit_concave_explicit>;

    static_assert(semidiscrete::is_semidiscrete_problem_v<split_problem>);
    static_assert(semidiscrete::is_split_semidiscrete_problem_v<split_problem>);
    static_assert(semidiscrete::is_semidiscrete_problem_v<regular_mass_problem>);
    static_assert(!semidiscrete::problem_traits<regular_mass_problem>::is_dae);
    static_assert(semidiscrete::is_semidiscrete_problem_v<singular_mass_dae_problem>);
    static_assert(semidiscrete::problem_traits<singular_mass_dae_problem>::is_dae);
    static_assert(!semidiscrete::is_semidiscrete_problem_v<incomplete_split_problem>);

    test_context test;
    opaque_vector_operations vector_operations;
    split_spatial_operator spatial_operator;
    split_problem problem(&vector_operations, &spatial_operator);
    const opaque_vector state{{2, -3}};
    const opaque_vector state_rate{{0.5, -1}};
    constexpr double time = 1.25;
    constexpr double parameter = 4;
    opaque_vector scratch;
    opaque_vector output;

    semidiscrete::assemble_residual_from_split(
        vector_operations,
        problem,
        time,
        state,
        parameter,
        scratch,
        output);
    test.check(close(output, opaque_vector{{2, 10}}), "linear/nonlinear policy assembles the spatial residual");

    semidiscrete::assemble_residual(
        vector_operations,
        problem,
        time,
        state,
        state_rate,
        parameter,
        scratch,
        output);
    test.check(close(output, opaque_vector{{2.5, 9}}), "identity-mass residual is state_rate + F(state)");

    convex_concave_spatial_operator convex_concave_operator;
    convex_split_problem convex_problem(&vector_operations, &convex_concave_operator);
    semidiscrete::assemble_residual_from_split(
        vector_operations,
        convex_problem,
        time,
        state,
        parameter,
        scratch,
        output);
    test.check(close(output, opaque_vector{{6, -9}}), "a policy can map convex/concave terms to implicit/explicit treatment");

    regular_mass_problem regular_problem;
    semidiscrete::assemble_residual(
        vector_operations,
        regular_problem,
        time,
        state,
        state_rate,
        parameter,
        scratch,
        output);
    test.check(close(output, opaque_vector{{8, -7.25}}), "regular non-identity mass action is assembled without inversion");

    singular_mass_dae_problem dae_problem;
    semidiscrete::assemble_residual(
        vector_operations,
        dae_problem,
        time,
        state,
        state_rate,
        parameter,
        scratch,
        output);
    test.check(close(output, opaque_vector{{-0.5, 5}}), "singular mass preserves the algebraic residual row");

    std::cout << "Semidiscrete contract checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
