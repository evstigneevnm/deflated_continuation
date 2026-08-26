#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include <common/scfd_serial_cpu_vector_operations.h>

#include <time_stepper/semidiscrete/problem_traits.h>
#include <time_stepper/semidiscrete/residual_assembly.h>
#include <time_stepper/tests/common/test_context.h>

namespace
{

template<class VectorOperations>
struct regular_mass_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using parameter_type = scalar_type;
    using mass_matrix_type = time_steppers::semidiscrete::regular_mass_matrix;

    void mass_action(
        const scalar_type,
        const vector_type&,
        const vector_type& state_rate,
        const parameter_type&,
        vector_type& output)
    {
        output(0) = scalar_type(2)*state_rate(0);
        output(1) = scalar_type(3)*state_rate(1);
    }

    void residual(
        const scalar_type,
        const vector_type& state,
        const parameter_type&,
        vector_type& output)
    {
        output(0) = scalar_type(2)*state(0);
        output(1) = scalar_type(6)*state(1);
    }
};

template<class VectorOperations>
struct index_one_dae_problem
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using parameter_type = scalar_type;
    using mass_matrix_type = time_steppers::semidiscrete::singular_mass_matrix;

    void mass_action(
        const scalar_type,
        const vector_type&,
        const vector_type& state_rate,
        const parameter_type&,
        vector_type& output)
    {
        output(0) = state_rate(0);
        output(1) = 0;
    }

    void residual(
        const scalar_type,
        const vector_type& state,
        const parameter_type&,
        vector_type& output)
    {
        output(0) = state(0);
        output(1) = state(0)+state(1)-scalar_type(1);
    }
};

} // namespace

int main()
{
    using scalar_type = double;
    using vector_operations_type = scfd_serial_cpu_vector_operations<scalar_type>;
    using vector_type = typename vector_operations_type::vector_type;
    using regular_problem_type = regular_mass_problem<vector_operations_type>;
    using dae_problem_type = index_one_dae_problem<vector_operations_type>;

    static_assert(time_steppers::semidiscrete::is_semidiscrete_problem_v<regular_problem_type>);
    static_assert(!time_steppers::semidiscrete::problem_traits<regular_problem_type>::is_dae);
    static_assert(time_steppers::semidiscrete::is_semidiscrete_problem_v<dae_problem_type>);
    static_assert(time_steppers::semidiscrete::problem_traits<dae_problem_type>::is_dae);

    time_steppers::tests::test_context test;
    vector_operations_type vector_operations(2);
    vector_type state;
    vector_type state_rate;
    vector_type scratch;
    vector_type residual;
    vector_operations.init_vectors(state, state_rate, scratch, residual);
    vector_operations.start_use_vectors(state, state_rate, scratch, residual);

    regular_problem_type regular_problem;
    constexpr scalar_type time = 0.7;
    state(0) = std::exp(-time);
    state(1) = scalar_type(2)*std::exp(-scalar_type(2)*time);
    state_rate(0) = -state(0);
    state_rate(1) = -scalar_type(2)*state(1);
    time_steppers::semidiscrete::assemble_residual(
        vector_operations,
        regular_problem,
        time,
        state,
        state_rate,
        scalar_type(0),
        scratch,
        residual);
    test.check(
        vector_operations.norm_l2(residual) <= 64*std::numeric_limits<scalar_type>::epsilon(),
        "regular mass analytical trajectory satisfies M*u_dot+R=0");

    dae_problem_type dae_problem;
    state(0) = std::exp(-time);
    state(1) = scalar_type(1)-state(0);
    state_rate(0) = -state(0);
    state_rate(1) = state(0);
    time_steppers::semidiscrete::assemble_residual(
        vector_operations,
        dae_problem,
        time,
        state,
        state_rate,
        scalar_type(0),
        scratch,
        residual);
    test.check(
        vector_operations.norm_l2(residual) <= 64*std::numeric_limits<scalar_type>::epsilon(),
        "index-one DAE analytical trajectory satisfies differential and algebraic rows");

    constexpr scalar_type step_size = 0.125;
    constexpr scalar_type current_x = 0.7;
    const scalar_type next_x = current_x/(scalar_type(1)+step_size);
    state(0) = next_x;
    state(1) = scalar_type(1)-next_x;
    state_rate(0) = (next_x-current_x)/step_size;
    state_rate(1) = 0;
    time_steppers::semidiscrete::assemble_residual(
        vector_operations,
        dae_problem,
        step_size,
        state,
        state_rate,
        scalar_type(0),
        scratch,
        residual);
    test.check(
        vector_operations.norm_l2(residual) <= 64*std::numeric_limits<scalar_type>::epsilon(),
        "backward Euler DAE stage satisfies both residual rows");

    state(1) += scalar_type(1e-3);
    time_steppers::semidiscrete::assemble_residual(
        vector_operations,
        dae_problem,
        step_size,
        state,
        state_rate,
        scalar_type(0),
        scratch,
        residual);
    test.check(
        std::abs(residual(1)-scalar_type(1e-3)) <= 32*std::numeric_limits<scalar_type>::epsilon(),
        "algebraic-row violation is retained by residual assembly");

    vector_operations.stop_use_vectors(state, state_rate, scratch, residual);
    vector_operations.free_vectors(state, state_rate, scratch, residual);

    std::cout << "DAE contract checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
