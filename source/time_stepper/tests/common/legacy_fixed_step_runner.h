#ifndef TIME_STEPPER_TESTS_COMMON_LEGACY_FIXED_STEP_RUNNER_H
#define TIME_STEPPER_TESTS_COMMON_LEGACY_FIXED_STEP_RUNNER_H

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include <time_stepper/detail/positive_preserving_dummy.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_step_adaptation_constant.h>

namespace time_steppers
{
namespace tests
{

template<std::size_t Dimension, class VectorOperations, class Problem, class Log>
std::array<typename VectorOperations::scalar_type, Dimension> integrate_legacy_explicit_fixed(
    VectorOperations& vector_operations,
    Problem& problem,
    Log& log,
    const std::string& method,
    const std::array<typename VectorOperations::scalar_type, Dimension>& initial_state,
    const typename VectorOperations::scalar_type parameter,
    const typename VectorOperations::scalar_type initial_time,
    const typename VectorOperations::scalar_type final_time,
    const std::size_t step_count)
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using positivity_type = detail::positive_preserving_dummy<VectorOperations>;
    using adaptation_type = time_step_adaptation_constant<VectorOperations, Log, positivity_type>;
    using step_type = explicit_time_step<VectorOperations, Problem, Log, adaptation_type>;

    const scalar_type step_size = (final_time-initial_time)/static_cast<scalar_type>(step_count);
    positivity_type positivity;
    adaptation_type adaptation(
        &vector_operations,
        &log,
        {initial_time, final_time},
        step_size,
        &positivity);
    step_type step(&vector_operations, &adaptation, &log, &problem, parameter, method);

    vector_type current;
    vector_type next;
    vector_operations.init_vectors(current, next);
    vector_operations.start_use_vectors(current, next);
    for(std::size_t i = 0; i < Dimension; ++i)
    {
        current(i) = initial_state[i];
    }
    step.init_steps(current);
    for(std::size_t step_index = 0; step_index < step_count; ++step_index)
    {
        step.execute_forced_dt(step_size, current, next);
        vector_operations.assign(next, current);
    }

    std::array<scalar_type, Dimension> result{};
    for(std::size_t i = 0; i < Dimension; ++i)
    {
        result[i] = current(i);
    }
    vector_operations.stop_use_vectors(current, next);
    vector_operations.free_vectors(current, next);
    return result;
}

} // namespace tests
} // namespace time_steppers

#endif
