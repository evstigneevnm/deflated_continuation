#ifndef TIME_STEPPER_SEMIDISCRETE_RESIDUAL_ASSEMBLY_H
#define TIME_STEPPER_SEMIDISCRETE_RESIDUAL_ASSEMBLY_H

#include <time_stepper/semidiscrete/problem_traits.h>

namespace time_steppers
{
namespace semidiscrete
{

template<class VectorOperations, class Problem>
void assemble_residual(
    VectorOperations& vector_operations,
    Problem& problem,
    const typename Problem::scalar_type time,
    const typename Problem::vector_type& state,
    const typename Problem::vector_type& state_rate,
    const typename Problem::parameter_type& parameter,
    typename Problem::vector_type& spatial_scratch,
    typename Problem::vector_type& output)
{
    validate_problem_contract<Problem>();
    problem.mass_action(time, state, state_rate, parameter, output);
    problem.residual(time, state, parameter, spatial_scratch);
    vector_operations.add_mul(typename Problem::scalar_type(1), spatial_scratch, output);
}

template<class VectorOperations, class Problem>
void assemble_residual_from_split(
    VectorOperations& vector_operations,
    Problem& problem,
    const typename Problem::scalar_type time,
    const typename Problem::vector_type& state,
    const typename Problem::parameter_type& parameter,
    typename Problem::vector_type& explicit_scratch,
    typename Problem::vector_type& output)
{
    static_assert(
        is_split_semidiscrete_problem_v<Problem>,
        "assemble_residual_from_split requires implicit_residual() and explicit_residual().");
    problem.implicit_residual(time, state, parameter, output);
    problem.explicit_residual(time, state, parameter, explicit_scratch);
    vector_operations.add_mul(typename Problem::scalar_type(1), explicit_scratch, output);
}

} // namespace semidiscrete
} // namespace time_steppers

#endif
