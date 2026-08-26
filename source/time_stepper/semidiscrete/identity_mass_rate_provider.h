#ifndef TIME_STEPPER_SEMIDISCRETE_IDENTITY_MASS_RATE_PROVIDER_H
#define TIME_STEPPER_SEMIDISCRETE_IDENTITY_MASS_RATE_PROVIDER_H

#include <type_traits>

#include <time_stepper/runge_kutta/stage_context.h>
#include <time_stepper/semidiscrete/problem_traits.h>

namespace time_steppers
{
namespace semidiscrete
{

template<class Problem, bool Valid = is_semidiscrete_problem_v<Problem>>
struct is_explicit_identity_mass_problem: std::false_type
{};

template<class Problem>
struct is_explicit_identity_mass_problem<Problem, true>:
    std::is_same<typename problem_traits<Problem>::mass_matrix_type, identity_mass_matrix>
{};

template<class Problem>
constexpr bool is_explicit_identity_mass_problem_v =
    is_explicit_identity_mass_problem<Problem>::value;

struct identity_mass_rate_provider
{
    template<class VectorOperations, class Problem>
    bool evaluate(
        VectorOperations& vector_operations,
        Problem& problem,
        const runge_kutta::stage_context<typename Problem::scalar_type>& context,
        const typename Problem::vector_type& state,
        const typename Problem::parameter_type& parameter,
        typename Problem::vector_type& rate) const
    {
        validate_problem_contract<Problem>();
        static_assert(
            is_explicit_identity_mass_problem_v<Problem>,
            "identity_mass_rate_provider requires a semidiscrete problem with identity_mass_matrix. "
            "Regular or singular mass matrices require a dedicated rate provider or an implicit method.");
        static_assert(
            std::is_same<typename VectorOperations::vector_type, typename Problem::vector_type>::value,
            "The problem and vector operations must use the same vector_type.");

        problem.residual(context.stage_time, state, parameter, rate);
        vector_operations.scale(typename Problem::scalar_type(-1), rate);
        return true;
    }
};

} // namespace semidiscrete
} // namespace time_steppers

#endif
