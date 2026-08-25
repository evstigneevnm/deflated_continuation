#ifndef TIME_STEPPER_SEMIDISCRETE_PROBLEM_TRAITS_H
#define TIME_STEPPER_SEMIDISCRETE_PROBLEM_TRAITS_H

#include <type_traits>
#include <utility>

namespace time_steppers
{
namespace semidiscrete
{

// Problems represent M(t,u,mu)*u_dot + R(t,u,mu) = 0. For singular M,
// algebraic equations are the residual rows associated with the null space of M.
struct identity_mass_matrix
{};

struct regular_mass_matrix
{};

struct singular_mass_matrix
{};

template<class Problem, class = void>
struct has_residual_action: std::false_type
{};

template<class Problem>
struct has_residual_action<Problem, std::void_t<decltype(
    std::declval<Problem&>().residual(
        std::declval<typename Problem::scalar_type>(),
        std::declval<const typename Problem::vector_type&>(),
        std::declval<const typename Problem::parameter_type&>(),
        std::declval<typename Problem::vector_type&>()))>>:
    std::is_same<void, decltype(
        std::declval<Problem&>().residual(
            std::declval<typename Problem::scalar_type>(),
            std::declval<const typename Problem::vector_type&>(),
            std::declval<const typename Problem::parameter_type&>(),
            std::declval<typename Problem::vector_type&>()))>
{};

template<class Problem, class = void>
struct has_mass_action: std::false_type
{};

template<class Problem>
struct has_mass_action<Problem, std::void_t<decltype(
    std::declval<Problem&>().mass_action(
        std::declval<typename Problem::scalar_type>(),
        std::declval<const typename Problem::vector_type&>(),
        std::declval<const typename Problem::vector_type&>(),
        std::declval<const typename Problem::parameter_type&>(),
        std::declval<typename Problem::vector_type&>()))>>:
    std::is_same<void, decltype(
        std::declval<Problem&>().mass_action(
            std::declval<typename Problem::scalar_type>(),
            std::declval<const typename Problem::vector_type&>(),
            std::declval<const typename Problem::vector_type&>(),
            std::declval<const typename Problem::parameter_type&>(),
            std::declval<typename Problem::vector_type&>()))>
{};

template<class Problem, class = void>
struct has_implicit_residual_action: std::false_type
{};

template<class Problem>
struct has_implicit_residual_action<Problem, std::void_t<decltype(
    std::declval<Problem&>().implicit_residual(
        std::declval<typename Problem::scalar_type>(),
        std::declval<const typename Problem::vector_type&>(),
        std::declval<const typename Problem::parameter_type&>(),
        std::declval<typename Problem::vector_type&>()))>>:
    std::is_same<void, decltype(
        std::declval<Problem&>().implicit_residual(
            std::declval<typename Problem::scalar_type>(),
            std::declval<const typename Problem::vector_type&>(),
            std::declval<const typename Problem::parameter_type&>(),
            std::declval<typename Problem::vector_type&>()))>
{};

template<class Problem, class = void>
struct has_explicit_residual_action: std::false_type
{};

template<class Problem>
struct has_explicit_residual_action<Problem, std::void_t<decltype(
    std::declval<Problem&>().explicit_residual(
        std::declval<typename Problem::scalar_type>(),
        std::declval<const typename Problem::vector_type&>(),
        std::declval<const typename Problem::parameter_type&>(),
        std::declval<typename Problem::vector_type&>()))>>:
    std::is_same<void, decltype(
        std::declval<Problem&>().explicit_residual(
            std::declval<typename Problem::scalar_type>(),
            std::declval<const typename Problem::vector_type&>(),
            std::declval<const typename Problem::parameter_type&>(),
            std::declval<typename Problem::vector_type&>()))>
{};

template<class Problem, class = void>
struct problem_traits
{
    static constexpr bool valid = false;
    static constexpr bool has_split_residual = false;
    static constexpr bool is_dae = false;
};

template<class Problem>
struct problem_traits<Problem, std::void_t<
    typename Problem::scalar_type,
    typename Problem::vector_type,
    typename Problem::parameter_type,
    typename Problem::mass_matrix_type>>
{
    using scalar_type = typename Problem::scalar_type;
    using vector_type = typename Problem::vector_type;
    using parameter_type = typename Problem::parameter_type;
    using mass_matrix_type = typename Problem::mass_matrix_type;

    static constexpr bool recognized_mass_matrix =
        std::is_same<mass_matrix_type, identity_mass_matrix>::value ||
        std::is_same<mass_matrix_type, regular_mass_matrix>::value ||
        std::is_same<mass_matrix_type, singular_mass_matrix>::value;
    static constexpr bool has_complete_split =
        has_implicit_residual_action<Problem>::value ==
        has_explicit_residual_action<Problem>::value;
    static constexpr bool has_split_residual =
        has_implicit_residual_action<Problem>::value &&
        has_explicit_residual_action<Problem>::value;
    static constexpr bool is_dae =
        std::is_same<mass_matrix_type, singular_mass_matrix>::value;
    static constexpr bool valid =
        recognized_mass_matrix &&
        has_residual_action<Problem>::value &&
        has_mass_action<Problem>::value &&
        has_complete_split;
};

template<class Problem>
constexpr bool is_semidiscrete_problem_v = problem_traits<Problem>::valid;

template<class Problem>
constexpr bool is_split_semidiscrete_problem_v =
    problem_traits<Problem>::valid && problem_traits<Problem>::has_split_residual;

template<class Problem>
constexpr void validate_problem_contract()
{
    static_assert(
        problem_traits<Problem>::valid,
        "A semidiscrete problem must provide scalar_type, vector_type, parameter_type, "
        "mass_matrix_type, residual(), and mass_action(). A split must provide both "
        "implicit_residual() and explicit_residual().");
}

} // namespace semidiscrete
} // namespace time_steppers

#endif
