#ifndef NONLINEAR_OPERATORS_ADJOINT_JACOBIAN_CAPABILITY_H
#define NONLINEAR_OPERATORS_ADJOINT_JACOBIAN_CAPABILITY_H

#include <type_traits>
#include <utility>

namespace nonlinear_operators
{

namespace detail
{

template<class Result>
struct is_preconditioner_application_result
    : std::integral_constant<
          bool,
          std::is_same<Result, void>::value ||
              std::is_same<Result, bool>::value>
{
};

} // namespace detail

template <class NonlinearOperator, class = void>
struct has_jacobian_u_adjoint : std::false_type
{
};

template <class NonlinearOperator>
struct has_jacobian_u_adjoint<
    NonlinearOperator,
    std::void_t<decltype(
        std::declval<NonlinearOperator&>().jacobian_u_adjoint(
            std::declval<const typename NonlinearOperator::T_vec&>(),
            std::declval<typename NonlinearOperator::T_vec&>()))>>
    : std::is_same<
          decltype(std::declval<NonlinearOperator&>().jacobian_u_adjoint(
              std::declval<const typename NonlinearOperator::T_vec&>(),
              std::declval<typename NonlinearOperator::T_vec&>())),
          void>
{
};

template <class NonlinearOperator>
inline constexpr bool has_jacobian_u_adjoint_v =
    has_jacobian_u_adjoint<NonlinearOperator>::value;

template <class NonlinearOperator, class = void>
struct has_component_jacobian_u_adjoint : std::false_type
{
};

template <class NonlinearOperator>
struct has_component_jacobian_u_adjoint<
    NonlinearOperator,
    std::void_t<
        decltype(std::declval<NonlinearOperator&>().linear_jacobian_u_adjoint(
            std::declval<const typename NonlinearOperator::T_vec&>(),
            std::declval<typename NonlinearOperator::T_vec&>())),
        decltype(std::declval<NonlinearOperator&>().nonlinear_jacobian_u_adjoint(
            std::declval<const typename NonlinearOperator::T_vec&>(),
            std::declval<typename NonlinearOperator::T_vec&>()))>>
    : std::integral_constant<
          bool,
          std::is_same<
              decltype(std::declval<NonlinearOperator&>().linear_jacobian_u_adjoint(
                  std::declval<const typename NonlinearOperator::T_vec&>(),
                  std::declval<typename NonlinearOperator::T_vec&>())),
              void>::value &&
              std::is_same<
                  decltype(std::declval<NonlinearOperator&>().nonlinear_jacobian_u_adjoint(
                      std::declval<const typename NonlinearOperator::T_vec&>(),
                      std::declval<typename NonlinearOperator::T_vec&>())),
                  void>::value>
{
};

template <class NonlinearOperator>
inline constexpr bool has_component_jacobian_u_adjoint_v =
    has_component_jacobian_u_adjoint<NonlinearOperator>::value;

template <class NonlinearOperator, class = void>
struct has_preconditioner_jacobian_affine_u : std::false_type
{
};

template <class NonlinearOperator>
struct has_preconditioner_jacobian_affine_u<
    NonlinearOperator,
    std::void_t<decltype(
        std::declval<const NonlinearOperator&>().
            preconditioner_jacobian_affine_u(
                std::declval<typename NonlinearOperator::T_vec&>(),
                std::declval<typename NonlinearOperator::T>(),
                std::declval<typename NonlinearOperator::T>()))>>
    : detail::is_preconditioner_application_result<
          decltype(std::declval<const NonlinearOperator&>().
              preconditioner_jacobian_affine_u(
                  std::declval<typename NonlinearOperator::T_vec&>(),
                  std::declval<typename NonlinearOperator::T>(),
                  std::declval<typename NonlinearOperator::T>()))>
{
};

template <class NonlinearOperator>
inline constexpr bool has_preconditioner_jacobian_affine_u_v =
    has_preconditioner_jacobian_affine_u<NonlinearOperator>::value;

template <class NonlinearOperator, class = void>
struct has_preconditioner_jacobian_affine_u_adjoint : std::false_type
{
};

template <class NonlinearOperator>
struct has_preconditioner_jacobian_affine_u_adjoint<
    NonlinearOperator,
    std::void_t<decltype(
        std::declval<const NonlinearOperator&>().
            preconditioner_jacobian_affine_u_adjoint(
                std::declval<typename NonlinearOperator::T_vec&>(),
                std::declval<typename NonlinearOperator::T>(),
                std::declval<typename NonlinearOperator::T>()))>>
    : detail::is_preconditioner_application_result<
          decltype(std::declval<const NonlinearOperator&>().
              preconditioner_jacobian_affine_u_adjoint(
                  std::declval<typename NonlinearOperator::T_vec&>(),
                  std::declval<typename NonlinearOperator::T>(),
                  std::declval<typename NonlinearOperator::T>()))>
{
};

template <class NonlinearOperator>
inline constexpr bool has_preconditioner_jacobian_affine_u_adjoint_v =
    has_preconditioner_jacobian_affine_u_adjoint<NonlinearOperator>::value;

template<class NonlinearOperator>
inline constexpr bool has_affine_preconditioner_adjoint_pair_v =
    has_preconditioner_jacobian_affine_u_v<NonlinearOperator> &&
    has_preconditioner_jacobian_affine_u_adjoint_v<NonlinearOperator>;

} // namespace nonlinear_operators

#endif
