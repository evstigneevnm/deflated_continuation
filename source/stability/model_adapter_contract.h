#ifndef __STABILITY_MODEL_ADAPTER_CONTRACT_H__
#define __STABILITY_MODEL_ADAPTER_CONTRACT_H__

#include <type_traits>
#include <utility>

#include <stability/eigensolvers/eigensolver_result.h>

namespace stability
{
namespace model_adapter
{

namespace detail
{

template<class Result, class = void>
struct is_status_result : std::is_void<Result>
{
};

template<class Result>
struct is_status_result<
    Result,
    std::void_t<decltype(static_cast<bool>(std::declval<Result>()))>>
    : std::true_type
{
};

template<class Provider, class Vector, class Scalar, class = void>
struct is_linearization_provider_impl : std::false_type
{
};

template<class Provider, class Vector, class Scalar>
struct is_linearization_provider_impl<
    Provider,
    Vector,
    Scalar,
    std::void_t<decltype(
        std::declval<Provider&>().set_linearization_point(
            std::declval<const Vector&>(),
            std::declval<Scalar>()))>> : std::true_type
{
};

template<class Operator, class Vector, class = void>
struct is_real_operator_impl : std::false_type
{
};

template<class Operator, class Vector>
struct is_real_operator_impl<
    Operator,
    Vector,
    std::void_t<decltype(
        std::declval<const Operator&>().apply(
            std::declval<const Vector&>(),
            std::declval<Vector&>()))>>
    : is_status_result<decltype(
          std::declval<const Operator&>().apply(
              std::declval<const Vector&>(),
              std::declval<Vector&>()))>
{
};

template<class Provider, class Vector, class Scalar, class = void>
struct is_real_affine_inverse_provider_impl : std::false_type
{
};

template<class Provider, class Vector, class Scalar>
struct is_real_affine_inverse_provider_impl<
    Provider,
    Vector,
    Scalar,
    std::void_t<
        typename Provider::scalar_type,
        typename Provider::vector_type,
        typename Provider::health_type,
        decltype(
            std::declval<const Provider&>().apply(
                std::declval<Scalar>(),
                std::declval<Scalar>(),
                std::declval<const Vector&>(),
                std::declval<Vector&>())),
        decltype(
            std::declval<const Provider&>().health(
                std::declval<Scalar>(),
                std::declval<Scalar>()))>>
    : std::integral_constant<
          bool,
          std::is_same<
              typename Provider::scalar_type,
              Scalar>::value &&
          std::is_same<
              typename Provider::vector_type,
              Vector>::value &&
          std::is_same<
              std::decay_t<decltype(
                  std::declval<const Provider&>().health(
                      std::declval<Scalar>(),
                      std::declval<Scalar>()))>,
              typename Provider::health_type>::value &&
          is_status_result<decltype(
              std::declval<const Provider&>().apply(
                  std::declval<Scalar>(),
                  std::declval<Scalar>(),
                  std::declval<const Vector&>(),
                  std::declval<Vector&>()))>::value>
{
};

template<class Eigensolver, class Vector, class Real, class = void>
struct is_eigensolver_adapter_impl : std::false_type
{
};

template<class Eigensolver, class Vector, class Real>
struct is_eigensolver_adapter_impl<
    Eigensolver,
    Vector,
    Real,
    std::void_t<decltype(
        std::declval<Eigensolver&>().execute(
            std::declval<const Vector&>()))>>
    : std::is_same<
          std::decay_t<decltype(
              std::declval<Eigensolver&>().execute(
                  std::declval<const Vector&>()))>,
          eigensolvers::eigensolver_result<Real>>
{
};

} // namespace detail

template<class Provider, class Vector, class Scalar>
constexpr bool is_linearization_provider_v =
    detail::is_linearization_provider_impl<
        Provider,
        Vector,
        Scalar>::value;

template<class Operator, class Vector>
constexpr bool is_real_operator_v =
    detail::is_real_operator_impl<Operator, Vector>::value;

template<class Provider, class Vector, class Scalar>
constexpr bool is_real_affine_inverse_provider_v =
    detail::is_real_affine_inverse_provider_impl<
        Provider,
        Vector,
        Scalar>::value;

template<class Eigensolver, class Vector, class Real>
constexpr bool is_eigensolver_adapter_v =
    detail::is_eigensolver_adapter_impl<
        Eigensolver,
        Vector,
        Real>::value;

template<
    class VectorOperations,
    class LinearizationProvider,
    class EigensolverAdapter>
struct evaluator_contract
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    static constexpr bool linearization_provider =
        is_linearization_provider_v<
            LinearizationProvider,
            vector_type,
            scalar_type>;
    static constexpr bool eigensolver_adapter =
        is_eigensolver_adapter_v<
            EigensolverAdapter,
            vector_type,
            scalar_type>;
    static constexpr bool value =
        linearization_provider && eigensolver_adapter;
};

template<
    class RealVectorSpace,
    class RealOperator,
    class RealAffineInverseProvider>
struct matrix_free_contract
{
    using scalar_type = typename RealVectorSpace::scalar_type;
    using vector_type = typename RealVectorSpace::vector_type;

    static constexpr bool real_operator =
        is_real_operator_v<RealOperator, vector_type>;
    static constexpr bool affine_inverse_provider =
        is_real_affine_inverse_provider_v<
            RealAffineInverseProvider,
            vector_type,
            scalar_type>;
    static constexpr bool value =
        real_operator && affine_inverse_provider;
};

} // namespace model_adapter
} // namespace stability

#endif
