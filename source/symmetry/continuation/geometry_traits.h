#ifndef __SYMMETRY_CONTINUATION_GEOMETRY_TRAITS_H__
#define __SYMMETRY_CONTINUATION_GEOMETRY_TRAITS_H__

#include <type_traits>
#include <utility>

#include <symmetry/continuation/isotropy_transition.h>

namespace symmetry
{
namespace continuation
{
namespace geometry
{

template<class NonlinearOperator, class Vector, class Scalar, class = void>
struct has_isotropy_transition: std::false_type
{
};

template<class NonlinearOperator, class Vector, class Scalar>
struct has_isotropy_transition<
    NonlinearOperator,
    Vector,
    Scalar,
    std::void_t<decltype(
        std::declval<NonlinearOperator*>()->detect_continuation_isotropy_transition(
            std::declval<const Vector&>(),
            std::declval<const Vector&>(),
            std::declval<const isotropy_transition_policy<Scalar>&>()))>>:
    std::true_type
{
};

template<class NonlinearOperator, class Vector, class Scalar, class = void>
struct has_continuation_chart: std::false_type
{
};

template<class NonlinearOperator, class Vector, class Scalar>
struct has_continuation_chart<
    NonlinearOperator,
    Vector,
    Scalar,
    std::void_t<
        decltype(std::declval<NonlinearOperator*>()->begin_continuation_chart(
            std::declval<const Vector&>(),
            std::declval<const Scalar&>(),
            std::declval<const Vector&>(),
            std::declval<const Scalar&>())),
        decltype(std::declval<NonlinearOperator*>()->stabilize_predictor_for_continuation(
            std::declval<const Vector&>(),
            std::declval<const Scalar&>(),
            std::declval<const Vector&>(),
            std::declval<const Scalar&>(),
            std::declval<const Vector&>(),
            std::declval<const Scalar&>(),
            std::declval<Vector&>(),
            std::declval<Scalar&>()))>>: std::true_type
{
};

template<class NonlinearOperator, class Vector, class Scalar, class = void>
struct has_projected_tangent_system: std::false_type
{
};

template<class NonlinearOperator, class Vector, class Scalar>
struct has_projected_tangent_system<
    NonlinearOperator,
    Vector,
    Scalar,
    std::void_t<
        decltype(std::declval<NonlinearOperator*>()->set_projected_linearization_point(
            std::declval<const Vector&>(),
            std::declval<const Scalar&>())),
        decltype(std::declval<NonlinearOperator*>()->projected_jacobian_alpha(
            std::declval<Vector&>())),
        decltype(std::declval<NonlinearOperator*>()->project_current_tangent(
            std::declval<const Vector&>(),
            std::declval<Vector&>()))>>: std::true_type
{
};

} // namespace geometry
} // namespace continuation
} // namespace symmetry

#endif // __SYMMETRY_CONTINUATION_GEOMETRY_TRAITS_H__
