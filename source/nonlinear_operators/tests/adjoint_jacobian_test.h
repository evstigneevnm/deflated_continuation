#ifndef NONLINEAR_OPERATORS_TESTS_ADJOINT_JACOBIAN_TEST_H
#define NONLINEAR_OPERATORS_TESTS_ADJOINT_JACOBIAN_TEST_H

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <utility>

#include <nonlinear_operators/adjoint_jacobian_capability.h>

namespace nonlinear_operators
{
namespace tests
{

namespace detail
{

template<class Action>
bool apply_preconditioner(Action&& action)
{
    using result_type = decltype(action());
    if constexpr(std::is_void<result_type>::value)
    {
        std::forward<Action>(action)();
        return true;
    }
    else
    {
        return static_cast<bool>(std::forward<Action>(action)());
    }
}

} // namespace detail

template <class VectorOperations, class ForwardAction, class AdjointAction>
typename VectorOperations::scalar_type adjoint_duality_relative_error(
    VectorOperations& vector_operations,
    const typename VectorOperations::vector_type& direction,
    const typename VectorOperations::vector_type& cotangent,
    ForwardAction&& forward_action,
    AdjointAction&& adjoint_action,
    typename VectorOperations::vector_type& forward_value,
    typename VectorOperations::vector_type& adjoint_value)
{
    using scalar_type = typename VectorOperations::scalar_type;
    forward_action(direction, forward_value);
    adjoint_action(cotangent, adjoint_value);

    const scalar_type left =
        vector_operations.scalar_prod(forward_value, cotangent);
    const scalar_type right =
        vector_operations.scalar_prod(direction, adjoint_value);
    const scalar_type scale = std::max(
        scalar_type(1),
        vector_operations.norm_l2(forward_value)*
                vector_operations.norm_l2(cotangent) +
            vector_operations.norm_l2(direction)*
                vector_operations.norm_l2(adjoint_value));
    using std::abs;
    return abs(left - right)/scale;
}

template <class VectorOperations, class NonlinearOperator, class Reporter>
void check_adjoint_jacobian(
    VectorOperations& vector_operations,
    NonlinearOperator& nonlinear_operator,
    const typename VectorOperations::vector_type& state,
    const typename VectorOperations::vector_type& direction,
    const typename VectorOperations::vector_type& cotangent,
    const typename VectorOperations::scalar_type parameter,
    const typename VectorOperations::scalar_type tolerance,
    Reporter&& report,
    const std::string& label)
{
    static_assert(
        has_jacobian_u_adjoint_v<NonlinearOperator>,
        "check_adjoint_jacobian requires jacobian_u_adjoint");

    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    vector_type forward_value;
    vector_type adjoint_value;
    vector_operations.init_vectors(forward_value, adjoint_value);
    vector_operations.start_use_vectors(forward_value, adjoint_value);

    nonlinear_operator.set_linearization_point(state, parameter);
    const scalar_type full_error = adjoint_duality_relative_error(
        vector_operations,
        direction,
        cotangent,
        [&](const vector_type& input, vector_type& output)
        {
            nonlinear_operator.jacobian_u(input, output);
        },
        [&](const vector_type& input, vector_type& output)
        {
            nonlinear_operator.jacobian_u_adjoint(input, output);
        },
        forward_value,
        adjoint_value);
    report(
        full_error <= tolerance,
        label + " full Jacobian adjoint duality, relative error=" +
            std::to_string(static_cast<double>(full_error)));

    if constexpr(has_component_jacobian_u_adjoint_v<NonlinearOperator>)
    {
        const scalar_type linear_error = adjoint_duality_relative_error(
            vector_operations,
            direction,
            cotangent,
            [&](const vector_type& input, vector_type& output)
            {
                nonlinear_operator.linear_jacobian_u(input, output);
            },
            [&](const vector_type& input, vector_type& output)
            {
                nonlinear_operator.linear_jacobian_u_adjoint(input, output);
            },
            forward_value,
            adjoint_value);
        report(
            linear_error <= tolerance,
            label + " linear Jacobian adjoint duality, relative error=" +
                std::to_string(static_cast<double>(linear_error)));

        const scalar_type nonlinear_error = adjoint_duality_relative_error(
            vector_operations,
            direction,
            cotangent,
            [&](const vector_type& input, vector_type& output)
            {
                nonlinear_operator.nonlinear_jacobian_u(input, output);
            },
            [&](const vector_type& input, vector_type& output)
            {
                nonlinear_operator.nonlinear_jacobian_u_adjoint(input, output);
            },
            forward_value,
            adjoint_value);
        report(
            nonlinear_error <= tolerance,
            label + " nonlinear Jacobian adjoint duality, relative error=" +
                std::to_string(static_cast<double>(nonlinear_error)));
    }

    vector_operations.stop_use_vectors(forward_value, adjoint_value);
    vector_operations.free_vectors(forward_value, adjoint_value);
}

template <class VectorOperations, class NonlinearOperator, class Reporter>
void check_affine_preconditioner_adjoint(
    VectorOperations& vector_operations,
    NonlinearOperator& nonlinear_operator,
    const typename VectorOperations::vector_type& state,
    const typename VectorOperations::vector_type& direction,
    const typename VectorOperations::vector_type& cotangent,
    const typename VectorOperations::scalar_type parameter,
    const typename VectorOperations::scalar_type jacobian_scale,
    const typename VectorOperations::scalar_type identity_shift,
    const typename VectorOperations::scalar_type tolerance,
    Reporter&& report,
    const std::string& label)
{
    static_assert(
        has_affine_preconditioner_adjoint_pair_v<NonlinearOperator>,
        "check_affine_preconditioner_adjoint requires a forward/adjoint pair");

    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    vector_type forward_value;
    vector_type adjoint_value;
    vector_operations.init_vectors(forward_value, adjoint_value);
    vector_operations.start_use_vectors(forward_value, adjoint_value);

    nonlinear_operator.set_linearization_point(state, parameter);
    vector_operations.assign(direction, forward_value);
    vector_operations.assign(cotangent, adjoint_value);
    const bool forward_succeeded = detail::apply_preconditioner([&]() {
        return nonlinear_operator.preconditioner_jacobian_affine_u(
            forward_value,
            jacobian_scale,
            identity_shift);
    });
    const bool adjoint_succeeded = detail::apply_preconditioner([&]() {
        return nonlinear_operator.preconditioner_jacobian_affine_u_adjoint(
            adjoint_value,
            jacobian_scale,
            identity_shift);
    });
    report(
        forward_succeeded,
        label + " forward affine preconditioner application");
    report(
        adjoint_succeeded,
        label + " adjoint affine preconditioner application");

    if(forward_succeeded && adjoint_succeeded)
    {
        const scalar_type left =
            vector_operations.scalar_prod(forward_value, cotangent);
        const scalar_type right =
            vector_operations.scalar_prod(direction, adjoint_value);
        const scalar_type scale = std::max(
            scalar_type(1),
            vector_operations.norm_l2(forward_value)*
                    vector_operations.norm_l2(cotangent) +
                vector_operations.norm_l2(direction)*
                    vector_operations.norm_l2(adjoint_value));
        using std::abs;
        const scalar_type error = abs(left - right)/scale;
        report(
            error <= tolerance,
            label + " affine preconditioner adjoint duality, relative error=" +
                std::to_string(static_cast<double>(error)));
    }

    vector_operations.stop_use_vectors(forward_value, adjoint_value);
    vector_operations.free_vectors(forward_value, adjoint_value);
}

template <class VectorOperations, class NonlinearOperator, class Reporter>
void check_affine_adjoint_inverse_residual(
    VectorOperations& vector_operations,
    NonlinearOperator& nonlinear_operator,
    const typename VectorOperations::vector_type& state,
    const typename VectorOperations::vector_type& right_hand_side,
    const typename VectorOperations::scalar_type parameter,
    const typename VectorOperations::scalar_type jacobian_scale,
    const typename VectorOperations::scalar_type identity_shift,
    const typename VectorOperations::scalar_type tolerance,
    Reporter&& report,
    const std::string& label)
{
    static_assert(
        has_jacobian_u_adjoint_v<NonlinearOperator>,
        "check_affine_adjoint_inverse_residual requires jacobian_u_adjoint");
    static_assert(
        has_preconditioner_jacobian_affine_u_adjoint_v<NonlinearOperator>,
        "check_affine_adjoint_inverse_residual requires an adjoint inverse");

    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    vector_type solution;
    vector_type image;
    vector_type residual;
    vector_operations.init_vectors(solution, image, residual);
    vector_operations.start_use_vectors(solution, image, residual);

    nonlinear_operator.set_linearization_point(state, parameter);
    vector_operations.assign(right_hand_side, solution);
    const bool succeeded = detail::apply_preconditioner([&]() {
        return nonlinear_operator.preconditioner_jacobian_affine_u_adjoint(
            solution,
            jacobian_scale,
            identity_shift);
    });
    report(
        succeeded,
        label + " adjoint affine inverse application");
    if(succeeded)
    {
        nonlinear_operator.jacobian_u_adjoint(solution, image);
        vector_operations.assign_mul(
            jacobian_scale,
            image,
            identity_shift,
            solution,
            residual);
        vector_operations.add_mul(
            scalar_type(-1),
            right_hand_side,
            residual);
        const scalar_type scale = std::max(
            scalar_type(1),
            vector_operations.norm_l2(right_hand_side));
        const scalar_type error =
            vector_operations.norm_l2(residual)/scale;
        report(
            error <= tolerance,
            label + " adjoint affine inverse residual, relative error=" +
                std::to_string(static_cast<double>(error)));
    }

    vector_operations.stop_use_vectors(solution, image, residual);
    vector_operations.free_vectors(solution, image, residual);
}

} // namespace tests
} // namespace nonlinear_operators

#endif
