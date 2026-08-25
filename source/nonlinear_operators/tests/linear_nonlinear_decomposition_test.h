#ifndef __NONLINEAR_OPERATORS_TESTS_LINEAR_NONLINEAR_DECOMPOSITION_TEST_H__
#define __NONLINEAR_OPERATORS_TESTS_LINEAR_NONLINEAR_DECOMPOSITION_TEST_H__

#include <algorithm>
#include <string>

namespace nonlinear_operators
{
namespace tests
{

template <class VectorOperations, class NonlinearOperator, class Reporter>
void check_linear_nonlinear_decomposition(
    VectorOperations &vector_operations,
    NonlinearOperator &nonlinear_operator,
    const typename VectorOperations::vector_type &state,
    const typename VectorOperations::vector_type &direction,
    const typename VectorOperations::scalar_type parameter,
    const typename VectorOperations::scalar_type finite_difference_step,
    const typename VectorOperations::scalar_type tolerance,
    Reporter &&report,
    const std::string &label
)
{
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    vector_type state_plus;
    vector_type state_minus;
    vector_type full;
    vector_type linear;
    vector_type nonlinear;
    vector_type sum;
    vector_type full_jacobian;
    vector_type linear_jacobian;
    vector_type nonlinear_jacobian;
    vector_type jacobian_sum;
    vector_type value_plus;
    vector_type value_minus;
    vector_type finite_difference;
    vector_type difference;
    vector_operations.init_vectors(
        state_plus,
        state_minus,
        full,
        linear,
        nonlinear,
        sum,
        full_jacobian,
        linear_jacobian,
        nonlinear_jacobian,
        jacobian_sum,
        value_plus,
        value_minus,
        finite_difference,
        difference
    );
    vector_operations.start_use_vectors(
        state_plus,
        state_minus,
        full,
        linear,
        nonlinear,
        sum,
        full_jacobian,
        linear_jacobian,
        nonlinear_jacobian,
        jacobian_sum,
        value_plus,
        value_minus,
        finite_difference,
        difference
    );

    const auto relative_error = [&](const vector_type &actual, const vector_type &expected)
    {
        vector_operations.assign_mul(
            scalar_type(1),
            actual,
            scalar_type(-1),
            expected,
            difference
        );
        return vector_operations.norm_l2(difference) /
               std::max(scalar_type(1), vector_operations.norm_l2(expected));
    };

    nonlinear_operator.F(state, parameter, full);
    nonlinear_operator.linear_residual(state, parameter, linear);
    nonlinear_operator.nonlinear_residual(state, parameter, nonlinear);
    vector_operations.assign_mul(
        scalar_type(1),
        linear,
        scalar_type(1),
        nonlinear,
        sum
    );
    const scalar_type residual_error = relative_error(sum, full);
    report(
        residual_error <= tolerance,
        label + " F = F_linear + F_nonlinear, relative error=" +
            std::to_string(static_cast<double>(residual_error))
    );

    nonlinear_operator.set_linearization_point(state, parameter);
    nonlinear_operator.jacobian_u(direction, full_jacobian);
    nonlinear_operator.linear_jacobian_u(direction, linear_jacobian);
    nonlinear_operator.nonlinear_jacobian_u(direction, nonlinear_jacobian);
    vector_operations.assign_mul(
        scalar_type(1),
        linear_jacobian,
        scalar_type(1),
        nonlinear_jacobian,
        jacobian_sum
    );
    const scalar_type jacobian_error = relative_error(jacobian_sum, full_jacobian);
    report(
        jacobian_error <= tolerance,
        label + " J = J_linear + J_nonlinear, relative error=" +
            std::to_string(static_cast<double>(jacobian_error))
    );

    vector_operations.assign_mul(
        scalar_type(1),
        state,
        finite_difference_step,
        direction,
        state_plus
    );
    vector_operations.assign_mul(
        scalar_type(1),
        state,
        -finite_difference_step,
        direction,
        state_minus
    );

    const auto check_finite_difference = [&](auto &&residual, auto &&jacobian, const std::string &component)
    {
        residual(state_plus, value_plus);
        residual(state_minus, value_minus);
        vector_operations.assign_mul(
            scalar_type(1) / (scalar_type(2) * finite_difference_step),
            value_plus,
            -scalar_type(1) / (scalar_type(2) * finite_difference_step),
            value_minus,
            finite_difference
        );
        nonlinear_operator.set_linearization_point(state, parameter);
        jacobian(direction, sum);
        const scalar_type error = relative_error(sum, finite_difference);
        report(
            error <= tolerance,
            label + " " + component + " Jacobian finite difference, relative error=" +
                std::to_string(static_cast<double>(error))
        );
    };

    check_finite_difference(
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.F(value, parameter, output);
        },
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.jacobian_u(value, output);
        },
        "full"
    );
    check_finite_difference(
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.linear_residual(value, parameter, output);
        },
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.linear_jacobian_u(value, output);
        },
        "linear"
    );
    check_finite_difference(
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.nonlinear_residual(value, parameter, output);
        },
        [&](const vector_type &value, vector_type &output)
        {
            nonlinear_operator.nonlinear_jacobian_u(value, output);
        },
        "nonlinear"
    );

    vector_operations.stop_use_vectors(
        state_plus,
        state_minus,
        full,
        linear,
        nonlinear,
        sum,
        full_jacobian,
        linear_jacobian,
        nonlinear_jacobian,
        jacobian_sum,
        value_plus,
        value_minus,
        finite_difference,
        difference
    );
    vector_operations.free_vectors(
        state_plus,
        state_minus,
        full,
        linear,
        nonlinear,
        sum,
        full_jacobian,
        linear_jacobian,
        nonlinear_jacobian,
        jacobian_sum,
        value_plus,
        value_minus,
        finite_difference,
        difference
    );
}

} // namespace tests
} // namespace nonlinear_operators

#endif
