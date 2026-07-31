#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_NONLINEAR_OPERATOR_REAL_AFFINE_INVERSE_PROVIDER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_NONLINEAR_OPERATOR_REAL_AFFINE_INVERSE_PROVIDER_H__

#include <cstddef>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "affine_inverse_health.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorOperations, class NonlinearOperator>
class nonlinear_operator_real_affine_inverse_provider
{
public:
    using vector_operations_type = VectorOperations;
    using nonlinear_operator_type = NonlinearOperator;
    using scalar_type =
        typename vector_operations_type::scalar_type;
    using vector_type =
        typename vector_operations_type::vector_type;
    using health_type = affine_inverse_health<scalar_type>;

    nonlinear_operator_real_affine_inverse_provider(
        vector_operations_type& vector_operations,
        nonlinear_operator_type& nonlinear_operator)
        : vector_operations_(vector_operations),
          nonlinear_operator_(nonlinear_operator)
    {
    }

    bool apply(
        scalar_type jacobian_scale,
        scalar_type identity_shift,
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        vector_operations_.assign(right_hand_side, solution);
        using result_type = decltype(
            nonlinear_operator_.preconditioner_jacobian_affine_u(
                solution,
                jacobian_scale,
                identity_shift));
        bool succeeded = true;
        if constexpr(std::is_void<result_type>::value)
        {
            nonlinear_operator_.preconditioner_jacobian_affine_u(
                solution,
                jacobian_scale,
                identity_shift);
        }
        else
        {
            succeeded = static_cast<bool>(
                nonlinear_operator_.
                    preconditioner_jacobian_affine_u(
                        solution,
                        jacobian_scale,
                        identity_shift));
        }
        succeeded =
            succeeded &&
            vector_operations_.check_is_valid_number(solution);
        if(!succeeded)
            ++failed_applications_;
        return succeeded;
    }

    nonlinear_operator_type& nonlinear_operator() const
    {
        return nonlinear_operator_;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

    std::size_t failed_applications() const
    {
        return failed_applications_;
    }

    health_type health(
        scalar_type jacobian_scale,
        scalar_type identity_shift) const
    {
        return query_health(
            nonlinear_operator_,
            jacobian_scale,
            identity_shift,
            0);
    }

private:
    template<class Operator>
    static auto query_health(
        const Operator& nonlinear_operator,
        scalar_type jacobian_scale,
        scalar_type identity_shift,
        int)
        -> decltype(
            nonlinear_operator.
                preconditioner_jacobian_affine_diagonal_range(
                    jacobian_scale,
                    identity_shift),
            health_type{})
    {
        const auto range =
            nonlinear_operator.
                preconditioner_jacobian_affine_diagonal_range(
                    jacobian_scale,
                    identity_shift);
        health_type result;
        result.available = true;
        result.minimum_abs_denominator =
            static_cast<scalar_type>(range.first);
        result.maximum_abs_denominator =
            static_cast<scalar_type>(range.second);
        query_relative_health(
            nonlinear_operator,
            jacobian_scale,
            identity_shift,
            result,
            0);
        return result;
    }

    template<class Operator>
    static health_type query_health(
        const Operator&,
        scalar_type,
        scalar_type,
        long)
    {
        return {};
    }

    template<class Operator>
    static auto query_relative_health(
        const Operator& nonlinear_operator,
        scalar_type jacobian_scale,
        scalar_type identity_shift,
        health_type& result,
        int)
        -> decltype(
            nonlinear_operator.
                preconditioner_jacobian_affine_min_relative_diagonal(
                    jacobian_scale,
                    identity_shift),
            void())
    {
        result.minimum_relative_denominator =
            static_cast<scalar_type>(
                nonlinear_operator.
                    preconditioner_jacobian_affine_min_relative_diagonal(
                        jacobian_scale,
                        identity_shift));
        result.relative_denominator_available = true;
    }

    template<class Operator>
    static void query_relative_health(
        const Operator&,
        scalar_type,
        scalar_type,
        health_type&,
        long)
    {
    }

    vector_operations_type& vector_operations_;
    nonlinear_operator_type& nonlinear_operator_;
    mutable std::size_t apply_calls_ = 0;
    mutable std::size_t failed_applications_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
