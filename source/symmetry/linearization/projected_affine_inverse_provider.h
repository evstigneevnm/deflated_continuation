#ifndef __SYMMETRY_LINEARIZATION_PROJECTED_AFFINE_INVERSE_PROVIDER_H__
#define __SYMMETRY_LINEARIZATION_PROJECTED_AFFINE_INVERSE_PROVIDER_H__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <stability/eigensolvers/transformations/affine_inverse_health.h>

namespace symmetry
{
namespace linearization
{

// Matches projected_linear_operator: PJP on the tangent space and a
// configurable scalar completion on the gauge complement. Newton solves use
// the default identity completion. Stability assemblies should place the
// artificial gauge eigenvalue strictly inside the configured stable half-plane.
template<
    class VectorOperations,
    class NonlinearOperator,
    class RealAffineInverseProvider>
class projected_affine_inverse_provider
{
public:
    using vector_operations_type = VectorOperations;
    using nonlinear_operator_type = NonlinearOperator;
    using provider_type = RealAffineInverseProvider;
    using scalar_type =
        typename vector_operations_type::scalar_type;
    using vector_type =
        typename vector_operations_type::vector_type;
    using health_type =
        stability::eigensolvers::transformations::
            affine_inverse_health<scalar_type>;

    projected_affine_inverse_provider(
        vector_operations_type& vector_operations,
        nonlinear_operator_type& nonlinear_operator,
        std::shared_ptr<const provider_type> provider,
        scalar_type gauge_completion = scalar_type(1))
        : vector_operations_(vector_operations),
          nonlinear_operator_(nonlinear_operator),
          provider_(std::move(provider)),
          gauge_completion_(gauge_completion)
    {
        if(!provider_)
            throw std::invalid_argument(
                "projected affine inverse requires a provider");
        initialize(projected_rhs_);
        initialize(gauge_rhs_);
        initialize(tangent_solution_);
    }

    projected_affine_inverse_provider(
        const projected_affine_inverse_provider&) = delete;
    projected_affine_inverse_provider& operator=(
        const projected_affine_inverse_provider&) = delete;

    ~projected_affine_inverse_provider()
    {
        release(tangent_solution_);
        release(gauge_rhs_);
        release(projected_rhs_);
    }

    bool apply(
        scalar_type jacobian_scale,
        scalar_type identity_shift,
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        nonlinear_operator_.project_current_tangent(
            right_hand_side,
            projected_rhs_);
        vector_operations_.assign_lin_comb(
            scalar_type(1),
            right_hand_side,
            scalar_type(-1),
            projected_rhs_,
            gauge_rhs_);

        const scalar_type gauge_factor =
            jacobian_scale*gauge_completion_ +
            identity_shift;
        const scalar_type absolute_gauge_factor =
            gauge_factor < scalar_type(0) ?
                -gauge_factor :
                gauge_factor;
        if(
            !(absolute_gauge_factor >
              std::numeric_limits<scalar_type>::min()))
        {
            ++failed_applications_;
            return false;
        }

        if(!provider_->apply(
               jacobian_scale,
               identity_shift,
               projected_rhs_,
               tangent_solution_))
        {
            ++failed_applications_;
            return false;
        }

        nonlinear_operator_.project_current_tangent(
            tangent_solution_,
            tangent_solution_);

        vector_operations_.assign_lin_comb(
            scalar_type(1)/gauge_factor,
            gauge_rhs_,
            scalar_type(1),
            tangent_solution_,
            solution);
        return true;
    }

    const provider_type& provider() const
    {
        return *provider_;
    }

    scalar_type gauge_completion() const
    {
        return gauge_completion_;
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
        health_type result = provider_->health(
            jacobian_scale,
            identity_shift);
        const scalar_type gauge_factor =
            jacobian_scale*gauge_completion_ +
            identity_shift;
        const scalar_type absolute_gauge_factor =
            gauge_factor < scalar_type(0)
            ? -gauge_factor
            : gauge_factor;
        if(result.available)
        {
            result.minimum_abs_denominator = std::min(
                result.minimum_abs_denominator,
                absolute_gauge_factor);
            result.maximum_abs_denominator = std::max(
                result.maximum_abs_denominator,
                absolute_gauge_factor);
            if(result.relative_denominator_available)
            {
                const scalar_type gauge_scale =
                    (
                        jacobian_scale*gauge_completion_ <
                            scalar_type(0)
                        ? -jacobian_scale*gauge_completion_
                        : jacobian_scale*gauge_completion_
                    ) +
                    (
                        identity_shift < scalar_type(0)
                        ? -identity_shift
                        : identity_shift
                    );
                const scalar_type relative_gauge_factor =
                    gauge_scale > scalar_type(0)
                    ? absolute_gauge_factor/gauge_scale
                    : absolute_gauge_factor;
                result.minimum_relative_denominator =
                    std::min(
                        result.minimum_relative_denominator,
                        relative_gauge_factor);
            }
        }
        return result;
    }

private:
    void initialize(vector_type& vector)
    {
        vector_operations_.init_vector(vector);
        vector_operations_.start_use_vector(vector);
    }

    void release(vector_type& vector)
    {
        vector_operations_.stop_use_vector(vector);
        vector_operations_.free_vector(vector);
    }

    vector_operations_type& vector_operations_;
    nonlinear_operator_type& nonlinear_operator_;
    std::shared_ptr<const provider_type> provider_;
    scalar_type gauge_completion_;
    mutable vector_type projected_rhs_;
    mutable vector_type gauge_rhs_;
    mutable vector_type tangent_solution_;
    mutable std::size_t apply_calls_ = 0;
    mutable std::size_t failed_applications_ = 0;
};

} // namespace linearization
} // namespace symmetry

#endif
