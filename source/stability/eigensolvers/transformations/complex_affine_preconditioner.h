#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_PRECONDITIONER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_PRECONDITIONER_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <nmfd/detail/vector_wrap.h>

#include "complex_affine_factor.h"
#include "affine_inverse_health.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

namespace detail
{

template<class Provider, class Scalar, class Vector>
bool apply_real_affine_inverse(
    const Provider& provider,
    Scalar jacobian_scale,
    Scalar identity_shift,
    const Vector& right_hand_side,
    Vector& solution)
{
    using result_type = decltype(provider.apply(
        jacobian_scale,
        identity_shift,
        right_hand_side,
        solution));
    if constexpr(std::is_void<result_type>::value)
    {
        provider.apply(
            jacobian_scale,
            identity_shift,
            right_hand_side,
            solution);
        return true;
    }
    else
    {
        return static_cast<bool>(provider.apply(
            jacobian_scale,
            identity_shift,
            right_hand_side,
            solution));
    }
}

} // namespace detail

template<
    class ProductVectorSpace,
    class LinearOperator,
    class RealAffineInverseProvider>
class complex_affine_preconditioner
{
public:
    using vector_space_type = ProductVectorSpace;
    using operator_type = LinearOperator;
    using provider_type = RealAffineInverseProvider;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using factor_type = complex_affine_factor<norm_type>;
    using health_type = affine_inverse_health<norm_type>;

    complex_affine_preconditioner(
        const vector_space_type& vector_space,
        std::shared_ptr<const provider_type> provider,
        factor_type factor)
        : vector_space_(vector_space),
          provider_(std::move(provider)),
          factor_(std::move(factor)),
          rotated_right_hand_side_(
              vector_space_,
              true,
              true),
          temporary_(vector_space_, true, true)
    {
        if(!provider_)
            throw std::invalid_argument(
                "complex affine preconditioner requires a provider");
        validate(factor_);
    }

    void set_operator(std::shared_ptr<const operator_type>)
    {
    }

    bool apply(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        const bool succeeded =
            apply_factor_preconditioner(right_hand_side);
        if(
            !succeeded ||
            !vector_space_.check_is_valid_number(*temporary_))
        {
            ++failed_applications_;
            return false;
        }
        vector_space_.assign(*temporary_, solution);
        return true;
    }

    bool apply(vector_type& vector) const
    {
        return apply(
            static_cast<const vector_type&>(vector),
            vector);
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        return apply(right_hand_side, solution);
    }

    const factor_type& factor() const
    {
        return factor_;
    }

    const provider_type& provider() const
    {
        return *provider_;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

    std::size_t failed_applications() const
    {
        return failed_applications_;
    }

    health_type health() const
    {
        const norm_type operator_scale_real =
            factor_.operator_scale.real();
        const norm_type operator_scale_imaginary =
            factor_.operator_scale.imag();
        const norm_type operator_scale_norm =
            std::hypot(
                operator_scale_real,
                operator_scale_imaginary);
        if(
            !(operator_scale_norm >
              std::numeric_limits<norm_type>::min()))
        {
            return {};
        }
        const norm_type phase_real =
            operator_scale_real/operator_scale_norm;
        const norm_type phase_imaginary =
            -operator_scale_imaginary/operator_scale_norm;
        const norm_type shifted_real =
            phase_real*factor_.diagonal_shift.real() -
            phase_imaginary*factor_.diagonal_shift.imag();
        return provider_->health(
            static_cast<scalar_type>(operator_scale_norm),
            static_cast<scalar_type>(shifted_real));
    }

private:
    bool apply_factor_preconditioner(
        const vector_type& right_hand_side) const
    {
        const norm_type operator_scale_real =
            factor_.operator_scale.real();
        const norm_type operator_scale_imaginary =
            factor_.operator_scale.imag();
        const norm_type operator_scale_norm =
            std::hypot(
                operator_scale_real,
                operator_scale_imaginary);
        if(
            operator_scale_norm >
            std::numeric_limits<norm_type>::min())
        {
            const norm_type phase_real =
                operator_scale_real/operator_scale_norm;
            const norm_type phase_imaginary =
                -operator_scale_imaginary/operator_scale_norm;
            const norm_type shifted_real =
                phase_real*factor_.diagonal_shift.real() -
                phase_imaginary*
                    factor_.diagonal_shift.imag();

            vector_space_.first_space().assign_lin_comb(
                static_cast<scalar_type>(phase_real),
                right_hand_side.first,
                static_cast<scalar_type>(-phase_imaginary),
                right_hand_side.second,
                (*rotated_right_hand_side_).first);
            vector_space_.second_space().assign_lin_comb(
                static_cast<scalar_type>(phase_imaginary),
                right_hand_side.first,
                static_cast<scalar_type>(phase_real),
                right_hand_side.second,
                (*rotated_right_hand_side_).second);

            // The phase makes the Jacobian coefficient real. The model
            // supplies the two real diagonal-block inverses; GMRES retains
            // the remaining imaginary scalar coupling.
            return
                detail::apply_real_affine_inverse(
                    *provider_,
                    static_cast<scalar_type>(
                        operator_scale_norm),
                    static_cast<scalar_type>(shifted_real),
                    (*rotated_right_hand_side_).first,
                    (*temporary_).first) &&
                detail::apply_real_affine_inverse(
                    *provider_,
                    static_cast<scalar_type>(
                        operator_scale_norm),
                    static_cast<scalar_type>(shifted_real),
                    (*rotated_right_hand_side_).second,
                    (*temporary_).second);
        }

        const norm_type diagonal_real =
            factor_.diagonal_shift.real();
        const norm_type diagonal_imaginary =
            factor_.diagonal_shift.imag();
        const norm_type diagonal_norm =
            diagonal_real*diagonal_real +
            diagonal_imaginary*diagonal_imaginary;
        if(
            !(diagonal_norm >
              std::numeric_limits<norm_type>::min()))
        {
            return false;
        }
        vector_space_.first_space().assign_lin_comb(
            static_cast<scalar_type>(
                diagonal_real/diagonal_norm),
            right_hand_side.first,
            static_cast<scalar_type>(
                diagonal_imaginary/diagonal_norm),
            right_hand_side.second,
            (*temporary_).first);
        vector_space_.second_space().assign_lin_comb(
            static_cast<scalar_type>(
                -diagonal_imaginary/diagonal_norm),
            right_hand_side.first,
            static_cast<scalar_type>(
                diagonal_real/diagonal_norm),
            right_hand_side.second,
            (*temporary_).second);
        return true;
    }

    const vector_space_type& vector_space_;
    std::shared_ptr<const provider_type> provider_;
    factor_type factor_;
    mutable nmfd::detail::vector_wrap<
        vector_space_type,
        true,
        true> rotated_right_hand_side_;
    mutable nmfd::detail::vector_wrap<
        vector_space_type,
        true,
        true> temporary_;
    mutable std::size_t apply_calls_ = 0;
    mutable std::size_t failed_applications_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
