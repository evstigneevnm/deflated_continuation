#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_BLOCK_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEX_AFFINE_BLOCK_OPERATOR_H__

#include <complex>
#include <cstddef>
#include <type_traits>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

#include "complex_affine_factor.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class ProductVectorSpace, class RealOperator>
class complex_affine_block_operator
{
public:
    using vector_space_type = ProductVectorSpace;
    using first_space_type =
        typename vector_space_type::first_space_type;
    using second_space_type =
        typename vector_space_type::second_space_type;
    using scalar_type = typename vector_space_type::scalar_type;
    using complex_type = std::complex<scalar_type>;
    using vector_type = typename vector_space_type::vector_type;
    using factor_type = complex_affine_factor<scalar_type>;

    static_assert(
        std::is_same<
            typename first_space_type::vector_type,
            typename second_space_type::vector_type>::value,
        "complex affine blocks require identical component vector types");

    complex_affine_block_operator(
        const vector_space_type& vector_space,
        const RealOperator& real_operator,
        complex_type operator_scale,
        complex_type diagonal_shift)
        : vector_space_(vector_space),
          real_operator_(real_operator),
          operator_scale_(operator_scale),
          diagonal_shift_(diagonal_shift),
          applied_first_(vector_space_.first_space(), true, true),
          applied_second_(vector_space_.second_space(), true, true)
    {
        validate(factor_type{operator_scale_, diagonal_shift_});
    }

    complex_affine_block_operator(
        const vector_space_type& vector_space,
        const RealOperator& real_operator,
        const factor_type& factor)
        : complex_affine_block_operator(
              vector_space,
              real_operator,
              factor.operator_scale,
              factor.diagonal_shift)
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        ++operator_calls_;

        if(operator_scale_ != complex_type{})
        {
            ++component_operator_calls_;
            if(!nmfd::solvers::krylov::apply_operator(
                   real_operator_,
                   source.first,
                   *applied_first_))
            {
                ++component_operator_failures_;
                return false;
            }

            ++component_operator_calls_;
            if(!nmfd::solvers::krylov::apply_operator(
                   real_operator_,
                   source.second,
                   *applied_second_))
            {
                ++component_operator_failures_;
                return false;
            }
        }

        apply_affine_combination(source, destination);
        return true;
    }

    complex_type operator_scale() const
    {
        return operator_scale_;
    }

    complex_type diagonal_shift() const
    {
        return diagonal_shift_;
    }

    factor_type factor() const
    {
        return factor_type{operator_scale_, diagonal_shift_};
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    std::size_t component_operator_calls() const
    {
        return component_operator_calls_;
    }

    std::size_t component_operator_failures() const
    {
        return component_operator_failures_;
    }

    void reset_counters() const
    {
        operator_calls_ = 0;
        component_operator_calls_ = 0;
        component_operator_failures_ = 0;
    }

private:
    void apply_affine_combination(
        const vector_type& source,
        vector_type& destination) const
    {
        const scalar_type alpha_real = operator_scale_.real();
        const scalar_type alpha_imaginary = operator_scale_.imag();
        const scalar_type beta_real = diagonal_shift_.real();
        const scalar_type beta_imaginary = diagonal_shift_.imag();
        auto& first_space = vector_space_.first_space();
        auto& second_space = vector_space_.second_space();

        if(operator_scale_ == complex_type{})
        {
            first_space.assign_lin_comb(
                beta_real,
                source.first,
                -beta_imaginary,
                source.second,
                destination.first);
            second_space.assign_lin_comb(
                beta_imaginary,
                source.first,
                beta_real,
                source.second,
                destination.second);
            return;
        }

        first_space.assign_lin_comb(
            alpha_real,
            *applied_first_,
            -alpha_imaginary,
            *applied_second_,
            destination.first);
        first_space.add_lin_comb(
            beta_real,
            source.first,
            scalar_type(1),
            destination.first);
        first_space.add_lin_comb(
            -beta_imaginary,
            source.second,
            scalar_type(1),
            destination.first);

        second_space.assign_lin_comb(
            alpha_imaginary,
            *applied_first_,
            alpha_real,
            *applied_second_,
            destination.second);
        second_space.add_lin_comb(
            beta_imaginary,
            source.first,
            scalar_type(1),
            destination.second);
        second_space.add_lin_comb(
            beta_real,
            source.second,
            scalar_type(1),
            destination.second);
    }

    const vector_space_type& vector_space_;
    const RealOperator& real_operator_;
    complex_type operator_scale_;
    complex_type diagonal_shift_;
    mutable nmfd::detail::vector_wrap<
        first_space_type,
        true,
        true> applied_first_;
    mutable nmfd::detail::vector_wrap<
        second_space_type,
        true,
        true> applied_second_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t component_operator_calls_ = 0;
    mutable std::size_t component_operator_failures_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
