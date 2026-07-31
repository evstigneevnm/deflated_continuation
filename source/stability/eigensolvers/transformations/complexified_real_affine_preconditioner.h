#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEXIFIED_REAL_AFFINE_PRECONDITIONER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEXIFIED_REAL_AFFINE_PRECONDITIONER_H__

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include <nmfd/detail/vector_wrap.h>

#include "complex_affine_preconditioner.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<
    class ProductVectorSpace,
    class ComplexVectorSpace,
    class ComplexVectorBridge,
    class LinearOperator,
    class RealAffineInverseProvider>
class complexified_real_affine_preconditioner
{
public:
    using product_space_type = ProductVectorSpace;
    using complex_space_type = ComplexVectorSpace;
    using bridge_type = ComplexVectorBridge;
    using operator_type = LinearOperator;
    using provider_type = RealAffineInverseProvider;
    using vector_type = typename complex_space_type::vector_type;
    using factor_type =
        complex_affine_factor<
            typename product_space_type::norm_type>;
    using implementation_type =
        complex_affine_preconditioner<
            product_space_type,
            operator_type,
            provider_type>;
    using health_type = typename implementation_type::health_type;

    static_assert(
        std::is_same<
            product_space_type,
            typename bridge_type::product_space_type>::value,
        "complexified preconditioner product space must match its bridge");
    static_assert(
        std::is_same<
            complex_space_type,
            typename bridge_type::complex_space_type>::value,
        "complexified preconditioner complex space must match its bridge");

    complexified_real_affine_preconditioner(
        const product_space_type& product_space,
        const bridge_type& bridge,
        std::shared_ptr<const provider_type> provider,
        factor_type factor)
        : bridge_(bridge),
          implementation_(
              product_space,
              std::move(provider),
              std::move(factor)),
          split_right_hand_side_(product_space, true, true),
          split_solution_(product_space, true, true)
    {
    }

    void set_operator(std::shared_ptr<const operator_type> linear_operator)
    {
        implementation_.set_operator(std::move(linear_operator));
    }

    bool apply(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        bridge_.unpack(
            right_hand_side,
            *split_right_hand_side_);
        if(!implementation_.apply(
               *split_right_hand_side_,
               *split_solution_))
        {
            ++failed_applications_;
            return false;
        }
        bridge_.pack(*split_solution_, solution);
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
        return implementation_.factor();
    }

    const provider_type& provider() const
    {
        return implementation_.provider();
    }

    const implementation_type& real_block_preconditioner() const
    {
        return implementation_;
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
        return implementation_.health();
    }

private:
    const bridge_type& bridge_;
    implementation_type implementation_;
    mutable nmfd::detail::vector_wrap<
        product_space_type,
        true,
        true> split_right_hand_side_;
    mutable nmfd::detail::vector_wrap<
        product_space_type,
        true,
        true> split_solution_;
    mutable std::size_t apply_calls_ = 0;
    mutable std::size_t failed_applications_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
