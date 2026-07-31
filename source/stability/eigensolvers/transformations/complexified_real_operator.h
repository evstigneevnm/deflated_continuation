#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEXIFIED_REAL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_COMPLEXIFIED_REAL_OPERATOR_H__

#include <cstddef>
#include <type_traits>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

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
    class RealOperator>
class complexified_real_operator
{
public:
    using product_space_type = ProductVectorSpace;
    using complex_space_type = ComplexVectorSpace;
    using bridge_type = ComplexVectorBridge;
    using real_operator_type = RealOperator;
    using real_vector_type =
        typename product_space_type::first_space_type::vector_type;
    using vector_type = typename complex_space_type::vector_type;

    static_assert(
        std::is_same<
            product_space_type,
            typename bridge_type::product_space_type>::value,
        "complexified operator product space must match its bridge");
    static_assert(
        std::is_same<
            complex_space_type,
            typename bridge_type::complex_space_type>::value,
        "complexified operator complex space must match its bridge");

    complexified_real_operator(
        const product_space_type& product_space,
        const bridge_type& bridge,
        const real_operator_type& real_operator)
        : bridge_(bridge),
          real_operator_(real_operator),
          split_source_(product_space, true, true),
          split_destination_(product_space, true, true)
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        ++operator_calls_;
        bridge_.unpack(source, *split_source_);

        ++component_operator_calls_;
        if(!nmfd::solvers::krylov::apply_operator(
               real_operator_,
               (*split_source_).first,
               (*split_destination_).first))
        {
            ++component_operator_failures_;
            return false;
        }

        ++component_operator_calls_;
        if(!nmfd::solvers::krylov::apply_operator(
               real_operator_,
               (*split_source_).second,
               (*split_destination_).second))
        {
            ++component_operator_failures_;
            return false;
        }

        bridge_.pack(*split_destination_, destination);
        return true;
    }

    const real_operator_type& real_operator() const
    {
        return real_operator_;
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
    const bridge_type& bridge_;
    const real_operator_type& real_operator_;
    mutable nmfd::detail::vector_wrap<
        product_space_type,
        true,
        true> split_source_;
    mutable nmfd::detail::vector_wrap<
        product_space_type,
        true,
        true> split_destination_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t component_operator_calls_ = 0;
    mutable std::size_t component_operator_failures_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
