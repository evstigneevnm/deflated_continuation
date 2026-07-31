#ifndef __STABILITY_EIGENSOLVERS_ROTATED_PROJECTED_SPECTRUM_RECOVERY_H__
#define __STABILITY_EIGENSOLVERS_ROTATED_PROJECTED_SPECTRUM_RECOVERY_H__

#include <cstddef>
#include <stdexcept>

#include <nmfd/detail/vector_wrap.h>

#include "projected_spectrum_recovery.h"
#include "ritz_recovery.h"

namespace stability
{
namespace eigensolvers
{

template<
    class ComponentVectorSpace,
    class ProductVectorSpace,
    class SmallDenseLapack>
class rotated_projected_spectrum_recovery
{
public:
    using component_space_type = ComponentVectorSpace;
    using product_space_type = ProductVectorSpace;
    using dense_lapack_type = SmallDenseLapack;
    using product_vector_type =
        typename product_space_type::vector_type;
    using result_type =
        projected_spectrum_result<
            typename component_space_type::norm_type>;
    using options_type =
        projected_spectrum_options<
            typename component_space_type::norm_type>;
    using ordinal_type =
        typename component_space_type::ordinal_type;

    rotated_projected_spectrum_recovery(
        const component_space_type& component_space,
        const product_space_type& product_space,
        const dense_lapack_type& dense_lapack)
        : component_space_(component_space),
          product_space_(product_space),
          dense_lapack_(dense_lapack)
    {
    }

    template<class OriginalOperator>
    result_type execute(
        const OriginalOperator& original_operator,
        const ritz_vector_storage<product_space_type>& transformed_vectors,
        const options_type& options = {},
        ritz_vector_storage<component_space_type>* recovered_vectors = nullptr)
        const
    {
        if(transformed_vectors.size() == 0)
        {
            result_type result;
            result.diagnostic =
                "rotated projected recovery requires transformed vectors";
            return result;
        }

        const std::size_t candidate_capacity =
            4 * transformed_vectors.size();
        ritz_vector_storage<component_space_type> candidates(
            component_space_,
            candidate_capacity);
        nmfd::detail::vector_wrap<
            product_space_type,
            true,
            true> product_candidate(product_space_);
        nmfd::detail::vector_wrap<
            component_space_type,
            true,
            true> zero(component_space_);
        component_space_.assign_scalar(
            typename component_space_type::scalar_type{},
            *zero);

        std::size_t destination = 0;
        for(std::size_t source = 0;
            source < transformed_vectors.size();
            ++source)
        {
            product_space_.assign(
                transformed_vectors.real(),
                product_ordinal(transformed_vectors.capacity()),
                product_ordinal(source),
                *product_candidate);
            append_component(
                (*product_candidate).first,
                *zero,
                candidates,
                destination++);
            append_component(
                (*product_candidate).second,
                *zero,
                candidates,
                destination++);

            product_space_.assign(
                transformed_vectors.imaginary(),
                product_ordinal(transformed_vectors.capacity()),
                product_ordinal(source),
                *product_candidate);
            append_component(
                (*product_candidate).first,
                *zero,
                candidates,
                destination++);
            append_component(
                (*product_candidate).second,
                *zero,
                candidates,
                destination++);
        }
        candidates.set_size(destination);

        projected_spectrum_recovery<
            component_space_type,
            dense_lapack_type>
            recovery(component_space_, dense_lapack_);
        return recovery.execute(
            original_operator,
            candidates,
            options,
            recovered_vectors);
    }

private:
    using product_ordinal_type =
        typename product_space_type::ordinal_type;
    using component_vector_type =
        typename component_space_type::vector_type;

    void append_component(
        const component_vector_type& source,
        const component_vector_type& zero,
        ritz_vector_storage<component_space_type>& destination,
        std::size_t column) const
    {
        if(column >= destination.capacity())
            throw std::out_of_range(
                "rotated projected recovery candidate capacity");
        component_space_.assign(
            source,
            destination.real(),
            component_ordinal(destination.capacity()),
            component_ordinal(column));
        component_space_.assign(
            zero,
            destination.imaginary(),
            component_ordinal(destination.capacity()),
            component_ordinal(column));
    }

    static ordinal_type component_ordinal(std::size_t value)
    {
        return static_cast<ordinal_type>(value);
    }

    static product_ordinal_type product_ordinal(std::size_t value)
    {
        return static_cast<product_ordinal_type>(value);
    }

    const component_space_type& component_space_;
    const product_space_type& product_space_;
    const dense_lapack_type& dense_lapack_;
};

} // namespace eigensolvers
} // namespace stability

#endif
