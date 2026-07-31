#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_AFFINE_PENCIL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_AFFINE_PENCIL_OPERATOR_H__

#include <cstddef>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace, class FirstOperator, class SecondOperator>
class affine_pencil_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    affine_pencil_operator(
        const vector_space_type& vector_space,
        const FirstOperator& first,
        const SecondOperator& second,
        scalar_type first_coefficient,
        scalar_type second_coefficient)
        : vector_space_(vector_space),
          first_(first),
          second_(second),
          first_coefficient_(first_coefficient),
          second_coefficient_(second_coefficient),
          temporary_(vector_space_, true, true)
    {
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        if(first_coefficient_ == scalar_type{})
        {
            if(!apply_second(source, destination))
                return false;
            vector_space_.scale(second_coefficient_, destination);
            return true;
        }
        if(second_coefficient_ == scalar_type{})
        {
            if(!apply_first(source, destination))
                return false;
            vector_space_.scale(first_coefficient_, destination);
            return true;
        }

        if(!apply_first(source, destination))
            return false;
        if(!apply_second(source, *temporary_))
            return false;
        vector_space_.add_lin_comb(
            second_coefficient_,
            *temporary_,
            first_coefficient_,
            destination);
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    scalar_type first_coefficient() const
    {
        return first_coefficient_;
    }

    scalar_type second_coefficient() const
    {
        return second_coefficient_;
    }

private:
    bool apply_first(
        const vector_type& source,
        vector_type& destination) const
    {
        return nmfd::solvers::krylov::apply_operator(
            first_,
            source,
            destination);
    }

    bool apply_second(
        const vector_type& source,
        vector_type& destination) const
    {
        return nmfd::solvers::krylov::apply_operator(
            second_,
            source,
            destination);
    }

    const vector_space_type& vector_space_;
    const FirstOperator& first_;
    const SecondOperator& second_;
    scalar_type first_coefficient_;
    scalar_type second_coefficient_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true>
        temporary_;
    mutable std::size_t operator_calls_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
