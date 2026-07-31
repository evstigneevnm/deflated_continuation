#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_POINTWISE_DIAGONAL_PRECONDITIONER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_POINTWISE_DIAGONAL_PRECONDITIONER_H__

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <nmfd/detail/vector_wrap.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace, class LinearOperator>
class pointwise_diagonal_preconditioner
{
public:
    using vector_space_type = VectorSpace;
    using operator_type = LinearOperator;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    pointwise_diagonal_preconditioner(
        const vector_space_type& vector_space,
        const vector_type& diagonal)
        : vector_space_(vector_space),
          inverse_diagonal_(vector_space_, true, true)
    {
        if(
            vector_space_.get_size(diagonal) !=
            vector_space_.get_default_size())
        {
            throw std::invalid_argument(
                "pointwise diagonal preconditioner size mismatch");
        }
        vector_space_.assign_scalar(
            scalar_type(1),
            *inverse_diagonal_);
        vector_space_.div_pointwise(
            *inverse_diagonal_,
            scalar_type(1),
            diagonal);
        if(
            !vector_space_.check_is_valid_number(
                *inverse_diagonal_))
        {
            throw std::invalid_argument(
                "pointwise diagonal preconditioner has a zero or invalid diagonal");
        }
    }

    void set_operator(std::shared_ptr<const operator_type>)
    {
    }

    bool apply(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        vector_space_.mul_pointwise(
            scalar_type(1),
            *inverse_diagonal_,
            scalar_type(1),
            right_hand_side,
            solution);
        return true;
    }

    bool apply(vector_type& vector) const
    {
        ++apply_calls_;
        vector_space_.mul_pointwise(
            vector,
            scalar_type(1),
            *inverse_diagonal_);
        return true;
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        return apply(right_hand_side, solution);
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

private:
    const vector_space_type& vector_space_;
    nmfd::detail::vector_wrap<
        vector_space_type,
        true,
        true> inverse_diagonal_;
    mutable std::size_t apply_calls_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
