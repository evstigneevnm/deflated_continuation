#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_INEXACT_EXPONENTIAL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_INEXACT_EXPONENTIAL_OPERATOR_H__

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace, class OriginalOperator>
class inexact_exponential_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    inexact_exponential_operator(
        const vector_space_type& vector_space,
        const OriginalOperator& original_operator,
        scalar_type step,
        std::size_t power)
        : vector_space_(vector_space),
          original_operator_(original_operator),
          step_(step),
          power_(power),
          current_(vector_space_, true, true),
          next_(vector_space_, true, true),
          applied_(vector_space_, true, true)
    {
        if(power_ == 0)
            throw std::invalid_argument(
                "inexact exponential power must be positive");
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        vector_space_.assign(source, *current_);
        vector_type* current = &*current_;
        vector_type* next = &*next_;
        for(std::size_t iteration = 0; iteration < power_; ++iteration)
        {
            ++original_operator_calls_;
            if(!nmfd::solvers::krylov::apply_operator(
                   original_operator_,
                   *current,
                   *applied_))
            {
                ++original_operator_failures_;
                return false;
            }
            vector_space_.assign_lin_comb(
                scalar_type(1),
                *current,
                step_,
                *applied_,
                *next);
            std::swap(current, next);
        }
        vector_space_.assign(*current, destination);
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    std::size_t original_operator_calls() const
    {
        return original_operator_calls_;
    }

    std::size_t original_operator_failures() const
    {
        return original_operator_failures_;
    }

    scalar_type step() const
    {
        return step_;
    }

    std::size_t power() const
    {
        return power_;
    }

private:
    const vector_space_type& vector_space_;
    const OriginalOperator& original_operator_;
    scalar_type step_;
    std::size_t power_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> current_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> next_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> applied_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t original_operator_calls_ = 0;
    mutable std::size_t original_operator_failures_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
