#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_INVERSE_COMPOSED_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_INVERSE_COMPOSED_OPERATOR_H__

#include <cstddef>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace, class NumeratorOperator, class DenominatorSolver>
class inverse_composed_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    inverse_composed_operator(
        const vector_space_type& vector_space,
        const NumeratorOperator& numerator,
        DenominatorSolver& denominator_solver)
        : vector_space_(vector_space),
          numerator_(numerator),
          denominator_solver_(denominator_solver),
          right_hand_side_(vector_space_, true, true)
    {
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        if(!nmfd::solvers::krylov::apply_operator(
               numerator_,
               source,
               *right_hand_side_))
        {
            ++numerator_failures_;
            return false;
        }

        // A nonzero or history-dependent initial guess would make an
        // inexact solve cease to be a deterministic linear operator.
        vector_space_.assign_scalar(scalar_type{}, destination);
        ++inner_solver_calls_;
        if(!nmfd::solvers::krylov::solve_operator(
               denominator_solver_,
               *right_hand_side_,
               destination))
        {
            ++inner_solver_failures_;
            return false;
        }
        return true;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    std::size_t inner_solver_calls() const
    {
        return inner_solver_calls_;
    }

    std::size_t numerator_failures() const
    {
        return numerator_failures_;
    }

    std::size_t inner_solver_failures() const
    {
        return inner_solver_failures_;
    }

private:
    const vector_space_type& vector_space_;
    const NumeratorOperator& numerator_;
    DenominatorSolver& denominator_solver_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true>
        right_hand_side_;
    mutable std::size_t operator_calls_ = 0;
    mutable std::size_t inner_solver_calls_ = 0;
    mutable std::size_t numerator_failures_ = 0;
    mutable std::size_t inner_solver_failures_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
