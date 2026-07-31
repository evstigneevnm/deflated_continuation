#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_FACTORIZED_INVERSE_SOLVER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_FACTORIZED_INVERSE_SOLVER_H__

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/operator_apply.h>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorSpace, class FactorSolver>
class factorized_inverse_solver
{
public:
    using vector_space_type = VectorSpace;
    using factor_solver_type = FactorSolver;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    static constexpr std::size_t no_factor_index =
        std::numeric_limits<std::size_t>::max();

    factorized_inverse_solver(
        const vector_space_type& vector_space,
        std::vector<std::shared_ptr<factor_solver_type>> factors)
        : vector_space_(vector_space),
          factors_(std::move(factors)),
          first_(vector_space_, true, true),
          second_(vector_space_, true, true)
    {
        if(factors_.empty())
            throw std::invalid_argument(
                "factorized inverse requires at least one factor solver");
        for(const auto& factor : factors_)
        {
            if(!factor)
                throw std::invalid_argument(
                    "factorized inverse received a null factor solver");
        }
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++solve_calls_;
        last_failed_factor_ = no_factor_index;
        completed_factors_last_solve_ = 0;
        vector_space_.assign(right_hand_side, *first_);
        vector_type* source = &*first_;
        vector_type* destination = &*second_;

        for(std::size_t index = 0; index < factors_.size(); ++index)
        {
            vector_space_.assign_scalar(scalar_type{}, *destination);
            ++factor_solve_calls_;
            if(!nmfd::solvers::krylov::solve_operator(
                   *factors_[index],
                   *source,
                   *destination))
            {
                ++failed_solves_;
                last_failed_factor_ = index;
                return false;
            }
            std::swap(source, destination);
            ++completed_factors_last_solve_;
        }

        vector_space_.assign(*source, solution);
        return true;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

    std::size_t factor_solve_calls() const
    {
        return factor_solve_calls_;
    }

    std::size_t failed_solves() const
    {
        return failed_solves_;
    }

    std::size_t factor_count() const
    {
        return factors_.size();
    }

    std::size_t last_failed_factor() const
    {
        return last_failed_factor_;
    }

    std::size_t completed_factors_last_solve() const
    {
        return completed_factors_last_solve_;
    }

    const factor_solver_type& factor(std::size_t index) const
    {
        return *factors_.at(index);
    }

    void reset_statistics() const
    {
        solve_calls_ = 0;
        factor_solve_calls_ = 0;
        failed_solves_ = 0;
        last_failed_factor_ = no_factor_index;
        completed_factors_last_solve_ = 0;
    }

private:
    const vector_space_type& vector_space_;
    std::vector<std::shared_ptr<factor_solver_type>> factors_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> first_;
    mutable nmfd::detail::vector_wrap<vector_space_type, true, true> second_;
    mutable std::size_t solve_calls_ = 0;
    mutable std::size_t factor_solve_calls_ = 0;
    mutable std::size_t failed_solves_ = 0;
    mutable std::size_t last_failed_factor_ = no_factor_index;
    mutable std::size_t completed_factors_last_solve_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
