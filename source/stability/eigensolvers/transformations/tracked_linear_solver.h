#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_TRACKED_LINEAR_SOLVER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_TRACKED_LINEAR_SOLVER_H__

#include <algorithm>
#include <cstddef>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class LinearSolver, class LinearOperator>
class tracked_linear_solver
{
public:
    using solver_type = LinearSolver;
    using operator_type = LinearOperator;
    using vector_type = typename solver_type::vector_type;
    using norm_type = typename solver_type::norm_type;

    tracked_linear_solver(
        solver_type& solver,
        const operator_type& linear_operator)
        : solver_(solver),
          linear_operator_(linear_operator)
    {
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++solve_calls_;
        const bool succeeded = solver_.solve(
            linear_operator_,
            right_hand_side,
            solution);
        const int signed_iterations =
            solver_.monitor().iters_performed();
        const std::size_t iterations = signed_iterations > 0
            ? static_cast<std::size_t>(signed_iterations)
            : std::size_t{};
        total_iterations_ += iterations;
        maximum_iterations_ =
            std::max(maximum_iterations_, iterations);
        last_iterations_ = iterations;
        last_residual_available_ = succeeded || iterations > 0;
        if(last_residual_available_)
            last_residual_ = solver_.monitor().resid_norm();
        if(!succeeded)
            ++failed_solves_;
        return succeeded;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

    std::size_t failed_solves() const
    {
        return failed_solves_;
    }

    std::size_t total_iterations() const
    {
        return total_iterations_;
    }

    std::size_t maximum_iterations() const
    {
        return maximum_iterations_;
    }

    std::size_t last_iterations() const
    {
        return last_iterations_;
    }

    norm_type last_residual() const
    {
        return last_residual_;
    }

    bool last_residual_available() const
    {
        return last_residual_available_;
    }

    solver_type& linear_solver()
    {
        return solver_;
    }

    const solver_type& linear_solver() const
    {
        return solver_;
    }

    const operator_type& linear_operator() const
    {
        return linear_operator_;
    }

    void reset_statistics() const
    {
        solve_calls_ = 0;
        failed_solves_ = 0;
        total_iterations_ = 0;
        maximum_iterations_ = 0;
        last_iterations_ = 0;
        last_residual_ = norm_type{};
        last_residual_available_ = false;
    }

private:
    solver_type& solver_;
    const operator_type& linear_operator_;
    mutable std::size_t solve_calls_ = 0;
    mutable std::size_t failed_solves_ = 0;
    mutable std::size_t total_iterations_ = 0;
    mutable std::size_t maximum_iterations_ = 0;
    mutable std::size_t last_iterations_ = 0;
    mutable norm_type last_residual_ = norm_type{};
    mutable bool last_residual_available_ = false;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
