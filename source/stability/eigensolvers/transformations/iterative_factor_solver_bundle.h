#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_ITERATIVE_FACTOR_SOLVER_BUNDLE_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_ITERATIVE_FACTOR_SOLVER_BUNDLE_H__

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "complex_affine_factor.h"
#include "factor_solver_statistics.h"
#include "factorized_inverse_solver.h"
#include "tracked_linear_solver.h"

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class FactorOperator, class Preconditioner>
struct iterative_factor_solver_components
{
    std::shared_ptr<FactorOperator> linear_operator;
    std::shared_ptr<Preconditioner> preconditioner;
};

template<
    class VectorSpace,
    class FactorOperator,
    class Preconditioner,
    class LinearSolver>
class iterative_factor_solver_bundle
{
public:
    using vector_space_type = VectorSpace;
    using operator_type = FactorOperator;
    using preconditioner_type = Preconditioner;
    using solver_type = LinearSolver;
    using tracker_type =
        tracked_linear_solver<solver_type, operator_type>;
    using inverse_type =
        factorized_inverse_solver<vector_space_type, tracker_type>;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using real_type = norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using factor_type = complex_affine_factor<real_type>;
    using components_type =
        iterative_factor_solver_components<
            operator_type,
            preconditioner_type>;
    using solver_parameters_type = typename solver_type::params;
    using log_type = typename solver_type::log_type;
    using factor_statistics_type =
        factor_solver_statistics<real_type, norm_type>;
    using statistics_type =
        factor_solver_bundle_statistics<real_type, norm_type>;

    struct uniform_solver_parameters
    {
        solver_parameters_type operator()(
            const solver_parameters_type& parameters,
            const factor_type&,
            std::size_t) const
        {
            return parameters;
        }
    };

    static_assert(
        std::is_same<
            typename solver_type::vector_operations_type,
            vector_space_type>::value,
        "factor bundle vector space must match the linear solver");
    static_assert(
        std::is_same<
            typename solver_type::linear_operator_type,
            operator_type>::value,
        "factor bundle operator must match the linear solver");
    static_assert(
        std::is_same<
            typename solver_type::preconditioner_type,
            preconditioner_type>::value,
        "factor bundle preconditioner must match the linear solver");

    class factor_state
    {
    public:
        factor_state(
            factor_type descriptor,
            std::shared_ptr<operator_type> linear_operator,
            std::shared_ptr<preconditioner_type> preconditioner,
            std::shared_ptr<solver_type> linear_solver,
            std::shared_ptr<tracker_type> tracked_solver,
            solver_parameters_type solver_parameters)
            : descriptor_(std::move(descriptor)),
              linear_operator_(std::move(linear_operator)),
              preconditioner_(std::move(preconditioner)),
              linear_solver_(std::move(linear_solver)),
              tracked_solver_(std::move(tracked_solver)),
              solver_parameters_(std::move(solver_parameters))
        {
        }

        factor_state(const factor_state&) = delete;
        factor_state& operator=(const factor_state&) = delete;
        factor_state(factor_state&&) noexcept = default;
        factor_state& operator=(factor_state&&) noexcept = default;

        const factor_type& descriptor() const
        {
            return descriptor_;
        }

        const operator_type& linear_operator() const
        {
            return *linear_operator_;
        }

        operator_type& linear_operator()
        {
            return *linear_operator_;
        }

        const preconditioner_type& preconditioner() const
        {
            return *preconditioner_;
        }

        preconditioner_type& preconditioner()
        {
            return *preconditioner_;
        }

        const solver_type& linear_solver() const
        {
            return *linear_solver_;
        }

        solver_type& linear_solver()
        {
            return *linear_solver_;
        }

        const tracker_type& tracked_solver() const
        {
            return *tracked_solver_;
        }

        tracker_type& tracked_solver()
        {
            return *tracked_solver_;
        }

        const solver_parameters_type& solver_parameters() const
        {
            return solver_parameters_;
        }

        factor_statistics_type statistics(std::size_t index) const
        {
            factor_statistics_type result;
            result.index = index;
            result.descriptor = descriptor_;
            result.solve_calls = tracked_solver_->solve_calls();
            result.failed_solves = tracked_solver_->failed_solves();
            result.total_iterations =
                tracked_solver_->total_iterations();
            result.maximum_iterations =
                tracked_solver_->maximum_iterations();
            result.last_iterations =
                tracked_solver_->last_iterations();
            result.last_residual =
                tracked_solver_->last_residual();
            result.last_residual_available =
                tracked_solver_->last_residual_available();
            return result;
        }

        void reset_statistics() const
        {
            tracked_solver_->reset_statistics();
        }

    private:
        factor_type descriptor_;
        std::shared_ptr<operator_type> linear_operator_;
        std::shared_ptr<preconditioner_type> preconditioner_;
        std::shared_ptr<solver_type> linear_solver_;
        std::shared_ptr<tracker_type> tracked_solver_;
        solver_parameters_type solver_parameters_;
    };

    template<class ComponentsFactory>
    iterative_factor_solver_bundle(
        std::shared_ptr<vector_space_type> vector_space,
        std::vector<factor_type> factors,
        const solver_parameters_type& solver_parameters,
        ComponentsFactory&& components_factory,
        log_type* log = nullptr)
        : iterative_factor_solver_bundle(
              std::move(vector_space),
              std::move(factors),
              solver_parameters,
              std::forward<ComponentsFactory>(
                  components_factory),
              uniform_solver_parameters{},
              log)
    {
    }

    template<
        class ComponentsFactory,
        class SolverParametersFactory>
    iterative_factor_solver_bundle(
        std::shared_ptr<vector_space_type> vector_space,
        std::vector<factor_type> factors,
        const solver_parameters_type& solver_parameters,
        ComponentsFactory&& components_factory,
        SolverParametersFactory&& solver_parameters_factory,
        log_type* log = nullptr)
        : vector_space_(std::move(vector_space))
    {
        if(!vector_space_)
            throw std::invalid_argument(
                "factor solver bundle requires a vector space");
        if(factors.empty())
            throw std::invalid_argument(
                "factor solver bundle requires at least one factor");

        states_.reserve(factors.size());
        std::vector<std::shared_ptr<tracker_type>> tracked_solvers;
        tracked_solvers.reserve(factors.size());
        for(std::size_t index = 0; index < factors.size(); ++index)
        {
            validate(factors[index]);
            auto components = std::invoke(
                components_factory,
                factors[index],
                index);
            if(!components.linear_operator)
                throw std::invalid_argument(
                    "factor solver factory returned a null operator");
            if(!components.preconditioner)
                throw std::invalid_argument(
                    "factor solver factory returned a null preconditioner");

            auto factor_solver_parameters = std::invoke(
                solver_parameters_factory,
                solver_parameters,
                factors[index],
                index);
            auto linear_solver = std::make_shared<solver_type>(
                vector_space_,
                log,
                factor_solver_parameters,
                components.preconditioner);
            auto tracked_solver = std::make_shared<tracker_type>(
                *linear_solver,
                *components.linear_operator);
            tracked_solvers.push_back(tracked_solver);
            states_.emplace_back(
                factors[index],
                std::move(components.linear_operator),
                std::move(components.preconditioner),
                std::move(linear_solver),
                std::move(tracked_solver),
                std::move(factor_solver_parameters));
        }
        inverse_ = std::make_unique<inverse_type>(
            *vector_space_,
            std::move(tracked_solvers));
    }

    iterative_factor_solver_bundle(
        const iterative_factor_solver_bundle&) = delete;
    iterative_factor_solver_bundle& operator=(
        const iterative_factor_solver_bundle&) = delete;
    iterative_factor_solver_bundle(
        iterative_factor_solver_bundle&&) = delete;
    iterative_factor_solver_bundle& operator=(
        iterative_factor_solver_bundle&&) = delete;

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        return inverse_->solve(right_hand_side, solution);
    }

    std::size_t factor_count() const
    {
        return states_.size();
    }

    const factor_state& factor(std::size_t index) const
    {
        return states_.at(index);
    }

    factor_state& factor(std::size_t index)
    {
        return states_.at(index);
    }

    std::size_t solve_calls() const
    {
        return inverse_->solve_calls();
    }

    std::size_t factor_solve_calls() const
    {
        return inverse_->factor_solve_calls();
    }

    std::size_t failed_solves() const
    {
        return inverse_->failed_solves();
    }

    std::size_t last_failed_factor() const
    {
        return inverse_->last_failed_factor();
    }

    std::size_t completed_factors_last_solve() const
    {
        return inverse_->completed_factors_last_solve();
    }

    statistics_type statistics() const
    {
        statistics_type result;
        result.solve_calls = solve_calls();
        result.factor_solve_calls = factor_solve_calls();
        result.failed_solves = failed_solves();
        result.last_failed_factor = last_failed_factor();
        result.completed_factors_last_solve =
            completed_factors_last_solve();
        result.factors.reserve(states_.size());
        for(std::size_t index = 0; index < states_.size(); ++index)
            result.factors.push_back(states_[index].statistics(index));
        return result;
    }

    void reset_statistics() const
    {
        inverse_->reset_statistics();
        for(const auto& state : states_)
            state.reset_statistics();
    }

private:
    std::shared_ptr<vector_space_type> vector_space_;
    std::vector<factor_state> states_;
    std::unique_ptr<inverse_type> inverse_;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
