#ifndef TIME_STEPPER_RUNGE_KUTTA_EXPLICIT_RK_STEP_H
#define TIME_STEPPER_RUNGE_KUTTA_EXPLICIT_RK_STEP_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <time_stepper/detail/butcher_tables.h>
#include <time_stepper/runge_kutta/stage_context.h>
#include <time_stepper/runge_kutta/stage_workspace.h>
#include <time_stepper/runge_kutta/step_result.h>
#include <time_stepper/semidiscrete/identity_mass_rate_provider.h>

namespace time_steppers
{
namespace runge_kutta
{

template<
    class VectorOperations,
    class RateProvider = semidiscrete::identity_mass_rate_provider>
class explicit_rk_step
{
public:
    using vector_operations_type = VectorOperations;
    using rate_provider_type = RateProvider;
    using scalar_type = typename vector_operations_type::scalar_type;
    using vector_type = typename vector_operations_type::vector_type;

    explicit explicit_rk_step(
        vector_operations_type& vector_operations,
        std::string method_name,
        rate_provider_type rate_provider = rate_provider_type()):
        vector_operations_(&vector_operations),
        method_name_(std::move(method_name)),
        tableau_(detail::butcher_tables().set_table_by_name(method_name_)),
        rate_provider_(std::move(rate_provider)),
        workspace_(vector_operations, checked_stage_count(tableau_, method_name_))
    {
    }

    explicit_rk_step(const explicit_rk_step&) = delete;
    explicit_rk_step& operator=(const explicit_rk_step&) = delete;
    explicit_rk_step(explicit_rk_step&&) = delete;
    explicit_rk_step& operator=(explicit_rk_step&&) = delete;

    template<class Problem>
    step_result advance(
        Problem& problem,
        const typename Problem::scalar_type time,
        const typename Problem::scalar_type step_size,
        const typename Problem::vector_type& current,
        const typename Problem::parameter_type& parameter,
        typename Problem::vector_type& next,
        typename Problem::vector_type& error_estimate)
    {
        semidiscrete::validate_problem_contract<Problem>();
        static_assert(
            std::is_same<typename Problem::scalar_type, scalar_type>::value,
            "The Runge-Kutta step and semidiscrete problem must use the same scalar_type.");
        static_assert(
            std::is_same<typename Problem::vector_type, vector_type>::value,
            "The Runge-Kutta step and semidiscrete problem must use the same vector_type.");

        step_result result;
        result.error_estimate_available = tableau_.is_embedded();
        const std::size_t stages = tableau_.get_size();

        for(std::size_t stage = 0; stage < stages; ++stage)
        {
            vector_operations_->assign(current, workspace_.stage_state());
            for(std::size_t previous = 0; previous < stage; ++previous)
            {
                const scalar_type coefficient =
                    step_size*tableau_.template get_A<scalar_type>(stage, previous);
                if(coefficient != scalar_type(0))
                {
                    vector_operations_->add_mul(
                        coefficient,
                        workspace_.stage_rate(previous),
                        workspace_.stage_state());
                }
            }

            const stage_context<scalar_type> context{
                time,
                time+step_size*tableau_.template get_c<scalar_type>(stage),
                step_size,
                stage,
                stages};
            if(!rate_provider_.evaluate(
                *vector_operations_,
                problem,
                context,
                workspace_.stage_state(),
                parameter,
                workspace_.stage_rate(stage)))
            {
                result.status = step_status::rate_evaluation_failure;
                result.failed_stage = stage;
                return result;
            }
            ++result.rate_evaluations;
        }

        vector_operations_->assign(current, next);
        for(std::size_t stage = 0; stage < stages; ++stage)
        {
            const scalar_type coefficient =
                step_size*tableau_.template get_b<scalar_type>(stage);
            if(coefficient != scalar_type(0))
            {
                vector_operations_->add_mul(coefficient, workspace_.stage_rate(stage), next);
            }
        }

        vector_operations_->assign_scalar(scalar_type(0), error_estimate);
        if(tableau_.is_embedded())
        {
            for(std::size_t stage = 0; stage < stages; ++stage)
            {
                const scalar_type coefficient =
                    step_size*tableau_.template get_err_b<scalar_type>(stage);
                if(coefficient != scalar_type(0))
                {
                    vector_operations_->add_mul(
                        coefficient,
                        workspace_.stage_rate(stage),
                        error_estimate);
                }
            }
        }

        return result;
    }

    const std::string& method_name() const
    {
        return method_name_;
    }

    std::size_t stage_count() const
    {
        return tableau_.get_size();
    }

    bool has_embedded_error_estimate() const
    {
        return tableau_.is_embedded();
    }

    std::size_t workspace_allocation_count() const
    {
        return workspace_.allocation_count();
    }

    rate_provider_type& rate_provider()
    {
        return rate_provider_;
    }

    const rate_provider_type& rate_provider() const
    {
        return rate_provider_;
    }

private:
    static std::size_t checked_stage_count(
        const detail::tableu& tableau,
        const std::string& method_name)
    {
        if(tableau.get_type() != detail::tableu::ERK)
        {
            throw std::invalid_argument(
                "explicit_rk_step requires an explicit Runge-Kutta tableau; method " +
                method_name + " is not explicit.");
        }
        for(std::size_t row = 0; row < tableau.get_size(); ++row)
        {
            for(std::size_t column = row; column < tableau.get_size(); ++column)
            {
                if(tableau.template get_A<long double>(row, column) != 0.0L)
                {
                    throw std::invalid_argument(
                        "explicit_rk_step requires a strictly lower-triangular tableau; method " +
                        method_name + " has an implicit stage coefficient.");
                }
            }
        }
        return tableau.get_size();
    }

    vector_operations_type* vector_operations_;
    std::string method_name_;
    detail::tableu tableau_;
    rate_provider_type rate_provider_;
    stage_workspace<vector_operations_type> workspace_;
};

} // namespace runge_kutta
} // namespace time_steppers

#endif
