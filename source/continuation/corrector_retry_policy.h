#ifndef __CONTINUATION_CORRECTOR_RETRY_POLICY_H__
#define __CONTINUATION_CORRECTOR_RETRY_POLICY_H__

#include <algorithm>
#include <stdexcept>

#include <common/scalar_math.h>

namespace continuation
{

enum class step_retry_result
{
    retry,
    retry_limit,
    minimum_step
};

template<class T>
struct corrector_retry_policy
{
    unsigned int maximum_retries = 8;
    T failure_reduction_factor = T(0.5);
    T minimum_step_size = T(0);
    T minimum_step_ratio = T(1.0e-6);
    unsigned int successes_before_growth = 5;
    T success_growth_factor = T(1.25);

    void validate() const
    {
        if(!common::scalar_math::isfinite(failure_reduction_factor) ||
           failure_reduction_factor <= T(0) || failure_reduction_factor >= T(1))
        {
            throw std::invalid_argument(
                "corrector retry failure_reduction_factor must be finite and in (0,1)");
        }
        if(!common::scalar_math::isfinite(minimum_step_size) || minimum_step_size < T(0))
        {
            throw std::invalid_argument(
                "corrector retry minimum_step_size must be finite and non-negative");
        }
        if(!common::scalar_math::isfinite(minimum_step_ratio) ||
           minimum_step_ratio < T(0) || minimum_step_ratio > T(1))
        {
            throw std::invalid_argument(
                "corrector retry minimum_step_ratio must be finite and in [0,1]");
        }
        if(successes_before_growth == 0)
        {
            throw std::invalid_argument(
                "corrector retry successes_before_growth must be positive");
        }
        if(!common::scalar_math::isfinite(success_growth_factor) ||
           success_growth_factor < T(1))
        {
            throw std::invalid_argument(
                "corrector retry success_growth_factor must be finite and at least one");
        }
    }
};

template<class T>
class adaptive_step_controller
{
public:
    adaptive_step_controller(
        const T initial_step,
        const T maximum_step,
        const corrector_retry_policy<T>& policy = corrector_retry_policy<T>())
    {
        configure(initial_step, maximum_step, policy);
    }

    void configure(
        const T initial_step,
        const T maximum_step,
        const corrector_retry_policy<T>& policy)
    {
        policy.validate();
        if(!common::scalar_math::isfinite(initial_step) || initial_step <= T(0))
        {
            throw std::invalid_argument("initial continuation step must be positive and finite");
        }
        if(!common::scalar_math::isfinite(maximum_step) || maximum_step < initial_step)
        {
            throw std::invalid_argument(
                "maximum continuation step must be finite and not smaller than the initial step");
        }

        initial_step_ = initial_step;
        maximum_step_ = maximum_step;
        policy_ = policy;
        reset_semicurve();
    }

    void reset_semicurve()
    {
        step_ = initial_step_;
        retries_ = 0;
        consecutive_first_attempt_successes_ = 0;
        step_retried_ = false;
    }

    void begin_step()
    {
        retries_ = 0;
        step_retried_ = false;
    }

    step_retry_result retry_after_failure()
    {
        return retry_with_factor(policy_.failure_reduction_factor);
    }

    step_retry_result retry_with_factor(const T factor)
    {
        if(!common::scalar_math::isfinite(factor) || factor <= T(0) || factor >= T(1))
        {
            throw std::invalid_argument("continuation retry factor must be finite and in (0,1)");
        }
        if(retries_ >= policy_.maximum_retries)
        {
            return step_retry_result::retry_limit;
        }

        const T minimum_step = effective_minimum_step();
        if(step_ <= minimum_step)
        {
            return step_retry_result::minimum_step;
        }

        step_ = std::max(step_*factor, minimum_step);
        ++retries_;
        step_retried_ = true;
        consecutive_first_attempt_successes_ = 0;
        return step_retry_result::retry;
    }

    step_retry_result retry_at_step(const T requested_step)
    {
        if(!common::scalar_math::isfinite(requested_step) || requested_step <= T(0))
        {
            throw std::invalid_argument(
                "continuation retry step must be positive and finite");
        }
        if(requested_step > maximum_step_)
        {
            throw std::invalid_argument(
                "continuation retry step must not exceed the configured maximum");
        }
        if(retries_ >= policy_.maximum_retries)
        {
            return step_retry_result::retry_limit;
        }

        const T minimum_step = effective_minimum_step();
        if(requested_step < minimum_step)
        {
            return step_retry_result::minimum_step;
        }

        step_ = requested_step;
        ++retries_;
        step_retried_ = true;
        consecutive_first_attempt_successes_ = 0;
        return step_retry_result::retry;
    }

    step_retry_result reduce_next_step(const T factor)
    {
        if(!common::scalar_math::isfinite(factor) || factor <= T(0) || factor >= T(1))
        {
            throw std::invalid_argument("continuation step reduction factor must be finite and in (0,1)");
        }

        const T minimum_step = effective_minimum_step();
        if(step_ <= minimum_step)
        {
            return step_retry_result::minimum_step;
        }

        step_ = std::max(step_*factor, minimum_step);
        consecutive_first_attempt_successes_ = 0;
        step_retried_ = true;
        return step_retry_result::retry;
    }

    void accept_step(const bool recovered = false)
    {
        if(recovered || step_retried_)
        {
            consecutive_first_attempt_successes_ = 0;
            return;
        }

        ++consecutive_first_attempt_successes_;
        if(consecutive_first_attempt_successes_ >= policy_.successes_before_growth)
        {
            step_ = std::min(step_*policy_.success_growth_factor, maximum_step_);
            consecutive_first_attempt_successes_ = 0;
        }
    }

    T step() const { return step_; }
    T initial_step() const { return initial_step_; }
    T maximum_step() const { return maximum_step_; }
    T minimum_step() const { return effective_minimum_step(); }
    unsigned int retries() const { return retries_; }
    const corrector_retry_policy<T>& policy() const { return policy_; }

private:
    T effective_minimum_step() const
    {
        return std::max(policy_.minimum_step_size, initial_step_*policy_.minimum_step_ratio);
    }

    T initial_step_ = T(0);
    T maximum_step_ = T(0);
    T step_ = T(0);
    corrector_retry_policy<T> policy_;
    unsigned int retries_ = 0;
    unsigned int consecutive_first_attempt_successes_ = 0;
    bool step_retried_ = false;
};

} // namespace continuation

#endif
