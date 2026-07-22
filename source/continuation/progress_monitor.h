#ifndef __CONTINUATION_PROGRESS_MONITOR_H__
#define __CONTINUATION_PROGRESS_MONITOR_H__

#include <deque>
#include <stdexcept>

#include <common/scalar_math.h>

namespace continuation
{

template<class T>
struct progress_monitor_policy
{
    bool enabled = false;
    unsigned int window_size = 25;
    T minimum_window_progress_ratio = T(1.0e-3);

    void validate() const
    {
        if(window_size == 0)
        {
            throw std::invalid_argument("continuation progress window_size must be positive");
        }
        if(!common::scalar_math::isfinite(minimum_window_progress_ratio) ||
           minimum_window_progress_ratio < T(0))
        {
            throw std::invalid_argument(
                "continuation minimum_window_progress_ratio must be finite and non-negative");
        }
    }
};

template<class T>
class progress_monitor
{
public:
    void configure(const progress_monitor_policy<T>& policy, const T reference_step)
    {
        policy.validate();
        if(!common::scalar_math::isfinite(reference_step) || reference_step <= T(0))
        {
            throw std::invalid_argument(
                "continuation progress reference step must be positive and finite");
        }
        policy_ = policy;
        reference_step_ = reference_step;
        reset();
    }

    void reset()
    {
        values_.clear();
        accumulated_progress_ = T(0);
    }

    bool observe(const T progress)
    {
        if(!policy_.enabled)
        {
            return false;
        }
        if(!common::scalar_math::isfinite(progress) || progress < T(0))
        {
            return true;
        }

        values_.push_back(progress);
        accumulated_progress_ += progress;
        if(values_.size() > policy_.window_size)
        {
            accumulated_progress_ -= values_.front();
            values_.pop_front();
        }
        return values_.size() == policy_.window_size &&
               accumulated_progress_ < reference_step_*policy_.minimum_window_progress_ratio;
    }

    T accumulated_progress() const { return accumulated_progress_; }
    std::size_t sample_count() const { return values_.size(); }

private:
    progress_monitor_policy<T> policy_;
    T reference_step_ = T(1);
    T accumulated_progress_ = T(0);
    std::deque<T> values_;
};

} // namespace continuation

#endif
