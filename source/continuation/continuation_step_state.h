#ifndef __CONTINUATION_CONTINUATION_STEP_STATE_H__
#define __CONTINUATION_CONTINUATION_STEP_STATE_H__

#include <string>

#include <continuation/continuation_result.h>
#include <symmetry/continuation/isotropy_transition.h>

namespace continuation
{

template<class T>
struct continuation_step_attempt_state
{
    bool any_corrector_retries = false;
    bool any_chart_retries = false;
    bool any_isotropy_retries = false;
    unsigned int chart_retries = 0;
    continuation_failure_kind failure = continuation_failure_kind::none;
    T attempted_step = T(0);
    unsigned int retry_count = 0;
    std::string failure_reason;

    bool recovered() const
    {
        return any_corrector_retries || any_chart_retries || any_isotropy_retries;
    }
};

template<class T>
class isotropy_refinement_bracket
{
public:
    bool active() const { return active_; }
    unsigned int refinements() const { return refinements_; }
    T event_lambda() const { return event_lambda_; }

    bool latch_event(const T step, const T lambda)
    {
        const bool replace = !active_ || step < event_step_;
        if(replace)
        {
            event_step_ = step;
            event_lambda_ = lambda;
        }
        if(!active_)
        {
            non_event_step_ = T(0);
            active_ = true;
        }
        return replace;
    }

    void observe_non_event(const T step)
    {
        if(active_ && step > non_event_step_ && step < event_step_)
        {
            non_event_step_ = step;
        }
    }

    bool next_step(
        const symmetry::continuation::isotropy_transition_policy<T>& policy,
        T& step)
    {
        if(!active_ || refinements_ >= policy.maximum_refinements)
        {
            return false;
        }
        const T bracket_width = event_step_ - non_event_step_;
        const T candidate =
            non_event_step_ + policy.refinement_step_factor*bracket_width;
        if(!(candidate > non_event_step_ && candidate < event_step_))
        {
            return false;
        }
        step = candidate;
        ++refinements_;
        return true;
    }

    T non_event_step() const { return non_event_step_; }
    T event_step() const { return event_step_; }

private:
    bool active_ = false;
    unsigned int refinements_ = 0;
    T non_event_step_ = T(0);
    T event_step_ = T(0);
    T event_lambda_ = T(0);
};

} // namespace continuation

#endif
