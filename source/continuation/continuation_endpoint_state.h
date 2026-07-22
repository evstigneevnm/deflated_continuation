#ifndef __CONTINUATION_ENDPOINT_STATE_H__
#define __CONTINUATION_ENDPOINT_STATE_H__

#include <containers/curve_endpoint_reason.h>

namespace continuation
{

inline unsigned int endpoint_reason_priority(
    const container::curve_endpoint_reason reason)
{
    using reason_t = container::curve_endpoint_reason;
    switch(reason)
    {
    case reason_t::none:
        return 0;
    case reason_t::boundary_min:
    case reason_t::boundary_max:
    case reason_t::closed_return:
    case reason_t::self_intersection:
        return 10;
    case reason_t::known_branch:
    case reason_t::analytical_branch:
    case reason_t::symmetry_intersection:
        return 20;
    case reason_t::max_steps:
        return 30;
    case reason_t::no_progress:
    case reason_t::knot_interpolation_failure:
        return 40;
    case reason_t::unresolved_branch_intersection:
        return 50;
    case reason_t::hard_failure:
        return 60;
    }
    return 0;
}

inline container::curve_endpoint_reason resolve_endpoint_reason(
    const container::curve_endpoint_reason current,
    const container::curve_endpoint_reason candidate)
{
    return endpoint_reason_priority(candidate) >= endpoint_reason_priority(current)
        ? candidate
        : current;
}

class continuation_endpoint_state
{
public:
    using reason_type = container::curve_endpoint_reason;

    void reset_curve()
    {
        pending_reason_ = reason_type::none;
        incomplete_ = false;
    }

    void reset_step()
    {
        pending_reason_ = reason_type::none;
    }

    void observe(const reason_type reason)
    {
        pending_reason_ = resolve_endpoint_reason(pending_reason_, reason);
        incomplete_ = incomplete_ || container::is_incomplete_endpoint(reason);
    }

    void replace_pending(const reason_type reason)
    {
        pending_reason_ = reason;
    }

    void mark_incomplete()
    {
        incomplete_ = true;
    }

    reason_type pending_reason() const
    {
        return pending_reason_;
    }

    bool incomplete() const
    {
        return incomplete_;
    }

private:
    reason_type pending_reason_ = reason_type::none;
    bool incomplete_ = false;
};

} // namespace continuation

#endif // __CONTINUATION_ENDPOINT_STATE_H__
