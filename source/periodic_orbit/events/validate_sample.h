#ifndef PERIODIC_ORBIT_EVENTS_VALIDATE_SAMPLE_H
#define PERIODIC_ORBIT_EVENTS_VALIDATE_SAMPLE_H

#include <optional>
#include <type_traits>
#include <periodic_orbit/events/event_types.h>

namespace periodic_orbit::detail
{
template <class Space, class Policy>
std::optional<section_status> validate_sample( Space &space, Policy &policy, typename Space::norm_type time,
                                               typename Space::vector_type &state )
{
    if ( !scfd::utils::scalar_traits<typename Space::norm_type>::is_finite( time ) || !space.is_valid_number( state ) )
        return section_status::invalid_state;

    // Only the built-in identity reconstruction is known to leave the checked state untouched.
    if constexpr ( !std::is_same_v<std::remove_cv_t<Policy>, identity_state_policy<Space>> )
    {
        if ( !policy.reconstruct( time, state ) )
            return section_status::reconstruction_failure;
        if ( !space.is_valid_number( state ) )
            return section_status::invalid_state;
    }
    return std::nullopt;
}
}
#endif
