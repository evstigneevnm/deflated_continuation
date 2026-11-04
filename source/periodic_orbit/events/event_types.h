#ifndef PERIODIC_ORBIT_EVENT_TYPES_H
#define PERIODIC_ORBIT_EVENT_TYPES_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <scfd/utils/scalar_traits.h>

namespace periodic_orbit
{
enum class section_status
{
    idle,
    not_found,
    located,
    initial_event,
    invalid_input,
    invalid_state,
    nonfinite_event,
    reconstruction_failure,
    rate_failure,
    grazing,
    refinement_limit,
    flight_limit,
    step_failure,
    insufficient_steps,
    placement_failure
};

enum class crossing_direction
{
    increasing,
    decreasing,
    either
};
enum class initial_event_policy
{
    leave_and_return,
    report_initial
};

template <class T>
struct event_params
{
    T           value_tolerance         = T( 1e-10 );
    T           time_tolerance          = T( 1e-12 );
    T           relative_time_tolerance = T( 0 );
    std::size_t maximum_iterations      = 80;
    std::size_t subdivisions            = 4;

    void validate() const
    {
        using traits = scfd::utils::scalar_traits<T>;
        if ( !traits::is_finite( value_tolerance ) || value_tolerance <= T( 0 ) ||
             !traits::is_finite( time_tolerance ) || time_tolerance <= T( 0 ) ||
             !traits::is_finite( relative_time_tolerance ) || relative_time_tolerance < T( 0 ) || !maximum_iterations ||
             !subdivisions )
            throw std::invalid_argument( "Invalid event refinement parameters" );
    }
};

template <class T>
struct crossing_rule
{
    crossing_direction   direction              = crossing_direction::increasing;
    initial_event_policy initial                = initial_event_policy::leave_and_return;
    T                    arming_band            = T( 1e-8 );
    T                    minimum_flight_time    = T( 0 );
    T                    minimum_crossing_speed = T( 0 );

    void validate( T value_tolerance ) const
    {
        using traits = scfd::utils::scalar_traits<T>;
        if ( !traits::is_finite( arming_band ) || arming_band < value_tolerance ||
             !traits::is_finite( minimum_flight_time ) || minimum_flight_time < T( 0 ) ||
             !traits::is_finite( minimum_crossing_speed ) || minimum_crossing_speed < T( 0 ) )
            throw std::invalid_argument( "Invalid section crossing rule" );
    }
};

template <class T>
struct event_bracket
{
    T step_start{}, step_size{}, lower{}, upper{}, lower_value{}, upper_value{};
};

template <class T>
struct event_report
{
    section_status status = section_status::idle;
    T              time{}, value{}, crossing_speed{};
    std::size_t    iterations = 0;
    bool           succeeded() const
    {
        return status == section_status::located || status == section_status::initial_event;
    }
};

template <class T>
struct map_report : event_report<T>
{
    T           flight_time{};
    std::size_t steps = 0;
};

template <class T, class Event>
struct flight_request
{
    T                            start_time{}, end_time{};
    std::shared_ptr<const Event> target;
    crossing_rule<T>             rule;
};

template <class Space>
class identity_state_policy
{
    std::shared_ptr<Space> space_;

public:
    using T = typename Space::norm_type;
    using X = typename Space::vector_type;
    explicit identity_state_policy( std::shared_ptr<Space> space ) : space_( std::move( space ) )
    {
        if ( !space_ )
            throw std::invalid_argument( "Missing state-policy space" );
    }
    bool validate_initial( T, const X &x ) const
    {
        return space_->is_valid_number( x );
    }
    bool reconstruct( T, X &x ) const
    {
        return space_->is_valid_number( x );
    }
};

struct no_section_rate
{
};
}
#endif
