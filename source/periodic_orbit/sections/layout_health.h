#ifndef PERIODIC_ORBIT_LAYOUT_HEALTH_H
#define PERIODIC_ORBIT_LAYOUT_HEALTH_H

#include <cmath>
#include <stdexcept>

namespace periodic_orbit
{
enum class layout_condition
{
    healthy,
    grazing,
    crowded,
    slow_flow,
    repeated_circuit
};
inline const char *name( layout_condition why )
{
    switch ( why )
    {
    case layout_condition::healthy:
        return "healthy";
    case layout_condition::grazing:
        return "near-tangent crossing";
    case layout_condition::crowded:
        return "short inter-section flight";
    case layout_condition::slow_flow:
        return "insufficient flow speed";
    case layout_condition::repeated_circuit:
        return "repeated circuit";
    }
    return "unknown";
}

// For automatic, oriented, unit-normal sections in the state-space metric.
// The baseline is the current pilot, not the previous step: slow drift matters.
template <class T>
struct layout_health
{
    T minimum_cosine = T( .25 ), minimum_flight_fraction = T( .3 ), minimum_speed = T( 1e-10 );

    template <class Layout, class Report, class Closes>
    layout_condition
    inspect( const Layout &layout, T pilot_period, T previous_period, const Report &r, Closes closes ) const
    {
        if ( !r.succeeded() || !( r.period > 0 ) || !( pilot_period > 0 ) || !( previous_period > 0 ) ||
             r.segments.size() != layout.size() )
            throw std::invalid_argument( "Missing section diagnostics" );
        // Only suspicious period growth pays for an independent shorter return.
        if ( r.period > T( 1.5 ) * previous_period )
        {
            const T laps = std::round( r.period / previous_period );
            if ( laps >= 2 && closes( r.period / laps ) )
                return layout_condition::repeated_circuit;
        }
        for ( std::size_t i = 0; i < layout.size(); ++i )
        {
            const auto &flight = r.segments[i].flight;
            if ( !std::isfinite( flight.rate_norm ) || !( flight.rate_norm > minimum_speed ) )
                return layout_condition::slow_flow;
            if ( flight.crossing_speed < minimum_cosine * flight.rate_norm )
                return layout_condition::grazing;
            if ( flight.flight_time / r.period <
                 minimum_flight_fraction * layout.at( i ).expected_flight / pilot_period )
                return layout_condition::crowded;
        }
        return layout_condition::healthy;
    }
};
}
#endif
