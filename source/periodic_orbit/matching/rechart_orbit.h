#ifndef PERIODIC_ORBIT_RECHART_ORBIT_H
#define PERIODIC_ORBIT_RECHART_ORBIT_H

#include <periodic_orbit/sections/section_layout.h>

namespace periodic_orbit
{
// Express the preceding accepted orbit in the new charts, for an oriented secant.
// Replaying consecutive flights costs at most a bounded number of circuits;
// neither a full trajectory nor a host copy of a section state is needed.
template <class Map, class Layout, class Charts, class X, class Z>
bool rechart_orbit(
    Map &map, const Layout &layout, const Charts &charts, X &physical, typename Map::T previous_period, Z &coordinates
)
{
    using T = typename Map::T;
    T t     = layout.at( 0 ).pilot_time;
    for ( std::size_t i = 0; i < layout.size(); ++i )
    {
        typename Map::request_type r;
        r.start_time = t;
        r.end_time   = t + ( i ? 2 * layout.at( i - 1 ).expected_flight : T( 1.25 ) * previous_period );
        r.target     = layout.at( i ).section;
        r.rule.minimum_crossing_speed = T( 1e-10 );
        const auto hit                = map.evaluate( physical, r, physical );
        if ( !hit.succeeded() )
            return false;
        charts[i]->restrict( physical, coordinates.comp( i ) );
        t = hit.time;
    }
    return true;
}
}
#endif
