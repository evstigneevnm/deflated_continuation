#ifndef PERIODIC_ORBIT_RECHART_ORBIT_H
#define PERIODIC_ORBIT_RECHART_ORBIT_H

#include <periodic_orbit/sections/section_layout.h>

namespace periodic_orbit
{
// The caller has a verified crossing of this layout's oriented first section.
template <class Map, class Layout, class Charts, class X, class Z>
bool rechart_from_first_crossing(
    Map &map, const Layout &layout, const Charts &charts, X &physical, typename Map::T t, Z &coordinates
)
{
    using T = typename Map::T;
    charts[0]->restrict( physical, coordinates.comp( 0 ) );
    for ( std::size_t i = 1; i < layout.size(); ++i )
    {
        typename Map::request_type r;
        r.start_time = t;
        r.end_time                    = t + 2 * layout.at( i - 1 ).expected_flight;
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

// An arbitrary phase or a different layout requires locating the first crossing.
template <class Map, class Layout, class Charts, class X, class Z>
bool rechart_orbit(
    Map &map, const Layout &layout, const Charts &charts, X &physical, typename Map::T previous_period, Z &coordinates
)
{
    using T = typename Map::T;
    typename Map::request_type r;
    r.start_time                  = layout.at( 0 ).pilot_time;
    r.end_time                    = r.start_time + T( 1.25 ) * previous_period;
    r.target                      = layout.at( 0 ).section;
    r.rule.minimum_crossing_speed = T( 1e-10 );
    const auto hit                = map.evaluate( physical, r, physical );
    return hit.succeeded() && rechart_from_first_crossing( map, layout, charts, physical, hit.time, coordinates );
}
}
#endif
