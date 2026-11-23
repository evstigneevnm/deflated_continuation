#include "section_test_support.h"
#include <periodic_orbit/orbit_collection.h>

namespace
{
using namespace section_tests;
using periodic_orbit::crossing_preparation;
using periodic_orbit::orbit_relation;
template <class Model>
using flight = periodic_orbit::section_map<operations, step<Model>, plane, policy, rate_provider<Model>>;
template <class Model>
using comparison = periodic_orbit::orbit_comparison<operations, flight<Model>>;

void test_circles()
{
    auto space = std::make_shared<operations>( 2 );
    auto state = buffer( space );
    const auto period = 2 * pi;
    auto rate = std::make_shared<rate_provider<harmonic>>();
    auto pol = std::make_shared<policy>( space );
    auto rk = make_step<harmonic>( space );
    flight<harmonic> map( space, rk, pol, {}, rate );
    flight<harmonic>::request_type request;
    request.target = make_plane( space, 0, 0, 0, 1 );
    request.rule.minimum_crossing_speed = .01;
    comparison<harmonic>::params p{ 1e-6, 1e-7, 1e-7, 1e-5 };
    comparison<harmonic> cmp( space, map, 1., request, p );
    auto a = cmp.make_crossing(), b = cmp.make_crossing();
    require( cmp.compare( a, b ).relation == orbit_relation::inconclusive, "Unprepared slots compared" );

    // Take physical anchors from different real shooting layouts of the same
    // analytical orbit. The first starts on the comparison plane, the second off it.
    periodic_orbit::automatic_layout<operations, step<harmonic>, rate_provider<harmonic>> builder(
        space, rk, rate, pol, std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( space ) );
    for ( unsigned m : { 1u, 4u } )
    {
        fill( *space, *state, 1, 0 );
        periodic_orbit::placement_params<double> placement;
        placement.sections = m;
        const auto pilot = builder.build( *state, { 0., period, nullptr }, placement );
        require( pilot.succeeded(), "Cannot build analytical circle layout" );
        const auto &anchor = pilot.layout->at( m == 1 ? 0 : 1 ).section->center();
        const auto allocations = guarded_memory::allocation_calls;
        const auto r = cmp.prepare( anchor, 1., period, m == 1 ? a : b );
        require( r.status == crossing_preparation::ready && *r.closure_error < p.closure_tolerance,
                 "A transverse single return did not close" );
        require( guarded_memory::allocation_calls == allocations, "Preparing a crossing allocated a backend vector" );
    }
    auto allocations = guarded_memory::allocation_calls;
    for ( int i = 0; i < 10; ++i )
    {
        const auto ab = cmp.compare( a, b ), ba = cmp.compare( b, a );
        require( ab.relation == orbit_relation::same && ba.relation == ab.relation && ab.distance == ba.distance,
                 "Phase-shifted M=1/M=4 circles did not match symmetrically" );
    }
    require( allocations == guarded_memory::allocation_calls, "Cached comparisons allocated backend vectors" );
    fill( *space, *state, 0, 1.1 );
    require( cmp.prepare( *state, 1., period, b ).status == crossing_preparation::ready, "Second circle failed" );
    const auto distinct = cmp.compare( a, b );
    require( distinct.relation == orbit_relation::distinct && std::abs( *distinct.distance - .1 ) < 1e-7 &&
             cmp.compare( b, a ).relation == distinct.relation, "Equal-period concentric orbits were merged" );
    fill( *space, *state, 0, 1.000002 );
    require( cmp.prepare( *state, 1., period, b ).status == crossing_preparation::ready &&
             cmp.compare( a, b ).relation == orbit_relation::inconclusive, "Ignored uncertainty band" );
    require( !cmp.compare( a, b, []( const X &, const X & ){ return NAN; } ).distance, "Accepted an invalid metric" );
    require( cmp.prepare( *state, 1., 2 * period, b ).status == crossing_preparation::wrong_return_time &&
             cmp.compare( a, b ).relation == orbit_relation::inconclusive, "Accepted a doubled traversal or stale hit" );
    require( cmp.prepare( *state, 2., period, b ).status == crossing_preparation::invalid_orbit,
             "Accepted a different parameter" );
    fill( *space, *state, 0, .001 );
    require( cmp.prepare( *state, 1., period, b ).first.status == status::grazing, "Ignored a slow crossing" );
    fill( *space, *state, 0, 0 );
    require( cmp.prepare( *state, 1., period, b ).first.status == status::flight_limit &&
             cmp.compare( a, b ).relation == orbit_relation::inconclusive, "Missing crossing classified" );

    flight<harmonic>::params cap;
    cap.maximum_steps = 2;
    flight<harmonic> capped_map( space, rk, pol, cap, rate );
    comparison<harmonic> capped( space, capped_map, 1., request, p );
    auto c = capped.make_crossing();
    fill( *space, *state, 1, 0 );
    require( capped.prepare( *state, 1., period, c ).status == crossing_preparation::flight_failed &&
             capped.compare( c, c ).relation == orbit_relation::inconclusive, "Budget exhaustion classified" );
    require( capped.compare( a, a ).relation == orbit_relation::inconclusive, "Mixed comparison domains" );

    // Each supplied period can agree with its measured return within tolerance
    // while the two estimates disagree with each other.
    const double offset = .75 * p.period_tolerance;
    fill( *space, *state, 0, 1 );
    require( cmp.prepare( *state, 1., period - offset, a ).status == crossing_preparation::ready &&
             cmp.prepare( *state, 1., period + offset, b ).status == crossing_preparation::ready,
             "Could not prepare two individually admissible period estimates" );
    const auto near = cmp.compare( a, b );
    require( near.distance && *near.distance <= p.same_distance && near.relation == orbit_relation::inconclusive,
             "Matching crossings with incompatible periods were classified" );
    fill( *space, *state, 0, 1.000002 );
    require( cmp.prepare( *state, 1., period + offset, b ).status == crossing_preparation::ready,
             "Cannot prepare nearby circle" );
    const auto gap = cmp.compare( a, b );
    require( gap.distance && *gap.distance > p.same_distance && *gap.distance < p.distinct_distance &&
             gap.relation == orbit_relation::inconclusive, "Period mismatch overrode the uncertainty band" );

    // An inconclusive earlier entry must not hide a later definite duplicate.
    comparison<harmonic> ccmp( space, map, 1., request, { 1e-6, 1e-7, .01, .1 } );
    periodic_orbit::orbit_collection catalog( ccmp );
    auto admit = [&]( double radius ) {
        fill( *space, *state, 0., radius );
        return catalog.admit_verified( *state, 1., period );
    };
    using periodic_orbit::orbit_admission;
    require( admit( 1. ).decision == orbit_admission::inserted &&
             admit( 1.105 ).decision == orbit_admission::inserted, "Two resolved circles were not admitted" );
    const auto before = guarded_memory::allocation_calls;
    auto uncertain = admit( 1.05 ), duplicate = admit( 1.098 );
    require( uncertain.decision == orbit_admission::inconclusive && uncertain.index == 0 &&
             duplicate.decision == orbit_admission::duplicate && duplicate.index == 1 &&
             catalog.size() == 2, "Incorrect ambiguity/duplicate precedence" );
    require( guarded_memory::allocation_calls == before, "Rejected admission allocated backend vectors" );
}

// Two oscillators with frequencies 2 and 1. The oriented plane y1+y2=0
// has two increasing hits on a single primitive orbit, at 0 and pi.
struct two_frequencies
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
    {
        return ( i < 2 ? 2 : 1 ) * harmonic{}.rhs( i, u, t );
    }
};

void test_multiple_hits()
{
    auto space = std::make_shared<operations>( 4 );
    auto u = buffer( space );
    fill( *space, *u, 1, 0 );
    flight<two_frequencies> map( space, make_step<two_frequencies>( space, .03 ), std::make_shared<policy>( space ),
                                 {}, std::make_shared<rate_provider<two_frequencies>>() );
    flight<two_frequencies>::request_type req;
    req.target = make_plane( space, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .1;
    comparison<two_frequencies> cmp( space, map, 1., req, { 1e-6, 1e-7, 1e-7, 1e-5 } );
    auto slot = cmp.make_crossing();
    const auto r = cmp.prepare( *u, 1., 2*pi, slot );
    require( r.status == crossing_preparation::wrong_return_time && std::abs( r.next.flight_time - pi ) < 1e-6 &&
             cmp.compare( slot, slot ).relation == orbit_relation::inconclusive, "Multi-hit section classified" );
    // Both oscillators hit the plane after pi, but the second has not closed.
    require( cmp.prepare( *u, 1., pi, slot ).status == crossing_preparation::not_closed,
             "Return time alone was accepted as closure" );
}

void test_backend_cache()
{
    auto space = std::make_shared<operations>( 8192 );
    auto u = buffer( space );
    flight<harmonic> map( space, make_step<harmonic>( space, .1 ), std::make_shared<policy>( space ),
                          {}, std::make_shared<rate_provider<harmonic>>() );
    flight<harmonic>::request_type req;
    req.target = make_plane( space, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    comparison<harmonic> cmp( space, map, 1., req, { 1e-5, 1e-4, 1e-4, 1e-3 } );
    auto a = cmp.make_crossing(), b = cmp.make_crossing();
    const auto allocations = guarded_memory::allocation_calls;
    fill( *space, *u, 0, 1 );
    require( cmp.prepare( *u, 1., 2*pi, a ).status == crossing_preparation::ready, "Large orbit preparation failed" );
    fill( *space, *u, 0, -1 );
    require( cmp.prepare( *u, 1., 2*pi, b ).status == crossing_preparation::ready, "Large shifted orbit failed" );
    for ( int i = 0; i < 5; ++i )
        require( cmp.compare( a, b ).relation == orbit_relation::same, "Large cached states did not match" );
    require( allocations == guarded_memory::allocation_calls, "Large repeated operations allocated backend vectors" );
}
}

int main()
{
    return section_tests::run( [] {
        const auto bytes = guarded_memory::host_bytes;
        test_circles();
        test_multiple_hits();
        test_backend_cache();
        require( guarded_memory::host_bytes == bytes, "Orbit comparison transferred physical state data" );
    } );
}
