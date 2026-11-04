#include "section_test_support.h"
using namespace section_tests;

struct controlled_policy : policy
{
    using policy::policy;
    bool fail = false, throws = false;
    bool reconstruct( double time, X &state )
    {
        if ( throws )
            throw std::runtime_error( "Injected reconstruction failure" );
        return !fail && policy::reconstruct( time, state );
    }
};

struct selective_rate : rate_provider<harmonic>
{
    bool evaluate( double time, const X &in, X &out )
    {
        return std::abs( time - 3.2 ) > .01 && rate_provider<harmonic>::evaluate( time, in, out );
    }
};

void test()
{
    auto ops = std::make_shared<operations>( 2 );
    auto u = buffer( ops ), out = buffer( ops );
    fill( *ops, *u, 1, 0 );
    fill( *ops, *out, 7, 8 );
    auto s    = make_step<harmonic>( ops, .2 );
    auto p    = std::make_shared<controlled_policy>( ops );
    auto rate = std::make_shared<rate_provider<harmonic>>();
    using map =
        periodic_orbit::section_map<operations, step<harmonic>, plane, controlled_policy, rate_provider<harmonic>>;
    map                                           flight( ops, s, p, {}, rate );
    periodic_orbit::flight_request<double, plane> request;
    request.end_time       = .5;
    request.target         = make_plane( ops, 0, 0, 1, 0 );
    request.rule.direction = periodic_orbit::crossing_direction::decreasing;
    require( flight.evaluate( *u, request, *out ).status == status::flight_limit, "Missing flight limit" );
    near_state( ops, *out, 7, 8, 1e-15 );
    request.end_time = 3;
    p->fail          = true;
    require(
        flight.evaluate( *u, request, *out ).status == status::reconstruction_failure, "Reconstruction failure lost"
    );
    near_state( ops, *out, 7, 8, 1e-15 );
    require( s->failed == 1, "Failed pending step was committed" );
    p->fail   = false;
    p->throws = true;
    require_throws<std::runtime_error>( [&] { flight.evaluate( *u, request, *out ); }, "Policy exception swallowed" );
    near_state( ops, *out, 7, 8, 1e-15 );
    p->throws     = false;
    rate->failure = true;
    require( flight.evaluate( *u, request, *out ).status == status::rate_failure, "Rate failure lost" );
    near_state( ops, *out, 7, 8, 1e-15 );
    rate->failure                       = false;
    request.rule.minimum_crossing_speed = 2;
    require( flight.evaluate( *u, request, *out ).status == status::grazing, "Transversality threshold ignored" );
    near_state( ops, *out, 7, 8, 1e-15 );
    request.rule.minimum_crossing_speed = 0;
    require( flight.evaluate( *u, request, *out ).succeeded(), "Failure contaminated repeated flight" );
    near_state( ops, *out, 0, 1, 1e-7 );

    request.target                      = make_plane( ops, .999999, 0, 1, 0 );
    request.rule.minimum_crossing_speed = .01;
    require(
        flight.evaluate( *u, request, *out ).status == status::grazing, "Near-grazing crossing was not diagnosed"
    );
    request.target                      = make_plane( ops, 0, 0, 1, 0 );
    request.rule.minimum_crossing_speed = 0;
    auto wrong_space                    = std::make_shared<operations>( 3 );
    auto wrong                          = buffer( wrong_space );
    wrong_space->assign_scalar( 0, *wrong );
    require(
        flight.evaluate( *wrong, request, *out ).status == status::invalid_input, "Wrong state dimension accepted"
    );
    auto invalid_problem     = std::make_shared<problem<harmonic>>( problem<harmonic>{ ops, {} } );
    invalid_problem->invalid = true;
    auto invalid_step =
        std::make_shared<step<harmonic>>( ops, invalid_problem, std::make_shared<fixed>( fixed::params{ .1 } ) );
    periodic_orbit::section_map<operations, step<harmonic>, plane> failed_step(
        ops, invalid_step, std::make_shared<policy>( ops )
    );
    fill( *ops, *out, 7, 8 );
    require( failed_step.evaluate( *u, request, *out ).status == status::step_failure, "Evolution failure lost" );
    near_state( ops, *out, 7, 8, 1e-15 );

    map::params limits;
    limits.maximum_steps = 1;
    map bounded( ops, s, p, limits, rate );
    require( bounded.evaluate( *u, request, *u ).status == status::flight_limit, "Step budget ignored" );
    near_state( ops, *u, 1, 0, 1e-15 );
    limits.maximum_steps            = 100;
    limits.event.maximum_iterations = 1;
    map no_refinement( ops, s, p, limits, rate );
    require( no_refinement.evaluate( *u, request, *out ).status == status::refinement_limit, "Refinement not bounded" );
    request.end_time = 0;
    require_throws<std::invalid_argument>( [&] { flight.evaluate( *u, request, *out ); }, "Empty flight accepted" );

    // Layout failure never replaces a previously published itinerary.
    auto metric = std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( ops );
    periodic_orbit::automatic_layout<operations, step<harmonic>, rate_provider<harmonic>, controlled_policy> builder(
        ops, s, rate, p, metric
    );
    periodic_orbit::placement_params<double> placement;
    auto                                     good = builder.build( *u, { 0, 2 * pi, nullptr }, placement );
    require( good.succeeded(), "Harmonic pilot layout failed" );
    auto saved = good.layout;
    auto bad   = builder.build( *u, { 0, 1, nullptr }, placement );
    require( !bad.succeeded() && saved->size() == 1, "Nonrecurring pilot published a layout" );
    require(
        !builder.build( *u, { 0, 4 * pi, nullptr }, placement ).succeeded(),
        "Wrong cycle count accepted as pilot itinerary"
    );
    require( builder.build( *u, { 0, -2 * pi, nullptr }, placement ).succeeded(), "Backward pilot itinerary failed" );
    auto selective = std::make_shared<selective_rate>();
    periodic_orbit::automatic_layout<operations, step<harmonic>, selective_rate, controlled_policy> neighboring(
        ops, s, selective, p, metric
    );
    placement.sections = 2;
    auto adjusted      = neighboring.build( *u, { 0, 2 * pi, nullptr }, placement );
    require(
        adjusted.succeeded() && adjusted.layout->at( 1 ).pilot_step != 16,
        "Neighbor placement did not avoid failed rate"
    );
    placement.neighbor_steps = 0;
    require(
        !neighboring.build( *u, { 0, 2 * pi, nullptr }, placement ).succeeded(), "Failed quantile rate was ignored"
    );
    placement.sections = 1000;
    require(
        builder.build( *u, { 0, 2 * pi, nullptr }, placement ).status == status::insufficient_steps,
        "Infeasible section count accepted"
    );

    std::weak_ptr<operations>     weak_space;
    std::weak_ptr<step<harmonic>> weak_step;
    std::weak_ptr<const plane>    weak_target;
    {
        auto owned  = std::make_shared<operations>( 2 );
        auto single = make_step<harmonic>( owned );
        auto target = make_plane( owned, 0, 0, 1, 0 );
        weak_space  = owned;
        weak_step   = single;
        weak_target = target;
        periodic_orbit::section_map<operations, step<harmonic>, plane> temporary(
            owned, single, std::make_shared<policy>( owned )
        );
        auto x = buffer( owned ), y = buffer( owned );
        fill( *owned, *x, 1, 0 );
        periodic_orbit::flight_request<double, plane> journey;
        journey.end_time       = 2;
        journey.target         = target;
        journey.rule.direction = periodic_orbit::crossing_direction::decreasing;
        require( temporary.evaluate( *x, journey, *y ).succeeded(), "Owned flight failed" );
    }
    require( weak_space.expired() && weak_step.expired() && weak_target.expired(), "Section ownership cycle" );
}
int main()
{
    return run( test );
}
