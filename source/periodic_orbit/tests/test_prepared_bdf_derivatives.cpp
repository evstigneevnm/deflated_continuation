#include "bdf_flight_test_support.h"
using namespace bdf_flights;

// Compare BDF's discrete history tangent to the same section map on the same
// mesh, including backward-Euler startup and the final dense-polynomial crossing.
void check(
    unsigned int order, system_kind kind = system_kind::cycle, std::size_t dimension = 2, bool backward = false
)
{
    model equation{ kind };
    auto  ops     = std::make_shared<operations>( dimension );
    auto  initial = buffer( ops ), arrival = buffer( ops ), v = buffer( ops ), w = buffer( ops );
    auto image = buffer( ops ), other = buffer( ops ), plus = buffer( ops ), minus = buffer( ops ), sum = buffer( ops );
    fill( *ops, *initial, 1, 0, 1, equation.components() );
    auto         providers        = std::make_shared<provider_factory>( provider_factory{ ops, equation } );
    const auto   allocated_before = guarded_memory::allocation_calls;
    factory_type factory( ops, providers, factory_type::params{ order } );
    require(
        providers->creations == 0 && guarded_memory::allocation_calls == allocated_before,
        "BDF factory allocated sensitivity workspace eagerly"
    );
    auto map = make_map( ops, equation, order, .025 );
    auto r   = request( ops, backward ? 2. : .4 );
    if ( kind == system_kind::linear_dae )
    {
        fill( *ops, *plus, 0, 0, 0, 3 );
        fill( *ops, *minus, 1, 0, 0, 3 );
        r.target = std::make_shared<plane>( ops, *plus, *minus );
    }
    if ( backward )
    {
        r.end_time       = 0;
        r.rule.direction = periodic_orbit::crossing_direction::increasing;
    }
    require( map->evaluate( *initial, r, *arrival ).succeeded(), "BDF base section flight failed" );
    require( providers->creations == 0, "Ordinary BDF flight constructed sensitivity provider" );
    require( map->records().front().order == 1, "BDF flight did not cold start at order one" );
    if ( order == 2 )
        require( map->records().at( 1 ).order == 2, "BDF2 did not promote after startup" );
    struct segment
    {
        std::shared_ptr<recorded_map> flight;
    };
    auto       workspace = factory.make_workspace( std::vector<segment>{ { map } } );
    auto       flight    = factory.prepare( workspace, *map, *initial, *arrival, r.target, .1, 1e-8 );
    const auto schedule  = flight.records();
    require( schedule.back().theta < 1, "BDF test did not exercise dense event truncation" );
    fill( *ops, *v, .3, -.4, -.1, equation.components() );
    fill( *ops, *w, -.2, .1, -.1, equation.components() );
    const auto allocated = guarded_memory::allocation_calls;
    flight.apply( *v, *image );
    const auto dt = flight.last_time_increment();
    flight.apply( *w, *other );
    ops->assign_lin_comb( 2, *v, -3, *w, *sum );
    flight.apply( *sum, *sum );
    ops->add_lin_comb( -2, *image, *sum );
    ops->add_lin_comb( 3, *other, *sum );
    const auto scale = std::sqrt( double( dimension / equation.components() ) );
    require( ops->norm( *sum ) / scale < 2e-8, "BDF history tangent is not linear" );
    ops->assign_lin_comb( 1e-12, *v, *sum );
    flight.apply( *sum, *sum );
    ops->scale( 1e12, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) / scale < 2e-8, "BDF Jv depends on direction scale" );
    require( guarded_memory::allocation_calls == allocated, "BDF Jv allocated backend vectors" );
    const double epsilon    = 2e-5;
    auto         check_mesh = [&] {
        require( map->records().size() == schedule.size(), "BDF finite difference changed step count" );
        for ( std::size_t i = 0; i < schedule.size(); ++i )
            require(
                map->records()[i].time == schedule[i].time && map->records()[i].dt == schedule[i].dt &&
                    map->records()[i].order == schedule[i].order,
                "BDF finite difference changed frozen history"
            );
    };
    ops->assign_lin_comb( 1, *initial, epsilon, *v, *plus );
    const auto positive = map->evaluate( *plus, r, *plus );
    require( positive.succeeded(), "Positive BDF finite difference failed" );
    check_mesh();
    ops->assign_lin_comb( 1, *initial, -epsilon, *v, *minus );
    const auto negative = map->evaluate( *minus, r, *minus );
    require( negative.succeeded(), "Negative BDF finite difference failed" );
    check_mesh();
    ops->assign_lin_comb( .5 / epsilon, *plus, -.5 / epsilon, *minus, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) / scale < 2e-5, "BDF section Jv finite difference mismatch" );
    require(
        std::abs( ( positive.flight_time - negative.flight_time ) / ( 2 * epsilon ) - dt ) < 2e-5,
        "BDF section-time derivative finite difference mismatch"
    );
    flight.apply( *v, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) / scale < 1e-12, "Trial residual corrupted prepared BDF history" );
    if ( kind == system_kind::cycle || kind == system_kind::coupled_mass || kind == system_kind::state_mass )
    {
        fill( *ops, *v, 1, 0, 1, equation.components() );
        flight.apply( *v, *sum );
        fill( *ops, *other, 0, std::exp( -pi ) );
        ops->add_lin_comb( -1, *other, *sum );
        require( ops->norm( *sum ) / scale < .04, "BDF analytic radial derivative" );
    }
    providers->flags->refuse_rhs = true;
    ops->assign_scalar( 7, *sum );
    const auto saved_norm = ops->norm( *sum ), saved_dt = flight.last_time_increment();
    require_throws<std::runtime_error>( [&] { flight.apply( *v, *sum ); }, "BDF tangent failure ignored" );
    require(
        ops->norm( *sum ) == saved_norm && flight.last_time_increment() == saved_dt,
        "Failed BDF Jv published partial results"
    );
    providers->flags->refuse_rhs = false;
    using replay                 = typename factory_type::flow_type::replay_type;
    auto       provider          = providers->make_provider();
    const auto live              = guarded_memory::live_allocations;
    replay     bounded( ops, provider, { order }, 1 );
    require(
        guarded_memory::live_allocations - live == ( order == 1 ? 9u : 11u ),
        "BDF replay workspace grows with trajectory length"
    );
    require_throws<std::invalid_argument>(
        [&] { bounded.run( { order }, schedule, *initial, nullptr, *sum, nullptr, nullptr ); },
        "BDF replay ignored step budget"
    );
    auto invalid          = schedule;
    invalid.front().order = 2;
    require_throws<std::invalid_argument>(
        [&] { workspace->run( { order }, invalid, *initial, nullptr, *sum, nullptr, nullptr ); },
        "BDF replay accepted missing startup history"
    );
    invalid              = schedule;
    invalid.back().order = 3;
    require_throws<std::invalid_argument>(
        [&] { workspace->run( { order }, invalid, *initial, nullptr, *sum, nullptr, nullptr ); },
        "BDF replay silently accepted a higher order"
    );
    factory_type limited( ops, providers, factory_type::params{ order, 1 } );
    const auto   creations = providers->creations;
    require_throws<std::invalid_argument>(
        [&] { limited.make_workspace( std::vector<segment>{ { map } } ); }, "BDF factory ignored step budget"
    );
    require( providers->creations == creations, "Invalid BDF schedule constructed a provider" );
}

void check_commits_and_restarts()
{
    auto ops     = std::make_shared<operations>( 2 );
    auto initial = buffer( ops ), candidate = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    auto                                                        native = make_step( ops, {}, 2, .04 );
    periodic_orbit::recording_stepper<native_step, record_type> recorder( native, 20 );
    recorder.set_time( .5 );
    recorder.set_target_time( 1 );
    recorder.apply( *initial, *candidate );
    require_throws<std::invalid_argument>(
        [&] { recorder.finalize( ts::adaptation_status::accepted, .4, *candidate ); }, "Invalid BDF commit accepted"
    );
    require( recorder.records().empty(), "Failed BDF commit advanced recording" );
    recorder.finalize( ts::adaptation_status::accepted, .54, *candidate );
    require( recorder.records().size() == 1, "Valid BDF commit not recorded" );
    recorder.reset();
    require( recorder.records().empty(), "BDF recorder reset retained trajectory" );

    // Target clipping changes actual steps; native fixed BDF2 must restart at
    // BDF1, then promote again after a clean equal-step pair.
    auto provider = provider_factory{ ops, {} }.make_provider();
    using replay  = factory_type::flow_type::replay_type;
    replay                   work( ops, provider, { 2 } );
    std::vector<record_type> records;
    double                   time = .5;
    ops->assign( *initial, *candidate );
    for ( const double h : { .04, .04, .02, .02 } )
    {
        native->set_time( time );
        native->set_target_time( time + h );
        native->apply( *candidate, *candidate );
        require( native->get_status() == ts::single_step_status::converged, "Clipped BDF step failed" );
        records.push_back( record_type::capture( *native, time, native->get_dt(), time + h, 1, time + h ) );
        time += h;
        native->finalize( ts::adaptation_status::accepted, time, *candidate );
    }
    require(
        records[0].order == 1 && records[1].order == 2 && records[2].order == 1 && records[3].order == 2,
        "Changed-step BDF history did not restart/promote"
    );
    auto replayed = buffer( ops );
    work.run( { 2 }, records, *initial, nullptr, *replayed, nullptr, nullptr );
    ops->add_lin_comb( -1, *candidate, *replayed );
    require( ops->norm( *replayed ) < 1e-10, "BDF changed-step base replay mismatch" );

    auto invalid     = records;
    invalid[2].order = 2;
    require_throws<std::invalid_argument>(
        [&] { work.run( { 2 }, invalid, *initial, nullptr, *replayed, nullptr, nullptr ); },
        "BDF replay accepted order two after an actual step change"
    );

    // An event-shortened commitment invalidates the old multistep history.
    native->set_time( time );
    native->set_target_time( time + .04 );
    native->apply( *candidate, *candidate );
    const auto view = native->get_continuous_integration();
    view.evaluate( .5, *candidate );
    time += native->get_dt() / 2;
    native->finalize( ts::adaptation_status::accepted_modified, time, *candidate );
    native->set_time( time );
    native->set_target_time( time + .04 );
    native->apply( *candidate, *candidate );
    require( native->get_order() == 1, "Modified BDF endpoint retained old history" );
    native->finalize( ts::adaptation_status::failed, time, *candidate );

    // Roundoff-only target clipping must preserve the native order-two choice.
    native->reset();
    fill( *ops, *candidate, 1, 0 );
    time = 100;
    records.clear();
    const double target = time + .12;
    for ( unsigned i = 0; i < 3; ++i )
    {
        native->set_time( time );
        native->set_target_time( target );
        native->apply( *candidate, *candidate );
        const auto h   = native->get_dt();
        const auto end = h == target - time ? target : time + h;
        records.push_back( record_type::capture( *native, time, h, end, 1, end ) );
        native->finalize( ts::adaptation_status::accepted, end, *candidate );
        time = end;
    }
    require( time == target && records.back().order == 2, "Roundoff clipping restarted BDF2" );
    work.run( { 2 }, records, *initial, nullptr, *replayed, nullptr, nullptr );
    ops->add_lin_comb( -1, *candidate, *replayed );
    require( ops->norm( *replayed ) < 1e-10, "Roundoff-clipped BDF replay mismatch" );
    native->set_time( time );
    native->set_target_time( time + 1 );
    native->apply( *candidate, *candidate );
    require( native->get_order() == 1, "BDF native post-clipping restart was not exercised" );
    const auto h = native->get_dt();
    records.push_back( record_type::capture( *native, time, h, time + h, 1, time + h ) );
    native->finalize( ts::adaptation_status::accepted, time + h, *candidate );
    work.run( { 2 }, records, *initial, nullptr, *replayed, nullptr, nullptr );
    ops->add_lin_comb( -1, *candidate, *replayed );
    require( ops->norm( *replayed ) < 1e-10, "Post-clipping BDF restart replay mismatch" );
}

void check_dense_time_derivative( unsigned int order )
{
    auto ops     = std::make_shared<operations>( 2 );
    auto initial = buffer( ops ), out = buffer( ops ), rate = buffer( ops );
    auto plus = buffer( ops ), minus = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    auto                                 provider = provider_factory{ ops, {} }.make_provider();
    factory_type::flow_type::replay_type work( ops, provider, { order } );
    std::vector<record_type>             records{ { .5, .04, .54, 1, .54, 1 }, { .54, .04, .556, .4, .58, order } };
    work.run( { order }, records, *initial, nullptr, *out, nullptr, &*rate );
    const auto   time    = records.back().end;
    const double epsilon = 1e-6;
    records.back().theta += epsilon;
    records.back().end = time + epsilon * records.back().dt;
    work.run( { order }, records, *initial, nullptr, *plus, nullptr, nullptr );
    records.back().theta -= 2 * epsilon;
    records.back().end = time - epsilon * records.back().dt;
    work.run( { order }, records, *initial, nullptr, *minus, nullptr, nullptr );
    ops->assign_lin_comb(
        .5 / ( epsilon * records.back().dt ), *plus, -.5 / ( epsilon * records.back().dt ), *minus, *out
    );
    ops->add_lin_comb( -1, *rate, *out );
    require( ops->norm( *out ) < 1e-7, "BDF dense polynomial time derivative mismatch" );
}
void test()
{
    for ( unsigned int order : { 1u, 2u } )
    {
        check_dense_time_derivative( order );
        check( order );
        check( order, system_kind::coupled_mass );
        check( order, system_kind::state_mass );
        check( order, system_kind::time_rotation );
        check( order, system_kind::time_rotation, 2, true );
        check( order, system_kind::linear_dae, 3 );
    }
    check( 2, system_kind::cycle, 8192 );
    check_commits_and_restarts();
    require( guarded_memory::host_bytes == 0, "BDF sensitivities transferred vectors" );
    require( guarded_memory::live_allocations == 0, "BDF sensitivities leaked storage" );
}
int main()
{
    return run( test );
}
