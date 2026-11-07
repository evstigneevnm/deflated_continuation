#include "bdf_flight_test_support.h"
using namespace bdf_flights;

// The isolated unit cycle, its coupled/state-dependent mass variants, and the
// manufactured rotating linear DAE are defined in implicit_flight_test_support.h.
// Freeze the adaptive mesh/orders before differentiating; controllers are not
// part of the return-map derivative. Final event times remain differentiable.
void check( unsigned int cap, system_kind kind = system_kind::cycle, bool backward = false, std::size_t size = 2 )
{
    std::cout << "Adaptive BDF derivative: cap=" << cap << ", model=" << int( kind ) << ", N=" << size
              << ", backward=" << backward << std::endl;
    model equation{ kind };
    auto  ops     = std::make_shared<operations>( size );
    auto  initial = buffer( ops ), arrival = buffer( ops ), v = buffer( ops ), image = buffer( ops );
    auto  plus = buffer( ops ), minus = buffer( ops ), rate = buffer( ops ), sum = buffer( ops );
    fill( *ops, *initial, 1, 0, 1, equation.components() );
    fill( *ops, *v, .3, -.4, -.1, equation.components() );
    auto native = make_adaptive_step( ops, equation, cap, cap == 1 ? 1e-4 : 1e-6 );
    auto map    = std::make_shared<adaptive_map>(
        ops, native, std::make_shared<physical_rate>( physical_rate{ ops, equation } )
    );
    auto r = request( ops, backward ? 2. : .4 );
    if ( backward )
    {
        r.end_time       = 0;
        r.rule.direction = periodic_orbit::crossing_direction::increasing;
    }
    if ( kind == system_kind::linear_dae )
    {
        fill( *ops, *plus, 0, 0, 0, 3 );
        fill( *ops, *minus, 1, 0, 0, 3 );
        r.target = std::make_shared<plane>( ops, *plus, *minus );
    }
    require( map->evaluate( *initial, r, *arrival ).succeeded(), "Adaptive BDF flight failed" );
    const auto schedule = map->records();
    require( schedule.front().order == 1, "Adaptive BDF did not cold start" );
    unsigned int maximum = 0;
    for ( const auto &s : schedule )
        maximum = std::max( maximum, s.order );
    require( maximum == cap, "Adaptive flight did not exercise its order cap" );
    require( schedule.front().dt != schedule.back().dt, "Adaptive flight used a constant mesh" );
    auto         providers = std::make_shared<provider_factory>( provider_factory{ ops, equation } );
    factory_type factory( ops, providers, { cap, 1000000, true } );
    struct segment
    {
        std::shared_ptr<adaptive_map> flight;
    };
    auto       workspace   = factory.make_workspace( std::vector<segment>{ { map } } );
    auto       flight      = factory.prepare( workspace, *map, *initial, *arrival, r.target, .1, 1e-8 );
    const auto allocations = guarded_memory::allocation_calls;
    flight.apply( *v, *image );
    const auto dt = flight.last_time_increment();
    ops->assign_lin_comb( 1e-10, *v, *sum );
    flight.apply( *sum, *sum );
    ops->scale( 1e10, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) < 2e-7, "Adaptive BDF Jv scale dependence" );
    if ( cap == 5 && kind == system_kind::cycle && size == 2 )
    {
        fill( *ops, *plus, .1, .2 );
        flight.apply( *plus, *minus );
        ops->assign_lin_comb( 2, *v, -3, *plus, *sum );
        flight.apply( *sum, *sum );
        ops->add_lin_comb( -2, *image, *sum );
        ops->add_lin_comb( 3, *minus, *sum );
        require( ops->norm( *sum ) < 2e-8, "Adaptive BDF Jv is not linear" );
    }
    require( guarded_memory::allocation_calls == allocations, "Adaptive BDF Jv allocated vectors" );
    typename plane::evaluation_workspace section_work( ops );
    auto                                 trial_initial = buffer( ops );
    const double                         epsilon       = 2e-5;
    auto                                 frozen_event  = [&]( const X &input, X &out ) {
        ops->assign( input, *trial_initial );
        auto mesh = schedule;
        for ( int k = 0; k < 3; ++k )
        {
            workspace->run( { cap, true }, mesh, *trial_initial, nullptr, out, nullptr, &*rate );
            const auto section = r.target->bind( section_work );
            const auto correction =
                section.value( mesh.back().end, out ) / section.state_derivative( mesh.back().end, out, *rate );
            mesh.back().end -= correction;
            mesh.back().theta = ( mesh.back().end - mesh.back().time ) / mesh.back().dt;
        }
        workspace->run( { cap, true }, mesh, *trial_initial, nullptr, out, nullptr, nullptr );
        return mesh.back().end;
    };
    ops->assign_lin_comb( 1, *initial, epsilon, *v, *plus );
    const auto positive_time = frozen_event( *plus, *plus );
    ops->assign_lin_comb( 1, *initial, -epsilon, *v, *minus );
    const auto negative_time = frozen_event( *minus, *minus );
    ops->assign_lin_comb( .5 / epsilon, *plus, -.5 / epsilon, *minus, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) < 2e-5 * std::sqrt( double( size ) ), "Adaptive frozen section finite difference" );
    require(
        std::abs( ( positive_time - negative_time ) / ( 2 * epsilon ) - dt ) < 2e-5,
        "Adaptive section-time finite difference"
    );
    providers->flags->refuse_rhs = true;
    ops->assign_scalar( 7, *sum );
    const auto saved = ops->norm( *sum ), saved_time = flight.last_time_increment();
    require_throws<std::runtime_error>( [&] { flight.apply( *v, *sum ); }, "Adaptive tangent failure ignored" );
    require(
        ops->norm( *sum ) == saved && flight.last_time_increment() == saved_time, "Failed adaptive Jv published output"
    );
    providers->flags->refuse_rhs = false;
    if ( cap == 5 && kind == system_kind::cycle && size == 2 )
    {
        flight.apply( *v, *sum );
        ops->add_lin_comb( -1, *image, *sum );
        require( ops->norm( *sum ) < 1e-12, "Adaptive BDF history did not recover after a failed Jv" );
    }
    auto invalid          = schedule;
    invalid.front().order = 2;
    require_throws<std::invalid_argument>(
        [&] { workspace->run( { cap, true }, invalid, *initial, nullptr, *sum, nullptr, nullptr ); },
        "Adaptive replay accepted missing history"
    );
}

// A scripted assessment isolates order/history handling from accuracy control.
// Two rejections lower order without discarding history; extreme clipping later
// really resets it. Auxiliary estimator steps must not enter the replay mesh.
struct scripted_control : adaptive_control
{
    using adaptive_control::adaptive_control;
    unsigned int          reject = 0;
    ts::adaptation_status assess(
        double, double, const X &, const X &, double &next, unsigned int = 0, const X * = nullptr, const X * = nullptr
    )
    {
        next = get_dt();
        if ( reject )
        {
            --reject;
            return ts::adaptation_status::rejected;
        }
        return ts::adaptation_status::accepted;
    }
};
void check_history_and_dense( unsigned int cap )
{
    auto ops     = std::make_shared<operations>( 2 );
    auto initial = buffer( ops ), candidate = buffer( ops ), replayed = buffer( ops ), rate = buffer( ops );
    auto plus = buffer( ops ), minus = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    ops->assign( *initial, *candidate );
    using native =
        ts::bdf::implicit_time_step<operations, residual_problem, scripted_control, newton<residual_problem>, true>;
    adaptive_control::params control_params;
    control_params.initial_step = .01;
    auto           control      = std::make_shared<scripted_control>( ops, control_params );
    native::params p;
    p.method        = "BDF";
    p.adaptive      = true;
    p.maximum_order = cap;
    native step(
        ops, std::make_shared<residual_problem>( ops, model{} ), control,
        implicit_flights::make_newton<residual_problem>( ops, make_linear( ops ) ), p
    );
    std::vector<record_type> records;
    double                   time = .4;
    for ( unsigned int i = 0; i < 70; ++i )
    {
        if ( i == 25 )
            control->reject = 2;
        step.set_time( time );
        step.set_target_time( time + ( i == 35 ? .001 : 1. ) );
        step.apply( *candidate, *candidate );
        require( step.get_status() == ts::single_step_status::converged, "Scripted adaptive step failed" );
        const auto end = time + step.get_dt();
        records.push_back( record_type::capture( step, time, step.get_dt(), end, 1, end ) );
        if ( i == 25 )
            require( step.get_attempts() == 3 && step.get_order() == cap - 1, "Retry did not lower BDF order" );
        if ( i == 35 )
            require( step.get_order() == 1, "Extreme shrinkage did not reset BDF history" );
        step.finalize( ts::adaptation_status::accepted, end, *candidate );
        time = end;
    }
    auto                                 provider = provider_factory{ ops, {} }.make_provider();
    const auto                           live     = guarded_memory::live_allocations;
    factory_type::flow_type::replay_type replay( ops, provider, { cap, true } );
    require(
        guarded_memory::live_allocations - live == 9 + 2 * ( cap + 1 ),
        "Adaptive replay storage is not bounded by its order cap"
    );
    replay.run( { cap, true }, records, *initial, nullptr, *replayed, nullptr, nullptr );
    ops->add_lin_comb( -1, *candidate, *replayed );
    require( ops->norm( *replayed ) < 1e-11, "Order reduction/reset replay differs from native BDF" );
    require( records.back().order == cap, "Scripted grid did not recover its high order" );
    records.back().theta = .37;
    records.back().end   = records.back().time + .37 * records.back().dt;
    replay.run( { cap, true }, records, *initial, nullptr, *replayed, nullptr, &*rate );
    const double epsilon = 1e-5;
    auto         shifted = records;
    shifted.back().theta += epsilon;
    shifted.back().end = shifted.back().time + shifted.back().theta * shifted.back().dt;
    replay.run( { cap, true }, shifted, *initial, nullptr, *plus, nullptr, nullptr );
    shifted.back().theta -= 2 * epsilon;
    shifted.back().end = shifted.back().time + shifted.back().theta * shifted.back().dt;
    replay.run( { cap, true }, shifted, *initial, nullptr, *minus, nullptr, nullptr );
    ops->assign_lin_comb(
        .5 / ( epsilon * records.back().dt ), *plus, -.5 / ( epsilon * records.back().dt ), *minus, *replayed
    );
    ops->add_lin_comb( -1, *rate, *replayed );
    require( ops->norm( *replayed ) < 1e-7, "High-order BDF dense time derivative" );
    auto invalid      = records;
    invalid[35].order = 2;
    require_throws<std::invalid_argument>(
        [&] { replay.run( { cap, true }, invalid, *initial, nullptr, *replayed, nullptr, nullptr ); },
        "High-order replay accepted history across restart"
    );
}
void test()
{
    for ( unsigned int cap : { 1u, 2u, 3u, 4u, 5u } )
        check( cap );
    check( 5, system_kind::coupled_mass );
    check( 5, system_kind::state_mass );
    check( 5, system_kind::linear_dae, false, 3 );
    check( 5, system_kind::time_rotation );
    check( 5, system_kind::time_rotation, true );
    check( 5, system_kind::cycle, false, 8192 );
    for ( unsigned int cap : { 3u, 4u, 5u } )
        check_history_and_dense( cap );
    require( guarded_memory::host_bytes == 0, "Adaptive BDF transferred vectors" );
    require( guarded_memory::live_allocations == 0, "Adaptive BDF leaked storage" );
}
int main()
{
    return run( test );
}
