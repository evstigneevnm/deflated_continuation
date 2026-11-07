#include "implicit_flight_test_support.h"
using namespace implicit_flights;

// Differentiate the same numerical section map, not an independently integrated
// variational ODE. Its fixed steps are also verified during finite differences.
void check(
    const std::string &method, system_kind kind = system_kind::cycle, std::size_t dimension = 2, bool backward = false
)
{
    model equation{ kind };
    auto  ops     = std::make_shared<operations>( dimension );
    auto  initial = buffer( ops ), arrival = buffer( ops ), v = buffer( ops ), w = buffer( ops ), image = buffer( ops ),
         other = buffer( ops ), plus = buffer( ops ), minus = buffer( ops ), sum = buffer( ops );
    fill( *ops, *initial, 1, 0, 1, equation.components() );
    auto         providers = std::make_shared<provider_factory>( provider_factory{ ops, equation } );
    const auto   before    = guarded_memory::allocation_calls;
    factory_type factory( ops, providers );
    require(
        providers->creations == 0 && guarded_memory::allocation_calls == before,
        "Factory eagerly allocated tangent storage"
    );
    auto map = make_map( ops, equation, method, .04 );
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
    const auto report = map->evaluate( *initial, r, *arrival );
    require( report.succeeded(), "DIRK base section flight failed" );
    require( providers->creations == 0, "Ordinary flight constructed derivative provider" );
    struct segment
    {
        std::shared_ptr<recorded_map> flight;
    };
    auto       workspace = factory.make_workspace( std::vector<segment>{ { map } } );
    auto       flight    = factory.prepare( workspace, *map, *initial, *arrival, r.target, .1, 1e-8 );
    const auto schedule  = flight.records();
    fill( *ops, *v, .3, -.4, -.1, equation.components() );
    fill( *ops, *w, -.2, .1, -.1, equation.components() );
    const auto allocated = guarded_memory::allocation_calls;
    flight.apply( *v, *image );
    flight.apply( *w, *other );
    ops->assign_lin_comb( 2, *v, -3, *w, *sum );
    flight.apply( *sum, *sum );
    ops->add_lin_comb( -2, *image, *sum );
    ops->add_lin_comb( 3, *other, *sum );
    const double scale = std::sqrt( double( dimension / equation.components() ) );
    require( ops->norm( *sum ) / scale < 2e-8, "DIRK numerical Jv is not linear" );
    require( guarded_memory::allocation_calls == allocated, "DIRK Jv allocated backend vectors" );
    const double epsilon = 2e-5;
    ops->assign_lin_comb( 1, *initial, epsilon, *v, *plus );
    require( map->evaluate( *plus, r, *plus ).succeeded(), "Positive DIRK finite difference failed" );
    auto check_mesh = [&] {
        require( map->records().size() == schedule.size(), "Finite difference changed step count" );
        for ( std::size_t i = 0; i < schedule.size(); ++i )
            require(
                map->records()[i].time == schedule[i].time && map->records()[i].dt == schedule[i].dt,
                "Finite difference changed frozen mesh"
            );
    };
    check_mesh();
    ops->assign_lin_comb( 1, *initial, -epsilon, *v, *minus );
    require( map->evaluate( *minus, r, *minus ).succeeded(), "Negative DIRK finite difference failed" );
    check_mesh();
    ops->assign_lin_comb( .5 / epsilon, *plus, -.5 / epsilon, *minus, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) / scale < 2e-5, "DIRK section derivative finite difference mismatch" );
    flight.apply( *v, *sum );
    ops->add_lin_comb( -1, *image, *sum );
    require( ops->norm( *sum ) / scale < 1e-12, "Trial residual corrupted prepared DIRK flight" );
    fill( *ops, *v, 1, 0, 1, equation.components() );
    flight.apply( *v, *sum );
    if ( kind != system_kind::linear_dae && kind != system_kind::time_rotation )
    {
        fill( *ops, *other, 0, std::exp( -pi ) );
        ops->add_lin_comb( -1, *other, *sum );
        require( ops->norm( *sum ) / scale < .04, "DIRK analytic radial return derivative" );
    }
    if ( kind == system_kind::linear_dae )
    {
        fill( *ops, *other, 0, 1, 1, 3 );
        ops->add_lin_comb( -1, *other, *sum );
        require( ops->norm( *sum ) < .03, "DAE consistent return derivative" );
    }
    providers->flags->refuse_rhs = true;
    ops->assign_scalar( 7, *sum );
    const auto norm = ops->norm( *sum ), dt = flight.last_time_increment();
    require_throws<std::runtime_error>( [&] { flight.apply( *v, *sum ); }, "DIRK tangent failure ignored" );
    require( ops->norm( *sum ) == norm && flight.last_time_increment() == dt, "Failed DIRK Jv published partial data" );
    providers->flags->refuse_rhs = false;
    if ( method == "ESDIRK2(1)3" || method == "SDIRK3(1)3" )
    {
        providers->flags->refuse_rate = true;
        require_throws<std::runtime_error>( [&] { flight.apply( *v, *sum ); }, "DIRK rate tangent failure ignored" );
        require(
            ops->norm( *sum ) == norm && flight.last_time_increment() == dt, "Rate tangent failure published data"
        );
        providers->flags->refuse_rate = false;
    }
    using replay        = typename factory_type::flow_type::replay_type;
    auto       provider = providers->make_provider();
    const auto live     = guarded_memory::live_allocations;
    auto       bounded  = std::make_shared<replay>( ops, provider, map->table().dense_outout_stage_count(), 1 );
    require(
        guarded_memory::live_allocations - live == 8 + 2 * map->table().dense_outout_stage_count(),
        "DIRK replay storage is not stage-bounded"
    );
    require_throws<std::invalid_argument>(
        [&] { bounded->run( map->table(), schedule, *initial, nullptr, *sum, nullptr, nullptr ); },
        "Unbounded DIRK replay accepted"
    );
    factory_type limited( ops, providers, factory_type::params{ 1 } );
    const auto   creations = providers->creations;
    require_throws<std::invalid_argument>(
        [&] { limited.make_workspace( std::vector<segment>{ { map } } ); }, "Factory ignored replay budget"
    );
    require( providers->creations == creations, "Over-budget replay constructed a provider" );
}

void check_optional_rates()
{
    auto ops       = std::make_shared<operations>( 2 );
    auto lin       = make_linear( ops );
    using provider = periodic_orbit::dirk_stage_provider<
        operations, residual_problem, newton<residual_problem>, linear, sensitivity>;
    using replay = periodic_orbit::dirk_flight_replay<operations, provider>;
    static_assert( !provider::supports_rate );
    require(
        replay::supports( ts::runge_kutta::make_butcher_table( "SDIRK2(1)2" ) ), "Residual-only DIRK derivative refused"
    );
    require(
        !replay::supports( ts::runge_kutta::make_butcher_table( "ESDIRK2(1)3" ) ),
        "Missing explicit-rate derivative silently accepted"
    );
    using plain  = ts::runge_kutta::implicit_time_step<operations, residual_problem, fixed, newton<residual_problem>>;
    auto problem = std::make_shared<residual_problem>( ops, model{} );
    auto state   = buffer( ops );
    fill( *ops, *state, 1, 0 );
    plain step(
        ops, problem, std::make_shared<fixed>( fixed::params{ .02 } ), make_newton<residual_problem>( ops, lin ),
        plain::params{ "IM" }
    );
    step.apply( *state, *state );
    require(
        step.get_status() == ts::single_step_status::converged,
        "Ordinary implicit user required sensitivity/rate methods"
    );
    step.finalize( ts::adaptation_status::accepted, step.get_dt(), *state );

    // A residual-only problem and sensitivity with no rate-Jacobian member
    // still execute a full dense DIRK tangent, rather than only passing traits.
    struct only_reference
    {
        sensitivity base;
        bool stage_tangent_rhs( const periodic_orbit::dirk_stage_context<operations> &c, const X &dw, X &out ) const
        {
            return base.stage_tangent_rhs( c, dw, out );
        }
    };
    using residual_provider = periodic_orbit::dirk_stage_provider<
        operations, residual_problem, newton<residual_problem>, linear, only_reference>;
    using residual_replay = periodic_orbit::dirk_flight_replay<operations, residual_provider>;
    auto reference =
        std::make_shared<only_reference>( only_reference{ { ops, {}, std::make_shared<control_flags>() } } );
    auto stage_provider =
        std::make_shared<residual_provider>( ops, problem, make_newton<residual_problem>( ops, lin ), lin, reference );
    const auto      table = ts::runge_kutta::make_butcher_table( "SDIRK2(1)2" );
    residual_replay work( ops, stage_provider, table.dense_outout_stage_count() );
    auto            out = buffer( ops ), v = buffer( ops ), dv = buffer( ops );
    fill( *ops, *state, 1, 0 );
    fill( *ops, *v, .3, -.4 );
    work.run( table, { { 0, .02, .008, .4 } }, *state, &*v, *out, &*dv, nullptr );
    require(
        ops->is_valid_number( *out ) && ops->is_valid_number( *dv ), "Residual-only dense tangent did not execute"
    );
}

struct observed_adaptive : adaptive
{
    using adaptive::adaptive;
    unsigned int          assessments = 0;
    ts::adaptation_status assess(
        double t, double h, const X &in, const X &out, double &next, unsigned int order = 0, const X *error = nullptr
    )
    {
        ++assessments;
        return adaptive::assess( t, h, in, out, next, order, error );
    }
};
void check_adaptive()
{
    auto  ops = std::make_shared<operations>( 2 );
    model equation{ system_kind::time_rotation };
    using step_type =
        ts::runge_kutta::implicit_time_step<operations, rate_problem, observed_adaptive, newton<rate_problem>, true>;
    auto             problem = std::make_shared<rate_problem>( ops, equation );
    adaptive::params p;
    p.initial_step       = .02;
    p.maximum_step       = .12;
    p.relative_tolerance = p.absolute_tolerance = 1e-8;
    auto controller                             = std::make_shared<observed_adaptive>( ops, p );
    auto native                                 = std::make_shared<step_type>(
        ops, problem, controller, make_newton<rate_problem>( ops, make_linear( ops ) ),
        step_type::params{ "ARK324L2SA" }
    );
    using map_type = periodic_orbit::recorded_section_map<operations, step_type, plane, physical_rate>;
    map_type map( ops, native, std::make_shared<physical_rate>( physical_rate{ ops, equation } ) );
    auto     initial = buffer( ops ), arrival = buffer( ops ), v = buffer( ops ), out = buffer( ops );
    fill( *ops, *initial, 1, 0 );
    fill( *ops, *v, 0, 1 );
    auto r = request( ops );
    require( map.evaluate( *initial, r, *arrival ).succeeded(), "Adaptive DIRK flight failed" );
    require(
        std::any_of(
            map.records().begin(), map.records().end(),
            [&]( const auto &s ) { return s.dt != map.records().front().dt; }
        ),
        "DIRK adaptive mesh did not vary"
    );
    auto providers = std::make_shared<provider_factory>( provider_factory{ ops, equation } );
    using replay   = factory_type::flow_type::replay_type;
    auto work = std::make_shared<replay>( ops, providers->make_provider(), map.table().dense_outout_stage_count() );
    factory_type::flow_type flight( ops, work, map.table(), map.records(), *initial, *arrival, r.target, .1, 1e-8 );
    const auto              count = controller->assessments;
    flight.apply( *v, *out );
    flight.apply( *v, *out );
    require( controller->assessments == count, "DIRK replay reran the adaptive controller" );
    require(
        ops->norm( *out ) < 2e-5 && std::abs( flight.last_time_increment() + 1 / std::sqrt( .16 + pi ) ) < 2e-5,
        "Adaptive non-autonomous section derivative"
    );
}

void test()
{
    for ( const char *method : { "IE", "IM", "CN", "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
        check( method );
    check( "SDIRK2(1)2", system_kind::coupled_mass );
    check( "SDIRK2(1)2", system_kind::state_mass );
    check( "ESDIRK2(1)3", system_kind::time_rotation );
    check( "ESDIRK2(1)3", system_kind::time_rotation, 2, true );
    check( "SDIRK2(1)2", system_kind::linear_dae, 3 );
    check( "SDIRK2(1)2", system_kind::cycle, 8192 );
    check_optional_rates();
    check_adaptive();
    require( guarded_memory::host_bytes == 0, "DIRK sensitivities transferred vector data" );
    require( guarded_memory::live_allocations == 0, "Optional sensitivity buffers leaked" );
}
int main()
{
    return run( test );
}
