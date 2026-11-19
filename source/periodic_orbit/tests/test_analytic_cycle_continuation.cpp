#include "prepared_cycle_support.h"
#include "benchmarks/branch_solver.h"
#include <periodic_orbit/matching/refresh_section_branch.h>
#include <periodic_orbit/matching/rechart_orbit.h>
#include <periodic_orbit/sections/layout_health.h>
#include <nmfd/time_steppers/integration/time_integrator.h>
#include <cstdio>

namespace
{
using namespace cycle_tests;
// r'=(lambda-r^2)r, angle'=omega. At r=sqrt(lambda) the transverse
// multiplier is exp(-2*lambda*T), with T=2*pi/omega.
struct cycle
{
    double                lambda = 1;
    __DEVICE_TAG__ double omega() const
    {
        return 1 + .25 * ( lambda - 1 );
    }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return ( lambda - u[0] * u[0] - u[1] * u[1] ) * u[i] + omega() * ( i ? u[0] : -u[1] );
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        return ( lambda - u[0] * u[0] - u[1] * u[1] ) * v[i] - 2 * u[i] * ( u[0] * v[0] + u[1] * v[1] ) +
               omega() * ( i ? v[0] : -v[1] );
    }
};
struct source_term
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return u[i] + .25 * ( i ? u[0] : -u[1] );
    }
};
struct rate : tangent_rate<cycle>
{
    bool parameter_derivative( double t, const X &u, X &out ) const
    {
        backend::for_each_type{}( rhs_kernel<source_term>{ u.raw_ptr(), out.raw_ptr(), t, {} }, out.size() );
        return true;
    }
};
using map     = periodic_orbit::recorded_section_map<operations, step<cycle>, plane, rate_provider<cycle>>;
using matcher = periodic_orbit::multiple_section_problem<operations, operations, map, rate>;
using space   = periodic_orbit::orbit_continuation_space<blocks>;
struct binding
{
    std::shared_ptr<operations>           physical;
    std::shared_ptr<problem<cycle>>       rhs;
    std::shared_ptr<rate_provider<cycle>> crossing;
    std::shared_ptr<bool>                 miss_next;
    void                                  set_trial_parameter( double p )
    {
        if ( std::exchange( *miss_next, false ) )
            p = -3; // Zero angular speed: no section return.
        rhs->model.lambda = crossing->model.lambda = p;
    }
    matcher::factory_type make_factory( double p ) const
    {
        auto frozen          = std::make_shared<rate>();
        frozen->model.lambda = p;
        return { physical, frozen };
    }
};
using orbit = periodic_orbit::parameterized_multiple_section_problem<matcher, binding>;
struct setup
{
    fixture                     pilot;
    std::shared_ptr<operations> physical = pilot.physical;
    std::shared_ptr<chart>      first    = pilot.segments.front().chart;
    std::shared_ptr<space>      coords;
    std::shared_ptr<orbit>      op;
    std::shared_ptr<bool>       miss_next = std::make_shared<bool>( false );
    auto                        workspace() const
    {
        return block_buffer( coords );
    }
    explicit setup( unsigned m ) : pilot( m, .08 ), coords( std::make_shared<space>( pilot.coordinates, m ) )
    {
        binding b{
            pilot.physical, std::make_shared<problem<cycle>>( problem<cycle>{ pilot.physical, {} } ),
            std::make_shared<rate_provider<cycle>>(), miss_next
        };
        std::vector<matcher::segment> segments;
        map::map_type::params         event;
        event.event.value_tolerance = event.event.time_tolerance = 2e-14;
        for ( unsigned i = 0; i < m; ++i )
        {
            auto rk = std::make_shared<step<cycle>>(
                pilot.physical, b.rhs, std::make_shared<fixed>( fixed::params{ .07 } ), step<cycle>::params{ "RK45" }
            );
            segments.push_back(
                { std::make_shared<map>( pilot.physical, rk, b.crossing, event ),
                  std::make_shared<chart>( pilot.physical, pilot.coordinates, pilot.layout->at( i ).section ) }
            );
        }
        auto matching = std::make_shared<matcher>(
            pilot.physical, coords, pilot.layout, std::move( segments ), std::make_shared<rate>()
        );
        op = std::make_shared<orbit>( matching, b, coords );
    }
};
void derivatives( setup &s )
{
    auto z = block_buffer( s.coords ), col = block_buffer( s.coords ), a = block_buffer( s.coords ),
         b = block_buffer( s.coords );
    s.coords->assign_scalar( .12, *z );
    constexpr double p = 1.04, eps = 1e-5;
    s.op->set_linearization_point( *z, p );
    auto prepared = s.op->get_jacobi_operator();
    s.op->jacobian_alpha( *col );
    prepared->apply( *z, *a );
    auto saved = block_buffer( s.coords );
    s.coords->assign( *a, *saved );
    // Fixed interior RK mesh; both perturbed solves locate their own final crossing.
    s.op->F( *z, p + eps, *a );
    const auto tplus = s.op->report().period;
    s.op->F( *z, p - eps, *b );
    const auto tprime = ( tplus - s.op->report().period ) / ( 2 * eps );
    require( std::abs( tprime - s.op->period_parameter_partial() ) < 2e-7, "Flight time parameter derivative" );
    s.coords->assign_lin_comb( .5 / eps, *a, -.5 / eps, *b, *a );
    s.coords->add_lin_comb( -1, *col, *a );
    require( s.coords->norm( *a ) < 1e-6, "Section parameter column" );
    s.op->set_linearization_point( *z, p + .1 );
    prepared->apply( *z, *a );
    s.coords->add_lin_comb( -1, *saved, *a );
    require( s.coords->norm( *a ) < 1e-13, "Rebinding changed retained flight parameters" );
}

using branch     = orbit_benchmarks::branch_solver<setup>;
using pair_space = branch::pair_space;
void check_point( setup &s, const blocks::vector_type &z, double p )
{
    auto       &f           = s.pilot;
    auto        work        = block_buffer( s.coords );
    const auto &diagnostics = s.op->inspect_crossings( z, p, *work );
    const auto  period      = diagnostics.period;
    using condition         = periodic_orbit::layout_condition;
    periodic_orbit::layout_health<double> health;
    require(
        health.inspect(
            *f.layout, 2 * pi, period, diagnostics,
            []( double ) {
                throw std::logic_error( "Healthy layout launched an extra orbit verification" );
                return false;
            }
        ) == condition::healthy,
        "Analytical branch should need no layout refresh"
    );
    auto two_laps = diagnostics;
    two_laps.period *= 2;
    require(
        health.inspect( *f.layout, 2 * pi, period, two_laps, [&]( double t ) { return t == period; } ) ==
            condition::repeated_circuit,
        "Shorter closure detects a second lap"
    );
    require(
        health.inspect( *f.layout, 2 * pi, period, two_laps, []( double ) { return false; } ) !=
            condition::repeated_circuit,
        "Period ratio alone is not proof of a repeated lap"
    );
    std::cout << "M=" << z.size() << " lambda=" << p << " gap=" << s.coords->norm( *work )
              << " period error=" << period - 2 * pi / cycle{ p }.omega() << '\n';
    require(
        s.coords->norm( *work ) < 2e-10 && std::abs( period - 2 * pi / cycle{ p }.omega() ) < 1e-7,
        "Periodic branch residual/period"
    );
    auto start = buffer( f.physical ), end = buffer( f.physical );
    for ( std::size_t i = 0; i < z.size(); ++i )
    {
        f.segments[i].chart->lift( z.comp( i ), *start );
        require( std::abs( f.physical->norm( *start ) - std::sqrt( p ) ) < 2e-7, "Periodic branch radius" );
    }
    using integrator = nmfd::time_steppers::integration::time_integrator<operations, step<cycle>>;
    integrator run( f.physical, make_step( f.physical, .02, cycle{ p } ), typename integrator::params{ 2000 } );
    run.set_time_interval( 0, period );
    run.apply( *start, *end );
    f.physical->add_lin_comb( -1, *start, *end );
    require(
        run.get_status() == nmfd::time_steppers::integration_status::completed && f.physical->norm( *end ) < 3e-7,
        "Independent full-period closure"
    );
}
// This fixture's layout is reproducible from M and its fixed pilot. Checkpoint
// I/O, unlike continuation, is explicitly allowed to transfer coordinates.
void checkpoint(
    setup &s, const blocks::vector_type &z, double p, const blocks::vector_type &v, double vp, blocks::vector_type &rz,
    double &rp, blocks::vector_type &rv, double &rvp
)
{
    std::unique_ptr<FILE, decltype( &std::fclose )> file( std::tmpfile(), std::fclose );
    require( bool( file ), "Creating orbit checkpoint" );
    auto write = [&]( double x ) {
        require( std::fwrite( &x, sizeof( x ), 1, file.get() ) == 1, "Writing orbit checkpoint" );
    };
    auto read = [&]() {
        double x;
        require( std::fread( &x, sizeof( x ), 1, file.get() ) == 1, "Reading orbit checkpoint" );
        return x;
    };
    write( double( z.size() ) );
    write( p );
    write( vp );
    for ( std::size_t i = 0; i < z.size(); ++i )
    {
        write( s.pilot.coordinates->get_value_at_point( 0, z.comp( i ) ) );
        write( s.pilot.coordinates->get_value_at_point( 0, v.comp( i ) ) );
    }
    std::rewind( file.get() );
    require( read() == double( rz.size() ), "Incompatible restart layout" );
    rp  = read();
    rvp = read();
    for ( std::size_t i = 0; i < rz.size(); ++i )
    {
        s.pilot.coordinates->assign_scalar( read(), rz.comp( i ) );
        s.pilot.coordinates->assign_scalar( read(), rv.comp( i ) );
    }
}
void check( unsigned m )
{
    setup s( m );
    derivatives( s );
    branch solver( s );
    auto   z = block_buffer( s.coords ), v = block_buffer( s.coords ), next = block_buffer( s.coords ),
         nv  = block_buffer( s.coords );
    double p = 1, vp = 0, q = 0, vq = 0;
    s.coords->assign_scalar( .01, *z );
    require( solver.ordinary.solve( s.op.get(), *z, p ), "Seed correction" );
    require( solver.tan->initial( *z, p, 1, *v, vp ), "Initial augmented tangent" );
    // Refuse a replacement after its fixed correction, before publishing ANY new coordinates.
    struct epoch
    {
        setup  flights;
        branch solver;
        explicit epoch( unsigned count ) : flights( count ), solver( flights )
        {
        }
    };
    auto saved_z = s.workspace(), saved_v = s.workspace();
    s.coords->assign( *z, *saved_z );
    s.coords->assign( *v, *saved_v );
    const double saved_vp = vp;
    s.coords->assign_scalar( .2, *next );
    bool refused = false;
    try
    {
        periodic_orbit::refresh_section_branch(
            s, *z, p, *next, p - .01, 2 * pi, *v, vp, 1e-6, [&]( const X & ) { return std::make_unique<epoch>( m ); },
            []( setup &, const auto & ) { return true; }, []( setup &, X &, auto & ) { return false; }
        );
    }
    catch ( const std::runtime_error &failure )
    {
        refused = std::string( failure.what() ) == "Previous orbit missed the replacement itinerary";
    }
    s.coords->add_lin_comb( -1, *z, *saved_z );
    s.coords->add_lin_comb( -1, *v, *saved_v );
    s.coords->assign_scalar( .2, *nv );
    s.coords->add_lin_comb( -1, *nv, *next );
    require(
        refused && s.coords->norm( *saved_z ) == 0 && s.coords->norm( *saved_v ) == 0 && vp == saved_vp &&
            s.coords->norm( *next ) == 0,
        "Failed rechart must preserve accepted history and oriented tangent"
    );
    // Exercise a successful transaction on all backends without running a long PDE/ODE branch.
    double previous_p = .99;
    s.coords->assign( *z, *next );
    require( solver.ordinary.solve( s.op.get(), *next, previous_p ), "Previous analytical orbit correction" );
    const double previous_period = 2 * pi / cycle{ previous_p }.omega();
    auto         fresh           = periodic_orbit::refresh_section_branch(
        s, *z, p, *next, previous_p, previous_period, *v, vp, 1e-6,
        [&]( const X & ) { return std::make_unique<epoch>( m ); },
        [&]( setup &f, const auto &root ) {
            check_point( f, root, p );
            return true;
        },
        [&]( setup &f, X &point, auto &out ) {
            using flight_type =
                periodic_orbit::section_map<operations, step<cycle>, plane, policy, rate_provider<cycle>>;
            auto rate          = std::make_shared<rate_provider<cycle>>();
            rate->model.lambda = previous_p;
            flight_type flight(
                f.physical, make_step( f.physical, .07, rate->model ), std::make_shared<policy>( f.physical ), {}, rate
            );
            std::vector<std::shared_ptr<chart>> charts;
            for ( const auto &segment : f.pilot.segments )
                charts.push_back( segment.chart );
            return periodic_orbit::rechart_orbit( flight, *f.pilot.layout, charts, point, previous_period, out );
        }
    );
    require(
        fresh && vp > 0 && guarded_memory::host_bytes == 0, "Successful rechart reversed direction or copied vectors"
    );
    for ( int i = 0; i < 3; ++i )
    {
        require(
            solver.driver.solve( s.op.get(), *z, p, *v, vp, *next, q, *nv, vq ) && q > p, "Pseudo-arclength advance"
        );
        check_point( s, *next, q );
        s.coords->assign( *next, *z );
        s.coords->assign( *nv, *v );
        p  = q;
        vp = vq;
    }
    // Fresh assembly/layout from the same deterministic pilot; resume from
    // committed coordinates, parameter, oriented tangent and fixed predictor step.
    setup  resumed( m );
    branch restart( resumed );
    auto   again = block_buffer( s.coords ), av = block_buffer( s.coords );
    auto   rz = block_buffer( resumed.coords ), rv = block_buffer( resumed.coords );
    double r = 0, vr = 0, rp = 0, rvp = 0;
    require( guarded_memory::host_bytes == 0, "Orbit preparation or correction transferred vectors" );
    checkpoint( s, *z, p, *v, vp, *rz, rp, *rv, rvp );
    guarded_memory::host_bytes = 0;
    require( restart.driver.solve( resumed.op.get(), *rz, rp, *rv, rvp, *again, r, *av, vr ), "Restart failed" );
    require( solver.driver.solve( s.op.get(), *z, p, *v, vp, *next, q, *nv, vq ), "Uninterrupted continuation failed" );
    s.coords->add_lin_comb( -1, *next, *again );
    s.coords->add_lin_comb( -1, *nv, *av );
    require(
        std::abs( r - q ) + std::abs( vr - vq ) + s.coords->norm( *again ) + s.coords->norm( *av ) < 1e-11,
        "Restart differs"
    );

    if ( m == 1 )
    {
        solver.pred.set_steps( .05, .05, .5 );
        *s.miss_next = true;
        require(
            solver.driver.solve( s.op.get(), *z, p, *v, vp, *next, q, *nv, vq ), "Retry after a missed crossing failed"
        );
        require(
            solver.driver.last_attempt_state().any_corrector_retries && q > p && q < r,
            "Missed crossing did not reduce the continuation step"
        );
        check_point( s, *next, q );
    }
    // omega=0 cannot reach the next oriented section. Never commit this trial.
    s.coords->assign( *z, *again );
    double failed = -3;
    require( !solver.ordinary.solve( s.op.get(), *again, failed ), "Missed section accepted" );
    s.coords->add_lin_comb( -1, *z, *again );
    require( s.coords->norm( *again ) == 0 && failed == -3, "Failed correction published a candidate" );
    s.op->set_linearization_point( *z, p );
    const auto Q         = s.op->make_return_operator();
    auto       direction = buffer( s.pilot.coordinates ), image = buffer( s.pilot.coordinates );
    s.pilot.coordinates->assign_scalar( 1, *direction );
    solver.op->prepare_tangent( *z, p, *v, vp );
    auto                                                J = solver.op->get_jacobi_operator();
    nmfd::detail::vector_wrap<pair_space, false, false> x( *solver.op->space() ), y( *solver.op->space() );
    x.init();
    x.start_use();
    y.init();
    y.start_use();
    solver.op->space()->assign_scalar( 1, *x );
    const auto allocated = guarded_memory::allocation_calls;
    for ( int repeat = 0; repeat < 3; ++repeat )
    {
        Q.apply( *direction, *image );
        J->apply( *x, *y );
    }
    require( allocated == guarded_memory::allocation_calls, "Repeated bordered/return action allocated vectors" );
    const auto multiplier = std::exp( -2 * p * 2 * pi / cycle{ p }.omega() );
    s.pilot.coordinates->add_lin_comb( -multiplier, *direction, *image );
    require( s.pilot.coordinates->norm( *image ) < 1e-8, "Transverse Floquet multiplier" );
    if ( m == 1 )
    {
        s.coords->assign_scalar( 1, *v );
        s.op->apply( *v, *nv );
        Q.apply( *direction, *image );
        s.pilot.coordinates->add_lin_comb( -1, *direction, *image );
        s.pilot.coordinates->add_lin_comb( -1, ( *nv ).comp( 0 ), *image );
        require( s.pilot.coordinates->norm( *image ) < 1e-13, "Matching is not Q-I" );
    }
}
void test()
{
    for ( auto m : { 1u, 4u } )
        check( m );
    require( guarded_memory::host_bytes == 0, "Continuation transferred vectors" );
}
}
int main()
{
    return section_tests::run( test );
}
