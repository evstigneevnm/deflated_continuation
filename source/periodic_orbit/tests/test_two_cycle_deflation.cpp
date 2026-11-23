#include "benchmarks/ode_continuation_fixture.h"
#include "benchmarks/branch_archive_trace.h"
#include <periodic_orbit/orbit_collection.h>
#include <periodic_orbit/matching/chart_deflation.h>

namespace
{
using namespace ode_orbit_tests;
using periodic_orbit::orbit_admission;

// u' = g(|u|^2)u + omega*Ju. The positive isolated radii are 1
// (repelling) and 2 (attracting); both primitive periods are 2*pi/omega.
struct two_cycles
{
    static constexpr double anchor_parameter = 1;
    inline static constexpr std::array<double, 2> seed{ 1, 0 };
    double omega = 1;
    two_cycles with_parameter( double p ) const { return { p }; }
    __DEVICE_TAG__ static double g( double s ) { return .03*(s-1)*(4-s)/(1+s); }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    { return i ? u[0] : -u[1]; }
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
    { return g( u[0]*u[0]+u[1]*u[1] )*u[i] + omega*parameter_rhs( i, u, t ); }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double t ) const
    {
        const double s = u[0]*u[0]+u[1]*u[1], dg = .03*(-1+10/((1+s)*(1+s)));
        return g(s)*v[i] + 2*dg*(u[0]*v[0]+u[1]*v[1])*u[i] + omega*parameter_rhs( i, v, t );
    }
};

// Keep the same isolated radii but change angular speed from 1 at r=1
// to outer_speed at r=2. Section comparison needs only the physical RHS.
struct unequal_period_cycles
{
    double outer_speed;
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
    {
        const double s = u[0]*u[0] + u[1]*u[1];
        return two_cycles{ 1 + (outer_speed - 1)*(s - 1)/3 }.rhs( i, u, t );
    }
};

void compare_unequal_periods( double outer_speed )
{
    using flight = periodic_orbit::section_map<
        operations, step<unequal_period_cycles>, plane, policy, rate_provider<unequal_period_cycles>>;
    auto space = std::make_shared<operations>( 2 );
    auto u = buffer( space );
    auto rate = std::make_shared<rate_provider<unequal_period_cycles>>();
    rate->model = { outer_speed };
    flight map( space, make_step( space, .03, rate->model ), std::make_shared<policy>( space ), {}, rate );
    flight::request_type req;
    req.target = make_plane( space, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    periodic_orbit::orbit_comparison<operations, flight> cmp( space, map, outer_speed, req,
                                                               { 1e-6, 1e-6, 1e-6, 1e-5 } );
    periodic_orbit::orbit_collection catalog( cmp );
    const double slow = 2*pi, fast = slow/outer_speed;
    fill( *space, *u, 0, 1 );
    require( catalog.admit_verified( *u, outer_speed, slow ).decision == orbit_admission::inserted,
             "Slow cycle not admitted" );
    fill( *space, *u, 0, -2 );
    require( catalog.admit_verified( *u, outer_speed, fast ).decision == orbit_admission::inserted,
             "Separated cycle with a different verified period not admitted" );

    const auto allocations = guarded_memory::allocation_calls;
    const auto ab = cmp.compare( catalog.at(0), catalog.at(1) ), ba = cmp.compare( catalog.at(1), catalog.at(0) );
    require( ab.relation == periodic_orbit::orbit_relation::distinct && ba.relation == ab.relation &&
             ab.distance && std::abs( *ab.distance - 1 ) < 1e-6 && ba.distance == ab.distance,
             "Unequal-period comparison ignored the physical crossing distance" );
    fill( *space, *u, -2, 0 );
    const auto duplicate = catalog.admit_verified( *u, outer_speed, fast );
    require( duplicate.decision == orbit_admission::duplicate && duplicate.index == 1,
             "Phase-shifted fast orbit not recognized" );
    const auto twice = catalog.admit_verified( *u, outer_speed, 2*fast );
    require( twice.preparation.status == periodic_orbit::crossing_preparation::wrong_return_time &&
             twice.decision == orbit_admission::inconclusive && catalog.size() == 2,
             "Double traversal of the fast orbit was classified" );
    require( guarded_memory::allocation_calls == allocations, "Cached comparisons allocated backend vectors" );
}

void run_discovery( unsigned m )
{
    using setup = continuation_fixture<two_cycles>;
    using flight = periodic_orbit::section_map<operations, step<two_cycles>, plane, policy, rate_provider<two_cycles>>;
    using plain_op = nonlinear_operators::fixed_parameter_operator<setup::space, setup::orbit, setup::orbit>;
    constexpr double p = 1., h = .06, tol = 2e-10;
    constexpr unsigned budget = 1000;
    const double period = 2*pi;
    setup f( {}, two_cycles::seed, period, m, h, 0, budget );
    setup check( f.physical, f.layout, p, period, h, budget );
    auto u = buffer( f.physical ), scratch = buffer( f.physical );
    auto trial = f.workspace(), solved = f.workspace(), residual = f.workspace();
    auto solver = make_newton( f.coords, std::make_shared<plain_op>( f.op, f.op, p ), tol, 30 );
    flight map( f.physical, make_step<two_cycles>( f.physical, h ), std::make_shared<policy>( f.physical ),
                {}, std::make_shared<rate_provider<two_cycles>>() );
    flight::request_type req;
    req.target = make_plane( f.physical, 0, 0, 0, 1 );
    req.rule.minimum_crossing_speed = .01;
    periodic_orbit::orbit_comparison<operations, flight> cmp( f.physical, map, p, req, { 1e-6, 1e-6, 1e-6, 1e-5 } );
    periodic_orbit::orbit_collection catalog( cmp );
    auto rechart = [&]( const X &state, double T, setup::state &z ) {
        f.physical->assign( state, *scratch );
        return periodic_orbit::rechart_orbit( map, *f.layout, f.charts, *scratch, T, z );
    };
    auto freeze = [&] {
        return periodic_orbit::prepare_chart_deflation( catalog, f.coords, f.op, rechart, 1e-6, 1e-6, 1e-7 );
    };
    auto seed = [&]( double radius ) {
        for ( std::size_t i = 0; i < m; ++i )
        {
            f.physical->assign_mul( radius, f.layout->at(i).section->center(), *scratch );
            f.charts[i]->restrict( *scratch, (*trial).comp(i) );
        }
    };
    auto measure = [&] {
        const orbit_benchmarks::archive_trace_settings settings{ p, p, 0, budget, 1e-6 };
        auto v = orbit_benchmarks::orbit_measurement( check, *solved, p, *residual, *u, settings );
        require( v && v->valid() && std::abs( v->period-period ) < 1e-6, "Correction failed independent verification" );
        const double radius = f.physical->norm( *u );
        require( std::min( std::abs(radius-1), std::abs(radius-2) ) < 1e-6, "Not an analytical cycle" );
        return std::make_pair( radius < 1.5 ? 1 : 2, v->period );
    };
    // Zero coordinates describe the pilot orbit. Only an explicitly admitted
    // solution may be excluded, including when it happens to have z=0.
    f.coords->assign_scalar( 0., *trial );
    require( freeze().op->deflated_residual_norm( *trial, *residual ) < 1e-7, "Implicit zero root" );
    seed( 1.04 );
    require( solver->solve( *trial, *solved ), "Initial ordinary solve failed" );
    const auto first = measure();
    require( first.first == 1 && catalog.admit_verified( *u, p, first.second ).decision == orbit_admission::inserted,
             "Inner cycle was not stored" );
    auto frozen = freeze();
    require( bool(frozen.op), "Known cycle failed chart preparation" );
    require( !std::isfinite( frozen.op->deflated_residual_norm( *solved, *residual ) ), "Known root not excluded" );
    auto search = make_newton( f.coords, frozen.op, tol, 30, deflation::deflated_residual_norm{} );

    seed( 1.3 );
    auto dir = f.workspace(), plus = f.workspace(), minus = f.workspace(), jv = f.workspace();
    for ( unsigned i = 0; i < m; ++i ) f.coordinates->assign_scalar( .4-.17*i, (*dir).comp(i) );
    frozen.op->set_linearization_point( *trial );
    auto J = frozen.op->get_jacobi_operator();
    constexpr double eps = 2e-5;
    f.coords->assign_lin_comb( 1, *trial, eps, *dir, *solved );
    frozen.op->apply( *solved, *plus );
    f.coords->assign_lin_comb( 1, *trial, -eps, *dir, *solved );
    frozen.op->apply( *solved, *minus );
    require( J->apply( *dir, *jv ), "Prepared rank-one action failed" );
    f.coords->add_mul( -.5/eps, *plus, .5/eps, *minus, 1, *jv );
    require( f.coords->norm( *jv ) < 5e-5, "Rank-one action differs from centered differences" );

    bool switched = false;
    for ( double radius : { 1.15, 1.45, 1.7 } )
    {
        seed( radius );
        int ordinary = 0, deflated = 0;
        if ( periodic_orbit::try_orbit_solve<double>( [&]{ return solver->solve( *trial, *solved ); } ) )
            ordinary = measure().first;
        seed( radius );
        if ( periodic_orbit::try_orbit_solve<double>( [&]{ return search->solve( *trial, *solved ); } ) )
        {
            const auto found = measure();
            deflated = found.first;
            require( frozen.op->deflated_residual_norm( *solved, *residual ) <= tol, "Deflated residual failed" );
            const auto expected = catalog.size() == 1 ? orbit_admission::inserted : orbit_admission::duplicate;
            require( deflated == 2 && catalog.admit_verified( *u, p, found.second ).decision == expected,
                     "Deflation rediscovered inner root or misclassified outer root" );
        }
        switched |= ordinary == 1 && deflated == 2;
        auto outcome = []( int r ) { return r == 0 ? "not converged" : r == 1 ? "inner" : "outer"; };
        std::cout << "M=" << m << " seed radius=" << radius << " ordinary=" << outcome(ordinary)
                  << " deflated=" << outcome(deflated) << '\n';
    }
    require( switched && catalog.size() == 2, "No same-seed evidence for discovery by deflation" );
    for ( int radius : {1, 2} )
    {
        fill( *f.physical, *u, 0, radius );
        const auto duplicate = catalog.admit_verified( *u, p, period );
        require( duplicate.decision == orbit_admission::duplicate && duplicate.index == std::size_t(radius-1),
                 "Phase-shifted cycle not identified" );
    }
    require( catalog.admit_verified( *u, p, 2*period ).decision == orbit_admission::inconclusive &&
             catalog.size() == 2, "Double traversal changed the collection" );
    unsigned calls = 0;
    auto fail_second = [&]( const X &x, double T, setup::state &z ){ return ++calls == 1 && rechart(x,T,z); };
    const auto bad = periodic_orbit::prepare_chart_deflation( catalog, f.coords, f.op, fail_second, 1e-6, 1e-6, 1e-7 );
    require( !bad.op && bad.failed_root == 1 && bad.status == periodic_orbit::chart_root_status::rechart_failed,
             "Partially prepared chart published" );
    auto wrong = [&]( const X &, double, setup::state &z ) {
        seed( 1.3 );
        f.coords->assign( *trial, z );
        return true;
    };
    const auto gap = periodic_orbit::prepare_chart_deflation( catalog, f.coords, f.op, wrong, 1e-6, 1e-6, 1e-7 );
    require( !gap.op && gap.failed_root == 0 && gap.status == periodic_orbit::chart_root_status::matching_failed,
             "Accepted coordinates with a matching gap" );
    const auto both = freeze();
    require( bool(both.op), "Cannot deflate both cycles" );
    for ( const double radius : {1., 2.} )
    {
        seed( radius );
        require( !std::isfinite( both.op->deflated_residual_norm( *trial, *residual ) ), "Stored cycle not excluded" );
    }
}
}
int main()
{
    return section_tests::run( [] {
        for ( double ratio : {1.5, 2.} ) compare_unequal_periods( ratio );
        for ( unsigned m : {1u, 4u} ) run_discovery( m );
        require( guarded_memory::host_bytes == 0, "Orbit discovery transferred state vectors" );
    } );
}
