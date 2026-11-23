#include "benchmarks/branch_solver.h"
#include "benchmarks/initial_tangent_solver.h"
#include "benchmarks/ks1d_branch_flights.h"
#include "benchmarks/branch_restart_case.h"
#include <containers/knots.hpp>
#include <continuation/continuation.hpp>

namespace
{
using namespace ks1d_orbit;
using corrections   = orbit_benchmarks::branch_corrections<branch_flights>;
using space         = branch_flights::space;
using orbit         = branch_flights::orbit;
using state         = branch_flights::state;
using reason        = container::curve_endpoint_reason;
using block_buffer  = nmfd::detail::vector_wrap<space, true, true>;
constexpr double p0 = 68., lower = 67.999, upper = 68.001, knot = 68.0005;

// Independent step-level RK45 benchmark, M=1, common-parameter probes.
// Compare the lifted Fourier state, not section coordinates or just an amplitude.
struct reference_point
{
    double                          parameter, period;
    std::array<double, seed.size()> coefficients;
};
constexpr reference_point reference[] = {
    { lower,
      .01515307799761766,
      { 3.4673219313511034, -3.025633467841686, 52.9335843425138, 4.731334841560447, -3.2241249956955165,
        15.822294020637356, 1.7723762302340096, -.7813868478525984, 1.8946804477848374, .36597388838684525,
        -.1942978551899088 } },
    { upper,
      .01515430221001677,
      { 3.475290144782186, -3.0283605619025598, 52.93172198242308, 4.730121294323739, -3.2244738085765574,
        15.821773615337458, 1.7722760988546171, -.7815278436413722, 1.8946036226157676, .3659522709150246,
        -.19430695839266474 } }
};

using state_solver = orbit_benchmarks::initial_tangent_solver<space, orbit>;

struct curve
{
    struct point
    {
        double parameter, norm;
        reason endpoint;
    };
    space                            &ops;
    bool                              capped;
    std::size_t                       segments = 0, samples = 0;
    std::array<std::vector<point>, 2> points;
    std::array<block_buffer, 2>       last;
    block_buffer                      sample;
    curve( space &s, bool cap = false )
        : ops( s ), capped( cap ), last{ block_buffer( s ), block_buffer( s ) }, sample( s )
    {
    }
    void start_new_segment()
    {
        require( ++segments <= 2, "Unexpected third semicurve" );
    }
    std::uint64_t get_current_segment_id() const
    {
        return segments;
    }
    void add( double p, const state &z, bool forced, reason why )
    {
        if ( p == knot && forced && why == reason::none )
        {
            ++samples;
            ops.assign( z, *sample );
            return;
        }
        points.at( segments - 1 ).push_back( { p, ops.norm( z ), why } );
        // In the capped run only one non-seed point is accepted per arm.
        if ( why != reason::none || ( capped && p != p0 ) )
            ops.assign( z, *last.at( segments - 1 ) );
    }
    void set_last_endpoint_reason( reason why )
    {
        points.at( segments - 1 ).back().endpoint = why;
    }
    void print_curve() const
    {
    }
};

using driver_type = continuation::continuation<
    space, void, corrections::alg::log, orbit, orbit, container::knots<double>, state_solver,
    corrections::fixed_correction_type, curve, continuation::system_operator_continuation, corrections>;

void trace( branch_flights &f, corrections &solvers, const state &anchor, curve &out, bool with_knot, double ds )
{
    state_solver             linear( f.coords );
    container::knots<double> knots;
    if ( with_knot )
        knots.add_element( knot );
    driver_type driver(
        f.coords.get(), nullptr, &solvers.log, f.op.get(), f.op.get(), &knots, &linear, &solvers.ordinary,
        solvers.coupled, solvers.tan
    );
    driver.set_parameter_bounds( lower, upper );
    driver.set_corrector_controls( 1e-10, false );
    driver.set_steps( out.capped ? 1 : 40, ds, ds, -1 );
    auto      *sink   = &out;
    const auto bytes  = guarded_memory::host_bytes;
    const auto result = driver.continuate_curve_result( sink, anchor, p0 );
    require( guarded_memory::host_bytes == bytes, "Host copy during continuation driver call" );
    require( linear.successful > 0, "Seed tangent did not exercise the state-space GMRES" );
    require( result.semicurves_started == 2 && out.segments == 2 && !result.branch_closed, "Expected two open arms" );
    require( result.complete() == !out.capped, "Wrong branch completion status" );
    require( out.samples == ( with_knot ? 1u : 0u ), "Missing or unexpected observational sample" );
    for ( unsigned i = 0; i < 2; ++i )
    {
        const auto &arm    = result.semicurves[i];
        const auto &points = out.points[i];
        const auto  why    = out.capped ? reason::max_steps : ( i == 0 ? reason::boundary_min : reason::boundary_max );
        require(
            arm.direction == ( i == 0 ? -1 : 1 ) && arm.start_parameter == p0 && !points.empty() &&
                points.front().parameter == p0,
            "Incorrect semicurve initialization"
        );
        require( arm.endpoint_reason == why && points.back().endpoint == why, "Unexpected endpoint reason" );
        if ( !out.capped )
        {
            require(
                arm.complete() && points.size() >= 4 && points.back().parameter == reference[i].parameter,
                "Did not reach the corrected boundary after ordinary steps"
            );
            continue;
        }
        auto   z = f.workspace(), tangent = f.workspace();
        double p = 0, dp = 0;
        require(
            arm.recoverable() && arm.failure == continuation::continuation_failure_kind::maximum_steps &&
                points.size() == 2 && driver.copy_recovery_checkpoint( i, *z, *tangent, p, dp ),
            "Missing recoverable step-limit checkpoint"
        );
        f.coords->add_lin_comb( -1., *out.last[i], *z );
        require(
            p != p0 && p == points.back().parameter && p == arm.last_parameter && f.coords->norm( *z ) < 1e-12 &&
                dp * arm.direction > 0 && std::abs( f.coords->norm_rank1( *tangent, dp ) - 1. ) < 1e-8,
            "Recovery checkpoint is not the last accepted state and oriented tangent"
        );
    }
}

void check( std::size_t m )
{
    branch_flights f( m, dt, 0., budget );
    corrections    solvers( f );
    auto           anchor = f.workspace(), residual = f.workspace();
    perturb_sections<seed.size()>( .001, *anchor );
    require( solvers.ordinary.solve( f.op.get(), *anchor, p0 ), "Seed correction failed" );
    // The two chart dimensions give different d(lambda)/ds. Bracket the knot
    // after the driver's first two steps, and before crossing the boundary.
    const double ds = m == 1 ? .001 : .003;
    curve        plain( *f.coords ), observed( *f.coords ), capped( *f.coords, true );
    trace( f, solvers, *anchor, plain, false, ds );
    trace( f, solvers, *anchor, observed, true, ds );
    trace( f, solvers, *anchor, capped, false, ds );
    f.op->F( *observed.sample, knot, *residual );
    require( f.coords->norm( *residual ) < 1e-9, "Observational knot is not a corrected orbit" );
    auto u = buffer( f.physical ), ref = buffer( f.physical );
    for ( unsigned i = 0; i < 2; ++i )
    {
        const auto &a = plain.points[i], &b = observed.points[i];
        require( a.size() == b.size(), "Knot changed the number of ordinary points" );
        for ( std::size_t j = 0; j < a.size(); ++j )
            require(
                std::abs( a[j].parameter - b[j].parameter ) < 1e-12 && std::abs( a[j].norm - b[j].norm ) < 1e-10,
                "Knot modified the continuation trajectory"
            );
        f.coords->assign_mul( 1., *plain.last[i], -1., *observed.last[i], *residual );
        require( f.coords->norm( *residual ) < 1e-10, "Knot changed the boundary state" );
        const auto &target = reference[i];
        f.op->F( *observed.last[i], target.parameter, *residual );
        const double T = f.op->report().period;
        require( f.coords->norm( *residual ) < 1e-9 && std::abs( T - target.period ) < 1e-9, "Boundary gap or period" );
        f.first->lift( ( *observed.last[i] ).comp( 0 ), *u );
        fill_profile( *ref, target.coefficients );
        f.physical->add_lin_comb( -1., *u, *ref );
        const double delta   = f.physical->norm( *ref );
        const double closure = closure_error( f.physical, f.verification_step( target.parameter ), *u, 0., T, budget );
        std::cout << "M=" << m << " lambda=" << target.parameter << " state difference=" << delta
                  << " refined closure=" << closure << '\n';
        require(
            delta < 1e-7 && closure < 1e-7, "Driver endpoint disagrees with reference or fails independent closure"
        );
    }
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        if ( argc != 1 )
        {
            if ( argc != 3 || (std::string( argv[1] ) != "--write" && std::string( argv[1] ) != "--restart") )
                throw std::invalid_argument( "Expected --write DIR or --restart DIR" );
            for ( const auto m : { 1u, 4u } )
                orbit_benchmarks::branch_restart_case<branch_flights>(
                    std::filesystem::path( argv[2] ) / ("ks-M" + std::to_string( m )), std::string( argv[1] ) == "--restart",
                    { "ks1d-M" + std::to_string( m ), "symmetric_KS1D", "N24:odd-R2C:a2:b4", "lambda", "l2_norm", "rms" },
                    m, p0, dt, { lower, upper, m == 1 ? .003 : .009, budget, 1e-7 },
                    [=] { return std::make_unique<branch_flights>( m, dt, 0., budget ); } );
            return;
        }
        check( 1 );
        check( 4 );
    } );
}
