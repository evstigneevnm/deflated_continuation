#include "ode_orbit_support.h"
#include "benchmarks/ks1d_evolution.h"
#include <iomanip>
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <external_libraries/fft_facade_cufft.h>
using fft = external_libraries::fft::cufft_backend;
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <external_libraries/fft_facade_hipfft.h>
using fft = external_libraries::fft::hipfft_backend;
#else
#    include <external_libraries/fft_facade_fftw.h>
using fft = external_libraries::fft::fftw_backend;
#endif

namespace
{
using namespace ode_orbit_tests;
using rate     = ks1d_orbit::evolution<operations, fft>;
using rk       = nmfd::time_steppers::runge_kutta::explicit_time_step<operations, rate, fixed, true>;
using map      = periodic_orbit::recorded_section_map<operations, rk, plane, rate>;
using matching = periodic_orbit::multiple_section_problem<operations, operations, map, rate>;

auto stepper( const std::shared_ptr<operations> &space, const std::shared_ptr<rate> &rhs, double dt )
{
    return std::make_shared<rk>( space, rhs, std::make_shared<fixed>( fixed::params{ dt } ), rk::params{ "RK45" } );
}

// Suppress advection: the two occupied modes have exact growth rates
// lambda*k^2-b*k^4, respectively +2 and -4. This also fixes the time unit.
void check_linear_modes()
{
    auto space   = std::make_shared<operations>( 3 );
    auto rhs     = std::make_shared<rate>( space, 8, 0., 1., 3. );
    auto initial = buffer( space ), result = buffer( space ), exact = buffer( space );
    fill_profile( *initial, std::array<double, 3>{ 1., 1., 0. } );
    using integrator = nmfd::time_steppers::integration::time_integrator<operations, rk>;
    integrator run( space, stepper( space, rhs, .001 ), integrator::params{ 64 } );
    run.set_time_interval( .2, .25 );
    run.apply( *initial, *result );
    require( run.get_status() == nmfd::time_steppers::integration_status::completed, "KS linear integration failed" );
    fill_profile( *exact, std::array<double, 3>{ std::exp( .1 ), std::exp( -.2 ), 0. } );
    space->add_lin_comb( -1, *exact, *result );
    require( space->norm( *result ) < 1e-12, "Wrong physical growth/decay of KS modes" );
}

void check_derivatives()
{
    const double h     = 1e-5;
    auto         space = std::make_shared<operations>( 5 );
    rate         rhs( space, 12, 2., 4., 68. ), plus( space, 12, 2., 4., 68 + h ), minus( space, 12, 2., 4., 68 - h );
    auto         u = buffer( space ), v = buffer( space ), trial = buffer( space );
    auto         left = buffer( space ), right = buffer( space ), expected = buffer( space ), actual = buffer( space );
    fill_profile( *u, std::array<double, 5>{ 4., -2., 3., .4, -.8 } );
    fill_profile( *v, std::array<double, 5>{ .5, -.3, .8, .2, .1 } );
    auto compare = [&] {
        space->assign_lin_comb( .5 / h, *right, -.5 / h, *left, *expected );
        space->add_lin_comb( -1, *actual, *expected );
        require( space->norm( *expected ) < 2e-8 * space->norm( *actual ), "KS derivative finite difference" );
    };
    auto exercise = [&] {
        space->assign_lin_comb( 1, *u, h, *v, *trial );
        rhs.evaluate( 0., *trial, *right );
        space->assign_lin_comb( 1, *u, -h, *v, *trial );
        rhs.evaluate_rate( *trial, *left );
        rhs.apply_jacobian( 0., *u, *v, *actual );
        compare();
        plus.evaluate_rate( *u, *right );
        minus.evaluate_rate( *u, *left );
        rhs.parameter_derivative( 0., *u, *actual );
        compare();
    };
    exercise();
    const auto allocations = guarded_memory::allocation_calls;
    space->add_lin_comb( .5, *v, *u ); // Same stage buffer, different base state.
    exercise();
    require( guarded_memory::allocation_calls == allocations, "Repeated KS actions allocated backend vectors" );
}

// Seed of the N=24 odd, 2*pi-periodic semidiscretization, in unnormalized
// imaginary R2C coefficients. Independent LSODA integration, sampled at a
// maximum of mode 1. This is a seed, not a spatially converged PDE reference.
constexpr std::array<double, 11> seed{ 3.4713118913,  -3.0269991860, 52.9326530721, 4.7307271608,
                                       -3.2242998383, 15.8220336620, 1.7723260826,  -.7814574934,
                                       1.8946420033,  .3659630652,   -.1943024199 };
constexpr double                 period = .015153690425, dt = .00002;
constexpr std::size_t            budget = 4096;

double correct( const std::shared_ptr<operations> &space, std::size_t count, X &anchor )
{
    auto provider = [&] { return std::make_shared<rate>( space, 24, 2., 4., 68. ); };
    auto trial    = provider();
    auto initial  = buffer( space );
    fill_profile( *initial, seed );
    periodic_orbit::automatic_layout<operations, rk, rate> builder(
        space, stepper( space, trial, dt ), trial, std::make_shared<policy>( space ),
        std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( space )
    );
    periodic_orbit::placement_params<double> p;
    p.sections                  = count;
    p.maximum_pilot_steps       = budget;
    p.maximum_endpoint_distance = 1e-5;
    const auto pilot            = builder.build( *initial, { 0., period, nullptr }, p );
    require( pilot.succeeded(), "KS pilot layout/itinerary failed" );

    auto                           coords = std::make_shared<operations>( seed.size() - 1 );
    auto                           blocks = std::make_shared<matching::space_type>( coords, count );
    std::vector<matching::segment> segments;
    map::map_type::params          flight;
    flight.maximum_steps = budget;
    for ( std::size_t i = 0; i < count; ++i )
        segments.push_back(
            { std::make_shared<map>( space, stepper( space, trial, dt ), trial, flight ),
              std::make_shared<chart>( space, coords, pilot.layout->at( i ).section ) }
        );
    auto problem = std::make_shared<matching>( space, blocks, pilot.layout, segments, provider() );
    nmfd::detail::vector_wrap<matching::space_type, true, true> guess( *blocks ), solution( *blocks ),
        residual( *blocks );
    perturb_sections<seed.size()>( .001, *guess );
    problem->apply( *guess, *residual );
    require( blocks->norm( *residual ) > 1e-5, "KS perturbation did not exercise correction" );
    auto solver = make_newton( blocks, problem );
    require( solver->solve( *guess, *solution ), "KS multiple-section Newton failed" );
    problem->apply( *solution, *residual );
    const auto gap = blocks->norm( *residual ), T = problem->report().period;
    require( gap < 1e-10 && std::abs( T - period ) < 1e-8, "KS matching residual or period" );
    segments.front().chart->lift( ( *solution ).comp( 0 ), anchor );
    auto       fine      = stepper( space, provider(), dt / 4 );
    const auto excursion = closure_error( space, fine, anchor, 0., T / 2, budget );
    const auto closure   = closure_error( space, fine, anchor, 0., T, budget );
    std::cout << std::setprecision( 12 ) << "M=" << count << " period=" << T << " gap=" << gap
              << " refined closure=" << closure << " half-period displacement=" << excursion << '\n';
    require( excursion > 1. && closure < 1e-7, "KS half/full-period independent verification" );
    return T;
}
}

int main( int argc, char **argv )
{
    return section_tests::run( [&] {
        if ( argc > 2 || ( argc == 2 && std::string( argv[1] ) != "--rates-only" ) )
            throw std::invalid_argument( "Usage: test_ks1d_symmetric_periodic [--rates-only]" );
        check_linear_modes();
        check_derivatives();
        if ( argc == 2 )
            return;
        auto       space = std::make_shared<operations>( seed.size() );
        auto       one = buffer( space ), four = buffer( space );
        const auto t1 = correct( space, 1, *one ), t4 = correct( space, 4, *four );
        space->add_lin_comb( -1, *one, *four );
        require(
            std::abs( t1 - t4 ) < 1e-9 && space->norm( *four ) < 1e-7, "Section counts found different KS orbits"
        );
        require( guarded_memory::host_bytes == 0, "KS orbit test transferred vector data" );
    } );
}
