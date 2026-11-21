#include "benchmarks/ks1d_orbit_support.h"
#include <iomanip>

namespace
{
using namespace ks1d_orbit;
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

double correct( const std::shared_ptr<operations> &space, std::size_t count, X &anchor )
{
    auto trial   = provider( space );
    auto initial = buffer( space );
    fill_profile( *initial, seed );
    auto layout   = make_layout( space, trial, *initial, period, count, dt, 0., budget );
    auto coords   = std::make_shared<operations>( seed.size() - 1 );
    auto blocks   = std::make_shared<matching::space_type>( coords, count );
    auto segments = make_segments( space, coords, layout, trial, dt, budget );
    auto problem  = std::make_shared<matching>( space, blocks, layout, segments, provider( space ) );
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
    auto       fine      = stepper( space, provider( space ), dt / 4 );
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
