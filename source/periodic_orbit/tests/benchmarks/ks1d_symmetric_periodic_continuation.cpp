#include "ks1d_branch_flights.h"
#include "ode_continuation_benchmark.h"

int main( int argc, char **argv )
{
    using namespace ks1d_orbit;
    return section_tests::run( [&] {
        const auto s       = branch_settings::read( argc, argv, "ks1d_symmetric_periodic" );
        auto       flights = [&] { return branch_flights( s.sections, s.step, s.start, s.budget ); };
        auto       solver  = [&]( auto &f ) {
            return orbit_benchmarks::branch_solver( f, s.arclength_step, s.newton_tolerance, s.iterations );
        };
        branch_case checks( flights, solver, s, nullptr );
        auto        anchor = checks.correct_anchor( 68., true );
        for ( int direction : { -1, 1 } )
            checks.trace( *anchor, 68., direction );
        checks.write_report();
    } );
}
