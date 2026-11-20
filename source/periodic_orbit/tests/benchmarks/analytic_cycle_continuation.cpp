#include "../parameterized_cycle.h"
#include "ode_continuation_benchmark.h"

int main( int argc, char **argv )
{
    using namespace ode_orbit_tests;
    return section_tests::run( [&] {
        const cycle_tests::parameterized_cycle model{};
        const auto settings = branch_settings::read( argc, argv, model.name );
        auto flights = [&] {
            return continuation_fixture(
                model, model.seed, model.period, settings.sections, settings.step, settings.start, settings.budget
            );
        };
        auto solver = [&]( auto &f ) {
            return orbit_benchmarks::branch_solver(
                f, settings.arclength_step, settings.newton_tolerance, settings.iterations
            );
        };
        branch_case checks( flights, solver, settings, origin_reference<2> );
        auto anchor = checks.correct_anchor( model.lambda );
        for ( int sign : { -1, 1 } ) checks.trace( *anchor, model.lambda, sign );
        checks.write_report();
    } );
}
