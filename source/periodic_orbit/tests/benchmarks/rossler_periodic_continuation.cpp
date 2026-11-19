#include "../rossler_periodic_problem.h"
#include "ode_continuation_benchmark.h"

namespace ode_orbit_tests
{
// Continue in c at fixed a=b=0.1; df/dc=(0,0,-z).
struct rossler_branch : rossler_periodic
{
    rossler_branch with_parameter( double value ) const
    {
        auto copy = *this;
        copy.c    = value;
        return copy;
    }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i % 3 == 2 ? -u[i] : 0.;
    }
};
}

int main( int argc, char **argv )
{
    using namespace ode_orbit_tests;
    return section_tests::run( [&] {
        const rossler_branch model{};
        const auto           settings     = branch_settings::read( argc, argv, model.name );
        auto                 make_flights = [&] {
            return continuation_fixture(
                model, model.seed, model.period, settings.sections, settings.step, settings.start, settings.budget
            );
        };
        auto make_solver = [&]( auto &flights ) {
            return orbit_benchmarks::branch_solver(
                flights, settings.arclength_step, settings.newton_tolerance, settings.iterations
            );
        };
        branch_case checks( make_flights, make_solver, settings );
        auto        anchor = checks.correct_anchor( model.c );
        // Independent periodic collocation, continued from the same anchor beyond both original layout failures.
        checks.trace( *anchor, model.c, -1, branch_reference{ .4, 5.53393796 } );
        checks.trace( *anchor, model.c, +1 );
        checks.write_report();
    } );
}
