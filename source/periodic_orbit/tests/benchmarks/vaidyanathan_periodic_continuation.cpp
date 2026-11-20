#include "../vaidyanathan_periodic_problem.h"
#include "ode_continuation_benchmark.h"

namespace ode_orbit_tests
{
// Continue the seeded unstable five-dimensional cycle in c. Only
// y' = c*x - x*z - y depends on c; the origin remains an equilibrium.
struct vaidyanathan_branch : vaidyanathan_periodic
{
    vaidyanathan_branch with_parameter( double value ) const
    {
        auto copy = *this;
        copy.c = value;
        return copy;
    }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i % 5 == 1 ? u[i - 1] : 0.;
    }
};
}

int main( int argc, char **argv )
{
    using namespace ode_orbit_tests;
    return section_tests::run( [&] {
        const vaidyanathan_branch model{};
        const auto settings = branch_settings::read( argc, argv, model.name );
        auto make_flights = [&] {
            return continuation_fixture(
                model, model.seed, model.period, settings.sections, settings.step, settings.start, settings.budget
            );
        };
        auto make_solver = [&]( auto &flights ) {
            return orbit_benchmarks::branch_solver(
                flights, settings.arclength_step, settings.newton_tolerance, settings.iterations
            );
        };
        branch_case checks( make_flights, make_solver, settings, origin_reference<5> );
        auto anchor = checks.correct_anchor( model.c );
        checks.trace( *anchor, model.c, -1 );
        checks.trace( *anchor, model.c, +1 );
        checks.write_report();
    } );
}
