#include "../lorenz_periodic_problem.h"
#include "ode_continuation_benchmark.h"

namespace ode_orbit_tests
{
// Continue in rho with the historical cubic damping; df/drho=(0,x,0).
struct lorenz_branch : lorenz_periodic
{
    lorenz_branch with_parameter( double value ) const
    {
        auto copy = *this;
        copy.rho  = value;
        return copy;
    }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i % 3 == 1 ? u[i - 1] : 0.;
    }
};
}

int main( int argc, char **argv )
{
    using namespace ode_orbit_tests;
    return section_tests::run( [&] {
        const lorenz_branch model{};
        const auto          settings     = branch_settings::read( argc, argv, model.name );
        auto                make_flights = [&] {
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
        auto        anchor = checks.correct_anchor( model.rho );
        checks.trace( *anchor, model.rho, -1 );
        checks.trace( *anchor, model.rho, +1 );
        checks.write_report();
    } );
}
