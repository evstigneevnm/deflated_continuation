#include "../lorenz_periodic_problem.h"
#include "ode_orbit_benchmark.h"

// Seeded correction of a short cubic-damped Lorenz orbit. No claim of general
// chaotic orbit discovery or long-time trajectory agreement is made.
int main( int argc, char **argv )
{
    return ode_orbit_tests::run_benchmark<ode_orbit_tests::lorenz_periodic>( argc, argv );
}
