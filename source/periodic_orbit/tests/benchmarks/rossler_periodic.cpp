#include "../rossler_periodic_problem.h"
#include "ode_orbit_benchmark.h"

// Seeded correction of the historical Rossler formulation. Every report is
// independently checked by half/quarter-step integration and reference period.
int main( int argc, char **argv )
{
    return ode_orbit_tests::run_benchmark<ode_orbit_tests::rossler_periodic>( argc, argv );
}
