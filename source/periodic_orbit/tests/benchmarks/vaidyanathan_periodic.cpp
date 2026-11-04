#include "../vaidyanathan_periodic_problem.h"
#include "ode_orbit_benchmark.h"

// Seeded unstable-orbit correction in the five-dimensional chaotic formulation.
// All five components enter matching, closure and independent refinement checks.
int main( int argc, char **argv )
{
    return ode_orbit_tests::run_benchmark<ode_orbit_tests::vaidyanathan_periodic>( argc, argv );
}
