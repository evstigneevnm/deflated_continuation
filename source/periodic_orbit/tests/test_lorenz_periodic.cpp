#include "lorenz_periodic_problem.h"
#include "ode_orbit_test.h"

// Cubic-damped Lorenz short unstable cycle. Closure is checked over one period,
// not by requiring a long chaotic trajectory to shadow the reference.
int main()
{
    return section_tests::run( ode_orbit_tests::test_system<ode_orbit_tests::lorenz_periodic> );
}
