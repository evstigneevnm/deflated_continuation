#include "rossler_periodic_problem.h"
#include "ode_orbit_test.h"

// Nontrivial Rossler cycle: correction, section derivatives and independent
// closure, for one and multiple Poincare sections on the selected backend.
int main()
{
    return section_tests::run( ode_orbit_tests::test_system<ode_orbit_tests::rossler_periodic> );
}
