#include "vaidyanathan_periodic_problem.h"
#include "ode_orbit_test.h"

// Five-dimensional unstable cycle: analytical Jv, one/multiple-section Newton
// correction, refined closure, repeatability and backend-only vector storage.
int main()
{
    return section_tests::run( ode_orbit_tests::test_system<ode_orbit_tests::vaidyanathan_periodic> );
}
