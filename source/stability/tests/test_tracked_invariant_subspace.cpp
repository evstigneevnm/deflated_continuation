#include <cstdlib>
#include <iostream>
#include <string>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <nmfd/operations/scfd_vector_space.h>

#include "common/tracked_invariant_subspace_test_suite.h"

namespace
{

std::size_t checks   = 0;
std::size_t failures = 0;

void require( bool condition, const std::string &message )
{
    ++checks;
    if ( !condition )
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template <class Backend>
void run_backend( const std::string &label )
{
    using vector_space_type = nmfd::operations::scfd_vector_space<Backend, double>;
    vector_space_type vector_space( 3 );
    stability_tests::run_tracked_invariant_subspace_test_suite( vector_space, label, require );
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( "serial" );
    run_backend<scfd::backend::omp<std::ptrdiff_t>>( "OMP" );
    std::cout << "Tracked invariant-subspace checks: " << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
