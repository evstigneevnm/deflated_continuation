#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include "common/iterative_factor_solver_bundle_test_suite.h"

int main()
{
    namespace test = stability::tests::iterative_factor_solver_bundle_test;
    test::run_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>( "serial" );
    test::run_backend<scfd::backend::omp<std::ptrdiff_t>>( "OMP" );
    return test::finish();
}
