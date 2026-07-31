#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include "common/complex_affine_transformations_test_suite.h"

int main()
{
    namespace test =
        stability::tests::complex_affine_transformations_test;
    test::run_factorization_tests("host");
    test::run_backend<scfd::backend::serial_cpu>("serial");
    test::run_backend<scfd::backend::omp>("OMP");
    return test::finish();
}
