#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include "common/matrix_free_factorized_krylov_schur_test_suite.h"

int main()
{
    namespace test =
        stability::tests::
            matrix_free_factorized_krylov_schur_test;
    test::run_backend<scfd::backend::serial_cpu>("serial");
    test::run_backend<scfd::backend::omp>("OMP");
    return test::finish();
}
