#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include "common/projected_spectrum_recovery_test_suite.h"

int main()
{
    namespace test =
        stability::tests::projected_spectrum_recovery_test;
    test::run_backend<scfd::backend::serial_cpu>("serial");
    test::run_backend<scfd::backend::omp>("OMP");
    return test::finish();
}
