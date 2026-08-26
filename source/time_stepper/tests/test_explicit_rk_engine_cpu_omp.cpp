#include <scfd/backend/omp.h>

#include <time_stepper/tests/common/explicit_rk_engine_test_suite.h>

int main()
{
    return time_steppers::tests::run_explicit_rk_engine_tests<scfd::backend::omp>(
        "scfd_omp");
}
