#include <external_libraries/fft_facade_fftw.h>
#include <scfd/backend/omp.h>

#include <discretization/fourier/tests/fourier_foundation_test_suite.h>

int main()
{
    return discretization::fourier::tests::run_fourier_foundation_tests<
        scfd::backend::omp,
        external_libraries::fft::fftw_backend
    >("scfd_omp_fftw");
}
