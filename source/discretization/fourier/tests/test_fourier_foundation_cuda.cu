#include <exception>
#include <iostream>

#include <common/cuda_init_scfd.h>
#include <external_libraries/fft_facade_cufft.h>
#include <scfd/backend/cuda.h>

#include <discretization/fourier/tests/fourier_foundation_test_suite.h>

int main()
{
    try
    {
        common::init_cuda_from_scfd_selector("auto");
        return discretization::fourier::tests::run_fourier_foundation_tests<
            scfd::backend::cuda,
            external_libraries::fft::cufft_backend
        >("scfd_cuda_cufft");
    }
    catch(const std::exception& error)
    {
        std::cerr << "Fourier CUDA foundation test failed: " << error.what() << std::endl;
        return 1;
    }
}
