#include <exception>
#include <iostream>
#include <string>

#include <common/cuda_init_scfd.h>
#include <scfd/backend/cuda.h>

#include <time_stepper/tests/common/explicit_rk_engine_test_suite.h>

int main(int argc, char** argv)
{
    try
    {
        const std::string device_selector = argc > 1 ? argv[1] : "auto";
        common::init_cuda_from_scfd_selector(device_selector);
        return time_steppers::tests::run_explicit_rk_engine_tests<scfd::backend::cuda>(
            "scfd_cuda");
    }
    catch(const std::exception& error)
    {
        std::cerr << "Explicit RK CUDA test failed: " << error.what() << '\n';
        return 1;
    }
}
