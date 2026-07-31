#include <iostream>
#include <string>

#include <scfd/backend/cuda.h>

#include <common/cuda_init_scfd.h>

#include "common/complex_affine_transformations_test_suite.h"

int main(int argc, char** argv)
{
    const std::string device_selector =
        argc > 1 ? argv[1] : "auto";
    const int device =
        common::init_cuda_from_scfd_selector(device_selector);
    std::cout << "CUDA device: " << device << '\n';

    namespace test =
        stability::tests::complex_affine_transformations_test;
    test::run_factorization_tests("host");
    test::run_backend<scfd::backend::cuda>("CUDA");
    return test::finish();
}
