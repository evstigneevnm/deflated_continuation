#include <cstdlib>
#include <iostream>
#include <string>

#include <scfd/backend/cuda.h>

#include <common/cuda_init_scfd.h>
#include <common/scfd_vector_operations.h>

#include "common/tracked_invariant_subspace_test_suite.h"

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    const std::string device_selector =
        argc > 1 ? argv[1] : "auto";
    const int device =
        common::init_cuda_from_scfd_selector(device_selector);
    std::cout << "CUDA device: " << device << '\n';

    using vector_space_type =
        scfd_vector_operations<scfd::backend::cuda, double>;
    vector_space_type vector_space(3);
    stability_tests::run_tracked_invariant_subspace_test_suite(
        vector_space,
        "CUDA",
        require);
    std::cout
        << "Tracked invariant-subspace checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
