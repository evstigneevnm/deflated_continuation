#include <cstdlib>
#include <iostream>
#include <string>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

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

template<class Backend>
void run_backend(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    vector_space_type vector_space(3);
    stability_tests::run_tracked_invariant_subspace_test_suite(
        vector_space,
        label,
        require);
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu>("serial");
    run_backend<scfd::backend::omp>("OMP");
    std::cout
        << "Tracked invariant-subspace checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
