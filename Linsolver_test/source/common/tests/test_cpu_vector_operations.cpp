#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <common/cpu_vector_operations.h>
#include <common/tests/vector_operations_template_tests.h>

namespace
{

struct cpu_vector_access
{
    template<class VecOps>
    void write(
        VecOps&,
        typename VecOps::vector_type& dst,
        const std::vector<typename VecOps::scalar_type>& src) const
    {
        dst = src;
    }

    template<class VecOps>
    std::vector<typename VecOps::scalar_type> read(
        VecOps&,
        const typename VecOps::vector_type& src,
        std::size_t) const
    {
        return src;
    }
};

template<class T>
void run_slice_tests(vector_operations_tests::test_report& report)
{
    using vec_ops_t = cpu_vector_operations<T>;
    using vec_t = typename vec_ops_t::vector_type;

    vec_ops_t vec_ops(10);

    vec_t x;
    vec_t y;
    vec_t z;
    vec_ops.init_vectors(x, y, z);
    vec_ops.start_use_vector(x, 10);
    vec_ops.start_use_vector(y, 3);
    vec_ops.start_use_vector(z, 9);

    for(std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] = static_cast<T>(i);
    }

    vec_ops.assign_slices(x, {{3, 4}, {6, 8}}, y);
    vec_ops.assign_skip_slices(x, {{5, 6}}, z);

    const vec_t expected_y = {T(3), T(6), T(7)};
    const vec_t expected_z = {T(0), T(1), T(2), T(3), T(4), T(6), T(7), T(8), T(9)};

    vector_operations_tests::check_vector_close(report, "CPU slice assign_slices", y, expected_y);
    vector_operations_tests::check_vector_close(report, "CPU slice assign_skip_slices", z, expected_z);

    vec_ops.stop_use_vectors(x, y, z);
    vec_ops.free_vectors(x, y, z);
}

template<class T>
void run_cpu_type(const std::string& label, const std::vector<std::size_t>& sizes, vector_operations_tests::test_report& report)
{
    for(const auto n : sizes)
    {
        cpu_vector_operations<T> vec_ops(n);
        vector_operations_tests::run_vector_operations_template_tests(vec_ops, cpu_vector_access{}, n, label + " n=" + std::to_string(n), report);
    }
}

}

int main()
{
    vector_operations_tests::test_report report;

    const std::vector<std::size_t> sizes = {1, 7, 64};
    using real = SCALAR_TYPE;

    run_cpu_type<real>("CPU real", sizes, report);
    run_slice_tests<real>(report);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
