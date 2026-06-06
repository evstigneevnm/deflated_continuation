#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/hip.h>

#include <common/hip_init_scfd.h>
#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
#include <common/tests/scfd_vector_operations_nmfd_interface_tests.h>
#include <common/tests/vector_operations_template_tests.h>

namespace
{

struct scfd_vector_access
{
    template<class VecOps>
    void write(
        VecOps& vec_ops,
        typename VecOps::vector_type& dst,
        const std::vector<typename VecOps::scalar_type>& src) const
    {
        vec_ops.set(src.data(), dst, src.size());
    }

    template<class VecOps>
    std::vector<typename VecOps::scalar_type> read(
        VecOps& vec_ops,
        const typename VecOps::vector_type& src,
        std::size_t n) const
    {
        std::vector<typename VecOps::scalar_type> host(n);
        vec_ops.get(src, host.data(), n);
        return host;
    }
};

template<class Backend, class T>
void run_scfd_type(
    const std::string& label,
    const std::vector<std::size_t>& sizes,
    vector_operations_tests::test_report& report)
{
    for(const auto n : sizes)
    {
        std::cout << label << " n=" << n << std::endl;
        scfd_vector_operations<Backend, T> vec_ops(n);
        vector_operations_tests::run_vector_operations_template_tests(
            vec_ops,
            scfd_vector_access{},
            n,
            label + " n=" + std::to_string(n),
            report);
        vector_operations_tests::run_nmfd_vector_space_interface_tests(
            vec_ops,
            scfd_vector_access{},
            n,
            label + " n=" + std::to_string(n),
            report);
    }
}

} // namespace

int main(int argc, char** argv)
{
    vector_operations_tests::test_report report;
    const std::string device_selector = (argc > 1) ? argv[1] : "auto";
    const std::string test_selector = (argc > 2) ? argv[2] : "all";

    using real = SCALAR_TYPE;
    using hip_complex = common::scfd_backend_ext::complex_t<scfd::backend::hip, real>;

    const int device = common::init_hip_from_scfd_selector(device_selector);
    std::cout << "HIP device: " << device << std::endl;

    const std::vector<std::size_t> sizes = {1, 2, 7, 31, 32, 33, 64, 1025, 4097};
    if(test_selector == "all" || test_selector == "real")
    {
        run_scfd_type<scfd::backend::hip, real>("SCFD HIP real", sizes, report);
    }
    if(test_selector == "all" || test_selector == "complex")
    {
        run_scfd_type<scfd::backend::hip, hip_complex>("SCFD HIP complex", sizes, report);
    }

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
