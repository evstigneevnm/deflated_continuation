#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/serial_cpu.h>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
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
        vec_ops.set(src.data(), dst);
    }

    template<class VecOps>
    std::vector<typename VecOps::scalar_type> read(
        VecOps& vec_ops,
        const typename VecOps::vector_type& src,
        std::size_t n) const
    {
        std::vector<typename VecOps::scalar_type> host(n);
        vec_ops.get(src, host.data());
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
        scfd_vector_operations<Backend, T> vec_ops(n);
        vector_operations_tests::run_vector_operations_template_tests(
            vec_ops,
            scfd_vector_access{},
            n,
            label + " n=" + std::to_string(n),
            report);
    }
}

} // namespace

int main()
{
    vector_operations_tests::test_report report;

    using real = SCALAR_TYPE;
    using complex = common::scfd_backend_ext::complex_t<scfd::backend::serial_cpu, real>;

    const std::vector<std::size_t> sizes = {1, 7, 64, 1025};
    run_scfd_type<scfd::backend::serial_cpu, real>("SCFD serial real", sizes, report);
    run_scfd_type<scfd::backend::serial_cpu, complex>("SCFD serial complex", sizes, report);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
