#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/copy/cuda.h>

#include <common/cuda_init_scfd.h>
#include <common/gpu_vector_operations.h>
#include <common/tests/vector_operations_template_tests.h>
#include <external_libraries/cublas_wrap.h>

namespace
{

struct gpu_vector_access
{
    template<class VecOps>
    void write(
        VecOps&,
        typename VecOps::vector_type& dst,
        const std::vector<typename VecOps::scalar_type>& src) const
    {
        scfd::cuda_copy<std::size_t>()(src.size(), src.data(), dst);
    }

    template<class VecOps>
    std::vector<typename VecOps::scalar_type> read(
        VecOps&,
        const typename VecOps::vector_type& src,
        std::size_t n) const
    {
        std::vector<typename VecOps::scalar_type> host(n);
        scfd::cuda_copy<std::size_t>()(n, src, host.data());
        return host;
    }
};

template<class T>
void run_gpu_type(
    const std::string& label,
    const std::vector<std::size_t>& sizes,
    cublas_wrap& cublas,
    vector_operations_tests::test_report& report)
{
    for(const auto n : sizes)
    {
        gpu_vector_operations<T> vec_ops(n, &cublas);
        vector_operations_tests::run_vector_operations_template_tests(vec_ops, gpu_vector_access{}, n, label + " n=" + std::to_string(n), report);
    }
}

}

int main(int argc, char** argv)
{
    vector_operations_tests::test_report report;

    const std::string device_selector = (argc > 1) ? argv[1] : "auto";
    const int device = common::init_cuda_from_scfd_selector(device_selector);
    std::cout << "CUDA device: " << device << std::endl;

    cublas_wrap cublas(true);

    const std::vector<std::size_t> sizes = {1, 7, 64, 1025};
    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;

    run_gpu_type<real>("GPU real", sizes, cublas, report);
    run_gpu_type<complex>("GPU complex", sizes, cublas, report);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
