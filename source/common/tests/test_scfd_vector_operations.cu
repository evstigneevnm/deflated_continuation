#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/cuda.h>
#include <scfd/backend/serial_cpu.h>

#include <common/cuda_init_scfd.h>
#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
#include <common/tests/scfd_vector_operations_high_precision_tests.h>
#include <common/tests/scfd_vector_operations_nmfd_interface_tests.h>
#include <common/tests/vector_operations_template_tests.h>

namespace
{

struct scfd_vector_access
{
    template <class VecOps>
    void write(
        VecOps &vec_ops, typename VecOps::vector_type &dst, const std::vector<typename VecOps::scalar_type> &src
    ) const
    {
        vec_ops.set( src.data(), dst, src.size() );
    }

    template <class VecOps>
    std::vector<typename VecOps::scalar_type>
    read( VecOps &vec_ops, const typename VecOps::vector_type &src, std::size_t n ) const
    {
        std::vector<typename VecOps::scalar_type> host( n );
        vec_ops.get( src, host.data(), n );
        return host;
    }
};

template <class Backend, class T>
void run_scfd_type(
    const std::string &label, const std::vector<std::size_t> &sizes, vector_operations_tests::test_report &report
)
{
    static_assert(
        std::is_same<typename scfd_vector_operations<Backend, T>::ordinal_type, typename Backend::ordinal_type>::value
    );
    for ( const auto n : sizes )
    {
        scfd_vector_operations<Backend, T> vec_ops( n );
        vector_operations_tests::run_vector_operations_template_tests(
            vec_ops, scfd_vector_access{}, n, label + " n=" + std::to_string( n ), report
        );
        vector_operations_tests::run_nmfd_vector_space_interface_tests(
            vec_ops, scfd_vector_access{}, n, label + " n=" + std::to_string( n ), report
        );
        vector_operations_tests::run_scfd_vector_operations_high_precision_tests(
            vec_ops, scfd_vector_access{}, n, label + " n=" + std::to_string( n ), report
        );
    }
}

} // namespace

int main( int argc, char **argv )
{
    vector_operations_tests::test_report report;
    const std::string                    device_selector = ( argc > 1 ) ? argv[1] : "auto";

    using real         = SCALAR_TYPE;
    using cuda_complex = common::scfd_backend_ext::complex_t<scfd::backend::cuda<std::ptrdiff_t>, real>;

    const std::vector<std::size_t> cpu_sizes = { 1, 7, 64 };
    run_scfd_type<scfd::backend::serial_cpu<std::ptrdiff_t>, real>( "SCFD serial real", cpu_sizes, report );

    if ( device_selector != "cpu" )
    {
        const int device = common::init_cuda_from_scfd_selector( device_selector );
        std::cout << "CUDA device: " << device << std::endl;

        const std::vector<std::size_t> cuda_sizes = { 1, 2, 7, 31, 32, 33, 64, 1025, 4097 };
        run_scfd_type<scfd::backend::cuda<std::ptrdiff_t>, real>( "SCFD CUDA real", cuda_sizes, report );
        run_scfd_type<scfd::backend::cuda<std::ptrdiff_t>, cuda_complex>( "SCFD CUDA complex", cuda_sizes, report );
        run_scfd_type<scfd::backend::cuda<int>, real>( "SCFD CUDA int real", cuda_sizes, report );
        run_scfd_type<scfd::backend::cuda<int>, cuda_complex>( "SCFD CUDA int complex", cuda_sizes, report );
    }

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if ( report.failures == 0 )
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
