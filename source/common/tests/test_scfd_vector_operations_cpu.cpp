#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

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

int main()
{
    vector_operations_tests::test_report report;

    using real           = SCALAR_TYPE;
    using serial_complex = common::scfd_backend_ext::complex_t<scfd::backend::serial_cpu<std::ptrdiff_t>, real>;
    using omp_complex    = common::scfd_backend_ext::complex_t<scfd::backend::omp<std::ptrdiff_t>, real>;

    const std::vector<std::size_t> sizes = { 1, 2, 7, 31, 32, 33, 64, 1025, 4097 };
    run_scfd_type<scfd::backend::serial_cpu<std::ptrdiff_t>, real>( "SCFD serial real", sizes, report );
    run_scfd_type<scfd::backend::serial_cpu<std::ptrdiff_t>, serial_complex>( "SCFD serial complex", sizes, report );
    run_scfd_type<scfd::backend::omp<std::ptrdiff_t>, real>( "SCFD OMP real", sizes, report );
    run_scfd_type<scfd::backend::omp<std::ptrdiff_t>, omp_complex>( "SCFD OMP complex", sizes, report );
    run_scfd_type<scfd::backend::serial_cpu<int>, real>( "SCFD serial int real", sizes, report );
    run_scfd_type<scfd::backend::serial_cpu<int>, serial_complex>( "SCFD serial int complex", sizes, report );
    run_scfd_type<scfd::backend::omp<int>, real>( "SCFD OMP int real", sizes, report );
    run_scfd_type<scfd::backend::omp<int>, omp_complex>( "SCFD OMP int complex", sizes, report );

    for ( bool narrow_backend : { false, true } )
    {
        bool rejected = false;
        try
        {
            if ( narrow_backend )
            {
                scfd_vector_operations<scfd::backend::serial_cpu<signed char>, real> invalid( 128 );
            }
            else
            {
                scfd_vector_operations<scfd::backend::serial_cpu<std::ptrdiff_t>, real, signed char> invalid( 128 );
            }
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.require( rejected, "oversized vector is rejected before narrowing or allocation" );
    }
    {
        scfd_vector_operations<scfd::backend::serial_cpu<std::ptrdiff_t>, real, signed char> operations( 1 );
        decltype( operations )::vector_type                                                  vector;
        bool                                                                                 rejected = false;
        try
        {
            operations.start_use_vector( vector, 128 );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.require( rejected && vector.is_free(), "oversized vector activation leaves storage unallocated" );
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
