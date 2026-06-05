#ifndef __NMFD_DENSE_OPERATIONS_CUDA_HIP_H__
#define __NMFD_DENSE_OPERATIONS_CUDA_HIP_H__

#include <scfd/utils/cuda_safe_call.h>

#include <nmfd/operations/dense_operations_base.h>

namespace nmfd
{
namespace operations
{

template <class Type, class Backend, class Ordinal = std::ptrdiff_t>
class dense_operations_cuda_hip : public dense_operations<Type, Backend, Ordinal>
{
public:
    using parent_t = dense_operations<Type, Backend, Ordinal>;

    using matrix_type      = typename parent_t::matrix_type;
    using vector_type      = typename parent_t::vector_type;
    using scalar_type      = typename parent_t::scalar_type;
    using blas_wrap_type   = typename Backend::blas_wrap_type;
    using solver_wrap_type = typename Backend::solver_wrap_type;

public:
    dense_operations_cuda_hip() = default;

    template <typename... Args>
    dense_operations_cuda_hip( Args &&...args ) : parent_t( std::forward<Args>( args )... )
    {
    }

    void add_matrix_vector_prod(
        const scalar_type alpha, const matrix_type &mat, const vector_type &x, const scalar_type beta, vector_type &y
    ) const
    {
        auto      &device_blas = blas_wrap_type::inst();
        const auto sz          = mat.size_nd();
        const auto rows        = sz[0];
        const auto cols        = sz[1];
        device_blas.gemv(
            'N', rows, mat.raw_ptr(), cols, rows, alpha, parent_t::vt_.get_raw_ptr( x ), beta,
            parent_t::vt_.get_raw_ptr( y )
        );
    }

    [[nodiscard]] std::shared_ptr<matrix_type>
    matrix_matrix_prod( const matrix_type &mat_a, const matrix_type &mat_b ) const
    {
        auto &device_blas = blas_wrap_type::inst();

        const auto sz_a = mat_a.size_nd();
        const auto sz_b = mat_b.size_nd();

        const auto rows_a        = sz_a[0];
        const auto cols_a_rows_b = sz_a[1];
        const auto cols_b        = sz_b[1];

        auto result = std::make_shared<matrix_type>();
        result->init( rows_a, cols_b );
        parent_t::assign_zero_matrix( *result );

        device_blas.gemm(
            'N', 'N', rows_a, cols_b, cols_a_rows_b, scalar_type{ 1 }, mat_a.raw_ptr(), rows_a, mat_b.raw_ptr(),
            cols_a_rows_b, scalar_type{ 0 }, result->raw_ptr(), rows_a
        );

        return result;
    }

    void solve( const matrix_type &mat, const vector_type &b, vector_type &x ) const
    {
        auto &cusolver = solver_wrap_type::inst();

        const auto n = mat.size_nd()[0];
        parent_t::verify_max_loc_size( n * n + n );

        scalar_type *const A_buf = parent_t::vt_.get_raw_ptr( parent_t::helper_ );
        scalar_type *const tau   = A_buf + n * n;

        CUDA_SAFE_CALL( cudaMemcpy( A_buf, mat.raw_ptr(), sizeof( scalar_type ) * n * n, cudaMemcpyDeviceToDevice ) );
        CUDA_SAFE_CALL( cudaMemcpy(
            parent_t::vt_.get_raw_ptr( x ), parent_t::vt_.get_raw_ptr( b ), sizeof( scalar_type ) * n,
            cudaMemcpyDeviceToDevice
        ) );

        cusolver.geqrf( n, n, A_buf, tau );
        cusolver.gesv_apply_qr( n, A_buf, tau, parent_t::vt_.get_raw_ptr( x ) );
    }
};

}

}

#endif
