#ifndef __SCFD_VECTOR_OPERATIONS_NMFD_INTERFACE_TESTS_H__
#define __SCFD_VECTOR_OPERATIONS_NMFD_INTERFACE_TESTS_H__

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <common/tests/vector_operations_template_tests.h>

namespace vector_operations_tests
{

template <class VecOps>
struct magnitude_transform
{
    using scalar_type                   = typename VecOps::scalar_type;
    using norm_type                     = typename VecOps::norm_type;
    using traits                        = typename VecOps::scalar_traits;
    norm_type                multiplier = 1, offset = 0;
    __DEVICE_TAG__ norm_type operator()( const scalar_type &x ) const
    {
        return multiplier * traits::asum_term( x ) + offset;
    }
};

template <class VecOps>
struct distance_transform
{
    using scalar_type = typename VecOps::scalar_type;
    using norm_type   = typename VecOps::norm_type;
    using traits      = typename VecOps::scalar_traits;
    __DEVICE_TAG__ norm_type operator()( const scalar_type &x, const scalar_type &y ) const
    {
        return traits::asum_term( x - scalar_type( 2 ) * y );
    }
    __DEVICE_TAG__ norm_type operator()( const scalar_type &x, const scalar_type &y, const scalar_type &z ) const
    {
        return traits::norm_sq_term( x + scalar_type( 2 ) * y - z );
    }
};

template <class VecOps, class Access>
void run_nmfd_vector_space_interface_tests(
    VecOps &vec_ops, Access access, std::size_t n, const std::string &label, test_report &report
)
{
    using base_type        = typename VecOps::nmfd_parent_type;
    using scalar_type      = typename VecOps::scalar_type;
    using vector_type      = typename VecOps::vector_type;
    using multivector_type = typename VecOps::multivector_type;

    const base_type &base = vec_ops;

    vector_type x;
    vector_type y;
    vector_type z;
    base.init_vector( x );
    base.init_vector( y );
    base.init_vector( z );
    base.start_use_vector( x );
    base.start_use_vector( y );
    base.start_use_vector( z );

    const auto two       = make_scalar<scalar_type>( 2.0L, 0.0L );
    const auto three     = make_scalar<scalar_type>( 3.0L, 0.0L );
    const auto minus_one = make_scalar<scalar_type>( -1.0L, 0.0L );
    const auto half      = make_scalar<scalar_type>( 0.5L, 0.0L );

    base.assign_scalar( two, x );
    base.assign_lin_comb( three, x, y );
    check_vector_close(
        report, label + " NMFD assign_lin_comb one-vector", access.read( vec_ops, y, n ),
        std::vector<scalar_type>( n, make_scalar<scalar_type>( 6.0L, 0.0L ) )
    );

    base.assign_lin_comb( two, x, minus_one, y, z );
    check_vector_close(
        report, label + " NMFD assign_lin_comb two-vector", access.read( vec_ops, z, n ),
        std::vector<scalar_type>( n, make_scalar<scalar_type>( -2.0L, 0.0L ) )
    );

    base.add_lin_comb( half, x, half, y );
    check_vector_close(
        report, label + " NMFD add_lin_comb", access.read( vec_ops, y, n ),
        std::vector<scalar_type>( n, make_scalar<scalar_type>( 4.0L, 0.0L ) )
    );

    const auto expected_dot = make_scalar<scalar_type>( 8.0L * static_cast<long double>( n ), 0.0L );
    check_close( report, label + " NMFD scalar_prod_l2", base.scalar_prod_l2( x, y ), expected_dot, 1024.0L );

    multivector_type mv;
    base.init_multivector( mv, 2 );
    base.start_use_multivector( mv, 2 );
    base.assign( x, mv, 2, 0 );
    base.assign( y, mv, 2, 1 );
    base.assign( mv, 2, 0, z );
    check_vector_close(
        report, label + " NMFD multivector assign out", access.read( vec_ops, z, n ), std::vector<scalar_type>( n, two )
    );
    check_close(
        report, label + " NMFD multivector scalar_prod", base.scalar_prod( mv, 2, 1, x ), expected_dot, 1024.0L
    );

    base.add_lin_comb( three, mv, 2, 0, minus_one, z );
    check_vector_close(
        report, label + " NMFD multivector add_lin_comb", access.read( vec_ops, z, n ),
        std::vector<scalar_type>( n, make_scalar<scalar_type>( 4.0L, 0.0L ) )
    );

    base.stop_use_multivector( mv, 2 );
    base.free_multivector( mv, 2 );

    // Exercise arbitrary mappings, including signed results and several inputs.
    using norm_type = typename VecOps::norm_type;
    std::vector<scalar_type> hx( n ), hy( n ), hz( n );
    long double              unary = -std::numeric_limits<long double>::infinity(), binary = 0, ternary = 0;
    for ( std::size_t i = 0; i < n; ++i )
    {
        const auto k = static_cast<long double>( i % 7 + 1 );
        hx[i]        = make_scalar<scalar_type>( -k, .25L * k );
        hy[i]        = make_scalar<scalar_type>( .5L * k, -.125L * k );
        hz[i]        = make_scalar<scalar_type>( k + 1, -k );
        unary        = std::max( unary, -100 + scalar_traits<scalar_type>::asum_term( hx[i] ) );
        binary       = std::max( binary, scalar_traits<scalar_type>::asum_term( hx[i] - two * hy[i] ) );
        ternary      = std::max( ternary, scalar_traits<scalar_type>::norm_sq_term( hx[i] + two * hy[i] - hz[i] ) );
    }
    access.write( vec_ops, x, hx );
    access.write( vec_ops, y, hy );
    access.write( vec_ops, z, hz );
    check_close_real(
        report, label + " transform max negative",
        vec_ops.transform_reduce_max( magnitude_transform<VecOps>{ 1, -100 }, x ), unary
    );
    check_close_real(
        report, label + " transform max binary", vec_ops.transform_reduce_max( distance_transform<VecOps>{}, x, y ),
        binary
    );
    check_close_real(
        report, label + " transform max ternary", vec_ops.transform_reduce_max( distance_transform<VecOps>{}, x, y, z ),
        ternary
    );
    check_vector_close( report, label + " transform preserves x", access.read( vec_ops, x, n ), hx );
    check_vector_close( report, label + " transform preserves y", access.read( vec_ops, y, n ), hy );
    check_vector_close( report, label + " transform preserves z", access.read( vec_ops, z, n ), hz );
    for ( const auto position : { std::size_t( 0 ), n - 1 } )
    {
        auto special      = hx;
        special[position] = make_scalar<scalar_type>( std::numeric_limits<norm_type>::quiet_NaN() );
        access.write( vec_ops, x, special );
        report.require(
            std::isnan( vec_ops.transform_reduce_max( magnitude_transform<VecOps>{}, x ) ),
            label + " transform max propagates NaN"
        );
        special[position] = make_scalar<scalar_type>( std::numeric_limits<norm_type>::infinity() );
        access.write( vec_ops, x, special );
        report.require(
            vec_ops.transform_reduce_max( magnitude_transform<VecOps>{}, x ) ==
                std::numeric_limits<norm_type>::infinity(),
            label + " transform max positive infinity"
        );
    }
    vec_ops.assign_scalar( make_scalar<scalar_type>( std::numeric_limits<norm_type>::infinity() ), x );
    report.require(
        vec_ops.transform_reduce_max( magnitude_transform<VecOps>{ -1, 0 }, x ) ==
            -std::numeric_limits<norm_type>::infinity(),
        label + " transform max negative infinity"
    );
    access.write( vec_ops, x, hx );
    check_close_real(
        report, label + " transform max reuses scratch",
        vec_ops.transform_reduce_max( magnitude_transform<VecOps>{ 1, -100 }, x ), unary
    );
    vector_type wrong_size;
    vec_ops.init_vector( wrong_size );
    vec_ops.start_use_vector( wrong_size, n + 1 );
    bool refused = false;
    try
    {
        vec_ops.transform_reduce_max( distance_transform<VecOps>{}, x, wrong_size, z );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    report.require( refused, label + " transform max rejects size mismatch" );
    vec_ops.stop_use_vector( wrong_size );
    vec_ops.free_vector( wrong_size );

    base.stop_use_vector( x );
    base.stop_use_vector( y );
    base.stop_use_vector( z );
    base.free_vector( x );
    base.free_vector( y );
    base.free_vector( z );
}

} // namespace vector_operations_tests

#endif
