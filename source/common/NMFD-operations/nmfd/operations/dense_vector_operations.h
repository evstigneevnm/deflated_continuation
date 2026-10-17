#ifndef __NMFD_DENSE_VECTOR_OPERATIONS_H__
#define __NMFD_DENSE_VECTOR_OPERATIONS_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#include <scfd/arrays/array.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

#include <common/scfd_backend_ext/arg_reduce.h>
#include <common/scfd_backend_ext/host_view.h>
#include <common/scfd_backend_ext/math.h>
#include <common/scfd_backend_ext/random.h>

#include <nmfd/operations/blas1/high_precision/compensated_reduction.h>
#include <nmfd/operations/default_multivector_operations_base.h>

namespace nmfd
{
namespace operations
{
namespace detail
{

// Namespace-level types can be passed to CUDA kernels.
template <class VectorOperations, class Transform, std::size_t N>
struct scfd_transform_kernel
{
    using scalar_type  = typename VectorOperations::scalar_type;
    using norm_type    = typename VectorOperations::norm_type;
    using ordinal_type = typename VectorOperations::ordinal_type;
    Transform                                     transform;
    scfd::static_vec::vec<const scalar_type *, N> inputs;
    norm_type                                    *output;

    template <std::size_t... I>
    __DEVICE_TAG__ norm_type evaluate( ordinal_type i, std::index_sequence<I...> ) const
    {
        return transform( inputs[I][i]... );
    }
    __DEVICE_TAG__ void operator()( ordinal_type i ) const
    {
        output[i] = evaluate( i, std::make_index_sequence<N>{} );
    }
};

template <class T>
struct scfd_transform_max_op
{
    __DEVICE_TAG__ T operator()( T a, T b ) const
    {
        if ( a != a )
            return a;
        if ( b != b )
            return b;
        return a > b ? a : b;
    }
};

}

template <class VectorTraits, class Backend, class Ordinal = typename Backend::ordinal_type>
class dense_vector_operations
    : public default_multivector_operations_base<
          dense_vector_operations<VectorTraits, Backend, Ordinal>, typename VectorTraits::scalar_type,
          typename VectorTraits::vector_type, Ordinal, std::vector<typename VectorTraits::vector_type>,
          vector_operations_base<
              typename VectorTraits::scalar_type, typename VectorTraits::vector_type,
              std::vector<typename VectorTraits::vector_type>, Ordinal,
              typename common::scfd_backend_ext::scalar_traits<Backend, typename VectorTraits::scalar_type>::real_type>>
{
public:
    using backend_type          = Backend;
    using scalar_type           = typename VectorTraits::scalar_type;
    using ordinal_type          = Ordinal;
    using big_ordinal_type      = Ordinal;
    using memory_type           = typename backend_type::memory_type;
    using vector_type           = typename VectorTraits::vector_type;
    using multivector_type      = std::vector<vector_type>;
    using scalar_traits         = common::scfd_backend_ext::scalar_traits<backend_type, scalar_type>;
    using norm_type             = typename scalar_traits::real_type;
    using Tsc                   = norm_type;
    using real_traits_type      = typename VectorTraits::template rebind<norm_type>;
    using vector_type_real      = typename real_traits_type::vector_type;
    using host_view_traits      = common::scfd_backend_ext::host_view_traits<vector_type>;
    using host_view_traits_real = common::scfd_backend_ext::host_view_traits<vector_type_real>;
    using for_each_type         = typename backend_type::for_each_type;
    using reduce_type           = typename backend_type::reduce_type;
    using copy_type             = typename backend_type::copy_type;
    using high_precision_reduction_type =
        nmfd::operations::blas1::high_precision::compensated_reduction<backend_type, scalar_type, ordinal_type>;
    using operations_base_type =
        vector_operations_base<scalar_type, vector_type, multivector_type, ordinal_type, norm_type>;

    using multivector_operations_base = default_multivector_operations_base<
        dense_vector_operations, scalar_type, vector_type, ordinal_type, multivector_type, operations_base_type>;
    using multivector_operations_base::add_lin_comb;
    using multivector_operations_base::assign;
    using multivector_operations_base::scalar_prod;
    using multivector_operations_base::scalar_prod_l2;

    dense_vector_operations() : dense_vector_operations( VectorTraits{} )
    {
    }
    template <class... Args>
    explicit dense_vector_operations( Args &&...args )
        : vt_( std::forward<Args>( args )... ), high_precision_reduction_( 0 )
    {
        verify_max_loc_size( checked_size( vt_.loc_size() ) );
        ensure_real_capacity( checked_size( vt_.loc_size() ) );
    }
    dense_vector_operations( const dense_vector_operations & )            = delete;
    dense_vector_operations &operator=( const dense_vector_operations & ) = delete;
    ~dense_vector_operations()
    {
        free_owned( helper_ );
        free_owned( helper_real_ );
    }
    std::size_t get_size( const vector_type &x ) const
    {
        return checked_size( vt_.get_loc_size( x ) );
    }
    ordinal_type get_loc_size( const vector_type &x ) const
    {
        return static_cast<ordinal_type>( get_size( x ) );
    }

    bool device_location() const
    {
        return !memory_type::is_host_visible;
    }

    unsigned int get_fp_prec() const
    {
        return std::numeric_limits<norm_type>::digits10 + 1;
    }

    void use_high_precision()
    {
        this->set_high_precision();
    }

    bool high_precision_requested() const
    {
        return this->use_high_precision_;
    }

    bool last_reduction_used_high_precision() const
    {
        return last_reduction_high_precision_;
    }

    vector_type &at( multivector_type &x, std::size_t m, std::size_t k ) const
    {
        if ( k >= m )
        {
            throw std::out_of_range( "scfd_vector_operations::at" );
        }
        return x[k];
    }

    auto view( vector_type &x, bool sync_from_array = true, bool sync_to_array_on_destroy = false ) const
    {
        return typename host_view_traits::scoped_view_type( x, sync_from_array, sync_to_array_on_destroy );
    }
    auto view( const vector_type &x, bool sync_from_array = true ) const
    {
        return typename host_view_traits::scoped_view_type( x, sync_from_array, false );
    }
    void set_value_at_point( scalar_type value, std::size_t index, vector_type &x ) const
    {
        if ( index >= get_size( x ) )
            throw std::out_of_range( "dense_vector_operations::set_value_at_point" );
        memory_type::copy_from_host( sizeof( scalar_type ), &value, vt_.get_raw_ptr( x ) + index );
    }
    scalar_type get_value_at_point( std::size_t index, const vector_type &x ) const
    {
        if ( index >= get_size( x ) )
            throw std::out_of_range( "dense_vector_operations::get_value_at_point" );
        scalar_type value;
        memory_type::copy_to_host( sizeof( scalar_type ), vt_.get_raw_ptr( x ) + index, &value );
        return value;
    }

    void set( const scalar_type *host, vector_type &x ) const
    {
        copy_type()( get_loc_size( x ), host, vt_.get_raw_ptr( x ) );
    }

    void set( const scalar_type *host, vector_type &x, std::size_t n ) const
    {
        copy_type()( static_cast<ordinal_type>( n ), host, vt_.get_raw_ptr( x ) );
    }

    void get( const vector_type &x, scalar_type *host ) const
    {
        copy_type()( get_loc_size( x ), vt_.get_raw_ptr( x ), host );
    }

    void get( const vector_type &x, scalar_type *host, std::size_t n ) const
    {
        copy_type()( static_cast<ordinal_type>( n ), vt_.get_raw_ptr( x ), host );
    }

    bool check_is_valid_number( const vector_type &x ) const
    {
        if ( get_size( x ) == 0 )
            return true;
        ensure_real_capacity( get_size( x ) );
        const auto xp = vt_.get_raw_ptr( x );
        auto       hp = helper_real_.raw_ptr();
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                hp[i] = scalar_traits::is_finite( xp[i] ) ? norm_type( 0 ) : norm_type( 1 );
            },
            get_loc_size( x )
        );
        for_each_inst_.wait();
        return reduce_inst_( get_loc_size( x ), hp, norm_type( 0 ) ) == norm_type( 0 );
    }

    bool is_valid_number( const vector_type &x ) const
    {
        return check_is_valid_number( x );
    }

    scalar_type scalar_prod( const vector_type &x, const vector_type &y ) const
    {
        if ( get_size( x ) != get_size( y ) )
            throw std::invalid_argument( "dense_vector_operations::scalar_prod: vector size mismatch" );
        if ( get_size( x ) == 0 )
            return scalar_traits::zero();
        verify_max_loc_size( get_size( x ) );
        if ( this->use_high_precision_ )
        {
            prepare_high_precision( get_size( x ) );
            last_reduction_high_precision_ = true;
            return high_precision_reduction_.dot( get_loc_size( x ), vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) );
        }
        last_reduction_high_precision_ = false;
        const auto xp                  = vt_.get_raw_ptr( x );
        const auto yp                  = vt_.get_raw_ptr( y );
        auto       hp                  = helper_.raw_ptr();
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { hp[i] = scalar_traits::conj_mul( xp[i], yp[i] ); }, get_loc_size( x )
        );
        for_each_inst_.wait();
        return reduce_inst_( get_loc_size( x ), hp, scalar_traits::zero() );
    }

    scalar_type scalar_prod_l2( const vector_type &x, const vector_type &y ) const override
    {
        return scalar_prod( x, y );
    }

    scalar_type sum( const vector_type &x ) const
    {
        if ( get_size( x ) == 0 )
            return scalar_traits::zero();
        if ( this->use_high_precision_ )
        {
            prepare_high_precision( get_size( x ) );
            last_reduction_high_precision_ = true;
            return high_precision_reduction_.sum( get_loc_size( x ), vt_.get_raw_ptr( x ) );
        }
        last_reduction_high_precision_ = false;
        return reduce_inst_( get_loc_size( x ), vt_.get_raw_ptr( x ), scalar_traits::zero() );
    }

    norm_type asum( const vector_type &x ) const
    {
        if ( get_size( x ) == 0 )
            return norm_type( 0 );
        ensure_real_capacity( get_size( x ) );
        if ( this->use_high_precision_ )
        {
            prepare_high_precision( get_size( x ) );
            last_reduction_high_precision_ = true;
            return high_precision_reduction_.asum( get_loc_size( x ), vt_.get_raw_ptr( x ) );
        }
        last_reduction_high_precision_ = false;
        const auto xp                  = vt_.get_raw_ptr( x );
        auto       hp                  = helper_real_.raw_ptr();
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { hp[i] = scalar_traits::asum_term( xp[i] ); }, get_loc_size( x )
        );
        for_each_inst_.wait();
        return reduce_inst_( get_loc_size( x ), hp, norm_type( 0 ) );
    }

    norm_type norm( const vector_type &x ) const
    {
        return common::scfd_backend_ext::math<backend_type, norm_type>::sqrt( norm_sq( x ) );
    }

    norm_type norm_sq( const vector_type &x ) const
    {
        if ( get_size( x ) == 0 )
            return norm_type( 0 );
        ensure_real_capacity( get_size( x ) );
        if ( this->use_high_precision_ )
        {
            prepare_high_precision( get_size( x ) );
            last_reduction_high_precision_ = true;
            return high_precision_reduction_.norm_sq( get_loc_size( x ), vt_.get_raw_ptr( x ) );
        }
        last_reduction_high_precision_ = false;
        const auto xp                  = vt_.get_raw_ptr( x );
        auto       hp                  = helper_real_.raw_ptr();
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { hp[i] = scalar_traits::norm_sq_term( xp[i] ); }, get_loc_size( x )
        );
        for_each_inst_.wait();
        return reduce_inst_( get_loc_size( x ), hp, norm_type( 0 ) );
    }

    norm_type norm_l2( const vector_type &x ) const
    {
        return norm( x );
    }

    norm_type norm2( const vector_type &x ) const
    {
        return norm_l2( x );
    }

    norm_type norm2_sq( const vector_type &x ) const
    {
        return norm_sq( x );
    }

    norm_type norm_l2_sq( const vector_type &x ) const override
    {
        return norm2_sq( x );
    }

    norm_type norm1( const vector_type &x ) const override
    {
        return asum( x );
    }

    norm_type norm_l1( const vector_type &x ) const override
    {
        return norm1( x );
    }

    norm_type norm_rank1( const vector_type &x, const scalar_type val_x ) const
    {
        return common::scfd_backend_ext::math<backend_type, norm_type>::sqrt(
            norm_sq( x ) + scalar_traits::norm_sq_term( val_x )
        );
    }

    norm_type norm_rank1_l2( const vector_type &x, const scalar_type val_x ) const
    {
        return norm_rank1( x, val_x );
    }

    norm_type norm_inf( const vector_type &x ) const
    {
        if ( get_size( x ) == 0 )
            return norm_type( 0 );
        ensure_real_capacity( get_size( x ) );
        const auto xp = vt_.get_raw_ptr( x );
        auto       hp = helper_real_.raw_ptr();
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { hp[i] = scalar_traits::asum_term( xp[i] ); }, get_loc_size( x )
        );
        for_each_inst_.wait();
        return reduce_inst_( get_loc_size( x ), hp, norm_type( 0 ), common::scfd_backend_ext::max_op<norm_type>() );
    }

    norm_type norm_l_inf( const vector_type &x ) const override
    {
        return norm_inf( x );
    }

    // Elementwise scalar mapping followed by maximum; empty -> -inf, NaNs propagate.
    // The mapping must be const-callable on the backend and return a real scalar.
    template <class Transform, class... Vectors>
    norm_type transform_reduce_max( const Transform &transform, const Vectors &...vectors ) const
    {
        static_assert( sizeof...( Vectors ) > 0, "Transform reduction requires at least one vector" );
        static_assert( ( std::is_same_v<Vectors, vector_type> && ... ), "Transform reduction vector type mismatch" );
        static_assert(
            std::is_convertible_v<decltype( transform( ( *vt_.get_raw_ptr( vectors ) )... ) ), norm_type>,
            "Transform reduction requires a real scalar result"
        );
        const std::size_t size = get_size( std::get<0>( std::tie( vectors... ) ) );
        if ( ( ( get_size( vectors ) != size ) || ... ) )
            throw std::invalid_argument( "Transform reduction: vector size mismatch" );
        const auto identity = -std::numeric_limits<norm_type>::infinity();
        if ( size == 0 )
            return identity;
        ensure_real_capacity( size );
        using kernel_type =
            nmfd::operations::detail::scfd_transform_kernel<dense_vector_operations, Transform, sizeof...( Vectors )>;
        kernel_type kernel{ transform, {}, helper_real_.raw_ptr() };
        std::size_t index = 0;
        ( ( kernel.inputs[index++] = vt_.get_raw_ptr( vectors ) ), ... );
        for_each_inst_( kernel, static_cast<ordinal_type>( size ) );
        for_each_inst_.wait();
        return reduce_inst_(
            static_cast<ordinal_type>( size ), helper_real_.raw_ptr(), identity,
            nmfd::operations::detail::scfd_transform_max_op<norm_type>{}
        );
    }

    // Elementwise real-valued mapping followed by sum; empty -> zero, NaNs propagate.
    template <class Transform, class... Vectors>
    norm_type transform_reduce_sum( const Transform &transform, const Vectors &...vectors ) const
    {
        static_assert( sizeof...( Vectors ) > 0, "Transform reduction requires at least one vector" );
        static_assert( ( std::is_same_v<Vectors, vector_type> && ... ), "Transform reduction vector type mismatch" );
        static_assert(
            std::is_convertible_v<decltype( transform( ( *vt_.get_raw_ptr( vectors ) )... ) ), norm_type>,
            "Transform reduction requires a real scalar result"
        );
        const std::size_t size = get_size( std::get<0>( std::tie( vectors... ) ) );
        if ( ( ( get_size( vectors ) != size ) || ... ) )
        {
            throw std::invalid_argument( "Transform reduction: vector size mismatch" );
        }
        last_reduction_high_precision_ = false;
        if ( size == 0 )
        {
            return norm_type( 0 );
        }
        ensure_real_capacity( size );
        using kernel_type =
            nmfd::operations::detail::scfd_transform_kernel<dense_vector_operations, Transform, sizeof...( Vectors )>;
        kernel_type kernel{ transform, {}, helper_real_.raw_ptr() };
        std::size_t index = 0;
        ( ( kernel.inputs[index++] = vt_.get_raw_ptr( vectors ) ), ... );
        for_each_inst_( kernel, static_cast<ordinal_type>( size ) );
        for_each_inst_.wait();
        if constexpr ( std::is_same_v<scalar_type, norm_type> )
        {
            if ( this->use_high_precision_ )
            {
                prepare_high_precision( size );
                last_reduction_high_precision_ = true;
                return high_precision_reduction_.sum( static_cast<ordinal_type>( size ), helper_real_.raw_ptr() );
            }
        }
        return reduce_inst_( static_cast<ordinal_type>( size ), helper_real_.raw_ptr(), norm_type( 0 ) );
    }

    norm_type normalize( vector_type &x ) const
    {
        const norm_type norm_x = norm( x );
        if ( norm_x > norm_type( 0 ) )
        {
            scale( scalar_type( norm_type( 1 ) / norm_x ), x );
        }
        return norm_x;
    }

    void assign_scalar( const scalar_type scalar, vector_type &x ) const
    {
        auto xp = vt_.get_raw_ptr( x );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] = scalar; }, static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void add_mul_scalar( const scalar_type scalar, const scalar_type mul_x, vector_type &x ) const
    {
        auto xp = vt_.get_raw_ptr( x );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] = mul_x * xp[i] + scalar; },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void scale( const scalar_type alpha, vector_type &x ) const
    {
        auto xp = vt_.get_raw_ptr( x );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] *= alpha; }, static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void assign( const vector_type &x, vector_type &y ) const
    {
        copy_type()( static_cast<ordinal_type>( get_size( x ) ), vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) );
    }

    void swap( vector_type &x, vector_type &y ) const
    {
        auto xp = vt_.get_raw_ptr( x );
        auto yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                const scalar_type tmp = xp[i];
                xp[i]                 = yp[i];
                yp[i]                 = tmp;
            },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void assign_mul( const scalar_type mul_x, const vector_type &x, vector_type &y ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        auto       yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] = mul_x * xp[i]; }, static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void assign_lin_comb( const scalar_type mul_x, const vector_type &x, vector_type &y ) const override
    {
        assign_mul( mul_x, x, y );
    }

    void assign_mul(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        auto       zp = vt_.get_raw_ptr( z );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { zp[i] = mul_x * xp[i] + mul_y * yp[i]; },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void assign_lin_comb(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const override
    {
        assign_mul( mul_x, x, mul_y, y, z );
    }

    void add_mul( const scalar_type mul_x, const vector_type &x, vector_type &y ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        auto       yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] += mul_x * xp[i]; }, static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void add_mul( const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, vector_type &y ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        auto       yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] = mul_x * xp[i] + mul_y * yp[i]; },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void add_lin_comb( const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, vector_type &y ) const
    {
        add_mul( mul_x, x, mul_y, y );
    }

    void add_lin_comb( const scalar_type mul_x, const vector_type &x, vector_type &y ) const
    {
        add_mul( mul_x, x, y );
    }

    void add_mul(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y,
        const scalar_type mul_z, vector_type &z
    ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        auto       zp = vt_.get_raw_ptr( z );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { zp[i] = mul_x * xp[i] + mul_y * yp[i] + mul_z * zp[i]; },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void add_lin_comb(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y,
        const scalar_type mul_z, vector_type &z
    ) const
    {
        add_mul( mul_x, x, mul_y, y, mul_z, z );
    }

    void mul_pointwise(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        auto       zp = vt_.get_raw_ptr( z );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { zp[i] = ( mul_x * xp[i] ) * ( mul_y * yp[i] ); },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void mul_pointwise( vector_type &x, const scalar_type mul_y, const vector_type &y ) const
    {
        auto       xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] *= mul_y * yp[i]; }, static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void div_pointwise(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        auto       zp = vt_.get_raw_ptr( z );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { zp[i] = ( mul_x * xp[i] ) / ( mul_y * yp[i] ); },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void div_pointwise( vector_type &x, const scalar_type mul_y, const vector_type &y ) const
    {
        auto       xp = vt_.get_raw_ptr( x );
        const auto yp = vt_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] /= ( mul_y * yp[i] ); },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void make_abs_copy( const vector_type &x, vector_type_real &y ) const
    {
        const auto xp = vt_.get_raw_ptr( x );
        auto       yp = real_traits_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] = scalar_traits::asum_term( xp[i] ); },
            static_cast<ordinal_type>( get_size( x ) )
        );
        for_each_inst_.wait();
    }

    void make_abs( vector_type_real &x ) const
    {
        auto xp = real_traits_.get_raw_ptr( x );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { xp[i] = xp[i] < norm_type( 0 ) ? -xp[i] : xp[i]; },
            static_cast<ordinal_type>( real_traits_.get_loc_size( x ) )
        );
        for_each_inst_.wait();
    }

    void max_pointwise( const norm_type sc, const vector_type_real &x, vector_type_real &y ) const
    {
        const auto xp = real_traits_.get_raw_ptr( x );
        auto       yp = real_traits_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                const norm_type xy = xp[i] > yp[i] ? xp[i] : yp[i];
                yp[i]              = xy > sc ? xy : sc;
            },
            static_cast<ordinal_type>( real_traits_.get_loc_size( x ) )
        );
        for_each_inst_.wait();
    }

    void max_pointwise( const norm_type sc, vector_type_real &y ) const
    {
        auto yp = real_traits_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] = yp[i] > sc ? yp[i] : sc; },
            static_cast<ordinal_type>( real_traits_.get_loc_size( y ) )
        );
        for_each_inst_.wait();
    }

    void min_pointwise( const norm_type sc, const vector_type_real &x, vector_type_real &y ) const
    {
        const auto xp = real_traits_.get_raw_ptr( x );
        auto       yp = real_traits_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) {
                const norm_type xy = xp[i] < yp[i] ? xp[i] : yp[i];
                yp[i]              = xy < sc ? xy : sc;
            },
            static_cast<ordinal_type>( real_traits_.get_loc_size( x ) )
        );
        for_each_inst_.wait();
    }

    void min_pointwise( const norm_type sc, vector_type_real &y ) const
    {
        auto yp = real_traits_.get_raw_ptr( y );
        for_each_inst_(
            [=] __DEVICE_TAG__( ordinal_type i ) { yp[i] = yp[i] < sc ? yp[i] : sc; },
            static_cast<ordinal_type>( real_traits_.get_loc_size( y ) )
        );
        for_each_inst_.wait();
    }

    void set_value_at_point( scalar_type value, std::size_t index, vector_type &x, std::size_t size ) const
    {
        if ( index >= size || size > get_size( x ) )
            throw std::out_of_range( "dense_vector_operations::set_value_at_point" );
        set_value_at_point( value, index, x );
    }

    norm_type max_element( vector_type_real &x ) const
    {
        return reduce_inst_(
            static_cast<ordinal_type>( real_traits_.get_loc_size( x ) ), real_traits_.get_raw_ptr( x ),
            -std::numeric_limits<norm_type>::infinity(), common::scfd_backend_ext::max_op<norm_type>()
        );
    }

    std::size_t argmax_element( vector_type_real &x ) const
    {
        return max_argmax_element( x ).second;
    }

    std::pair<norm_type, std::size_t> max_argmax_element( vector_type_real &x ) const
    {
        return common::scfd_backend_ext::arg_reduce<backend_type, norm_type, ordinal_type>::max_argmax(
            static_cast<ordinal_type>( real_traits_.get_loc_size( x ) ), real_traits_.get_raw_ptr( x )
        );
    }

    void assign_slices(
        const vector_type &x, const std::vector<std::pair<std::size_t, std::size_t>> &slices, vector_type &y
    ) const
    {
        const auto  x_size   = static_cast<std::size_t>( get_size( x ) );
        const auto  y_size   = static_cast<std::size_t>( get_size( y ) );
        std::size_t out_size = 0;
        for ( const auto &slice : slices )
        {
            if ( slice.first > slice.second || slice.second > x_size )
            {
                throw std::logic_error( "scfd_vector_operations::assign_slices: invalid slice range" );
            }
            out_size += slice.second - slice.first;
        }
        if ( out_size != y_size )
        {
            throw std::logic_error(
                "scfd_vector_operations::assign_slices: slice output size does not match destination vector size"
            );
        }

        std::vector<scalar_type> host_x( x_size );
        std::vector<scalar_type> host_y( y_size );
        get( x, host_x.data(), x_size );
        std::size_t out = 0;
        for ( const auto &slice : slices )
        {
            for ( std::size_t j = slice.first; j < slice.second; ++j )
            {
                host_y[out++] = host_x[j];
            }
        }
        set( host_y.data(), y, y_size );
    }

    void assign_skip_slices(
        const vector_type &x, const std::vector<std::pair<std::size_t, std::size_t>> &skip_slices, vector_type &y
    ) const
    {
        const auto x_size = static_cast<std::size_t>( get_size( x ) );
        const auto y_size = static_cast<std::size_t>( get_size( y ) );
        for ( const auto &slice : skip_slices )
        {
            if ( slice.first > slice.second || slice.second > x_size )
            {
                throw std::logic_error( "scfd_vector_operations::assign_skip_slices: invalid skip range" );
            }
        }

        std::vector<scalar_type> host_x( x_size );
        std::vector<scalar_type> host_y( y_size );
        get( x, host_x.data(), x_size );
        std::size_t out = 0;
        for ( std::size_t j = 0; j < x_size; ++j )
        {
            bool skip = false;
            for ( const auto &slice : skip_slices )
            {
                if ( j >= slice.first && j < slice.second )
                {
                    skip = true;
                    break;
                }
            }
            if ( !skip )
            {
                if ( out >= y_size )
                {
                    throw std::logic_error(
                        "scfd_vector_operations::assign_skip_slices: destination vector is too small"
                    );
                }
                host_y[out++] = host_x[j];
            }
        }
        if ( out != y_size )
        {
            throw std::logic_error(
                "scfd_vector_operations::assign_skip_slices: skip output size does not match destination vector size"
            );
        }
        set( host_y.data(), y, y_size );
    }

    void assign_random( vector_type &vec ) const
    {
        const std::size_t seed = random_seed_++;
        common::scfd_backend_ext::random_fill<backend_type, scalar_type, ordinal_type>::uniform01(
            for_each_inst_, static_cast<ordinal_type>( get_size( vec ) ), vt_.get_raw_ptr( vec ), seed
        );
    }

    void assign_random( vector_type &vec, scalar_type a, scalar_type b ) const
    {
        assign_random( vec );
        add_mul_scalar( a, b - a, vec );
    }

protected:
    void verify_max_loc_size( std::size_t size ) const
    {
        size = checked_size( size );
        if ( size != 0 && ( helper_.is_free() || size > static_cast<std::size_t>( helper_.size() ) ) )
        {
            free_owned( helper_ );
            helper_.init( static_cast<ordinal_type>( size ) );
        }
    }
    mutable VectorTraits vt_;

private:
    void ensure_real_capacity( std::size_t size ) const
    {
        size = checked_size( size );
        if ( size != 0 && ( helper_real_.is_free() || size > static_cast<std::size_t>( helper_real_.size() ) ) )
        {
            free_owned( helper_real_ );
            helper_real_.init( static_cast<ordinal_type>( size ) );
        }
    }
    void prepare_high_precision( std::size_t size ) const
    {
        size = checked_size( size );
        if ( size != high_precision_size_ )
        {
            high_precision_reduction_ = high_precision_reduction_type( static_cast<ordinal_type>( size ) );
            high_precision_size_      = size;
        }
    }

protected:
    template <class Index>
    static void validate_size( std::size_t size )
    {
        static_assert( std::is_integral<Index>::value, "SCFD ordinal types must be integral" );
        if constexpr ( std::numeric_limits<Index>::digits < std::numeric_limits<std::size_t>::digits )
        {
            if ( size > static_cast<std::size_t>( std::numeric_limits<Index>::max() ) )
            {
                throw std::invalid_argument( "SCFD vector size exceeds an ordinal range" );
            }
        }
    }

    static std::size_t checked_size( std::size_t size )
    {
        validate_size<ordinal_type>( size );
        validate_size<typename backend_type::ordinal_type>( size );
        validate_size<scfd::arrays::ordinal_type>( size );
        return size;
    }

    template <class Array>
    static void free_owned( Array &x )
    {
        if ( !x.is_free() )
        {
            if ( !x.is_own() )
            {
                x = Array();
                return;
            }
            x.free();
        }
    }

protected:
    mutable for_each_type                                 for_each_inst_;
    mutable reduce_type                                   reduce_inst_;
    mutable scfd::arrays::array<scalar_type, memory_type> helper_;

private:
    real_traits_type                                    real_traits_;
    mutable scfd::arrays::array<norm_type, memory_type> helper_real_;
    mutable high_precision_reduction_type               high_precision_reduction_;
    mutable std::size_t                                 high_precision_size_           = 0;
    mutable std::size_t                                 random_seed_                   = 1;
    mutable bool                                        last_reduction_high_precision_ = false;
};

} // namespace operations
} // namespace nmfd

#endif
