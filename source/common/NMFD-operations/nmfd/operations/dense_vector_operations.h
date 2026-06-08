#ifndef __NMFD_DENSE_VECTOR_OPERATIONS_H__
#define __NMFD_DENSE_VECTOR_OPERATIONS_H__

#include <scfd/utils/todo.h>

#include <nmfd/operations/kernels/dense_vector_space.h>
#include <nmfd/operations/vector_operations_base.h>
#include <nmfd/operations/default_multivector_operations_base.h>

namespace nmfd
{
namespace operations
{

template <class VectorTraits, class Backend, class Ordinal = std::ptrdiff_t>
class dense_vector_operations : public default_multivector_operations_base<
                                    dense_vector_operations<VectorTraits, Backend, Ordinal>,
                                    typename VectorTraits::scalar_type, typename VectorTraits::vector_type, Ordinal>
{
    using multivector_operations_base = default_multivector_operations_base<
        dense_vector_operations<VectorTraits, Backend, Ordinal>, typename VectorTraits::scalar_type,
        typename VectorTraits::vector_type, Ordinal>;

public:
    using multivector_type = typename multivector_operations_base::multivector_type;
    using scalar_type      = typename VectorTraits::scalar_type;
    using vector_type      = typename VectorTraits::vector_type;
    using ordinal_type     = Ordinal;
    using Ord              = Ordinal;
    using for_each_type    = typename Backend::template for_each_type<Ordinal>;
    using reduce_type      = typename Backend::reduce_type;
    using memory_type      = typename Backend::memory_type;

public:
    using scalar_prod_kernel       = kernels::scalar_prod<scalar_type, vector_type>;
    using add_mul_scalar_kernel    = kernels::add_mul_scalar<scalar_type>;
    using assign_scalar_kernel     = kernels::assign_scalar<scalar_type>;
    using norm_inf_kernel          = kernels::norm_inf<scalar_type, vector_type>;
    using sum_kernel               = kernels::sum<scalar_type, vector_type>;
    using asum_kernel              = kernels::asum<scalar_type, vector_type>;
    using assign_kernel            = kernels::assign<scalar_type>;
    using assign_lin_comb_1_kernel = kernels::assign_lin_comb_1<scalar_type>;
    using assign_lin_comb_2_kernel = kernels::assign_lin_comb_2<scalar_type>;
    using add_lin_comb_1_kernel    = kernels::add_lin_comb_1<scalar_type>;
    using add_lin_comb_2_kernel    = kernels::add_lin_comb_2<scalar_type>;
    using add_lin_comb_3_kernel    = kernels::add_lin_comb_3<scalar_type>;
    using make_abs_copy_kernel     = kernels::make_abs_copy<scalar_type>;
    using make_abs_kernel          = kernels::make_abs<scalar_type>;
    using max_pointwise_kernel     = kernels::max_pointwise<scalar_type>;
    using min_pointwise_kernel     = kernels::min_pointwise<scalar_type>;
    using mul_pointwise_kernel     = kernels::mul_pointwise<scalar_type>;
    using div_pointwise_kernel     = kernels::div_pointwise<scalar_type>;
    using assign_random_kernel     = kernels::assign_random<scalar_type>;

public:
    using multivector_operations_base::add_lin_comb;
    using multivector_operations_base::assign;
    using multivector_operations_base::scalar_prod;
    using multivector_operations_base::scalar_prod_l2;

public:
    dense_vector_operations() = default;

    template <typename... Args>
    dense_vector_operations( Args &&...args ) : vt_( std::forward<Args>( args )... )
    {
        vt_.alloc( vt_.loc_size(), helper_ );
    }

    void init_vector( Ordinal loc_sz, vector_type &v ) const
    {
        vt_.alloc( loc_sz, v );
    }

    [[nodiscard]] Ordinal get_loc_size( const vector_type &x ) const
    {
        return vt_.get_loc_size( x );
    }

    bool is_valid_number( const vector_type &x ) const
    {
        return std::isfinite( norm2_sq( x ) );
    }

    [[nodiscard]] scalar_type scalar_prod( const vector_type &x, const vector_type &y ) const
    {
        verify_max_loc_size( get_loc_size( x ) );
        for_each_inst_(
            scalar_prod_kernel{ vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ), vt_.get_raw_ptr( helper_ ) },
            get_loc_size( x )
        );
        return reduce_inst_( get_loc_size( x ), vt_.get_raw_ptr( helper_ ), scalar_type{ 0.f } );
    }
    [[nodiscard]] scalar_type scalar_prod_l2( const vector_type &x, const vector_type &y ) const
    {
        return scalar_prod( x, y );
    }

    [[nodiscard]] scalar_type norm( const vector_type &x ) const
    {
        return std::sqrt( scalar_prod( x, x ) );
    }
    [[nodiscard]] scalar_type norm_sq( const vector_type &x ) const
    {
        return scalar_prod( x, x );
    }
    [[nodiscard]] scalar_type norm2_sq( const vector_type &x ) const
    {
        return norm_sq( x );
    }
    [[nodiscard]] scalar_type norm2( const vector_type &x ) const
    {
        return std::sqrt( scalar_prod( x, x ) );
    }
    [[nodiscard]] scalar_type norm_l2_sq( const vector_type &x ) const
    {
        return norm_sq( x );
    }
    [[nodiscard]] scalar_type norm_l2( const vector_type &x ) const
    {
        return std::sqrt( scalar_prod( x, x ) );
    }
    /// asum(x)
    [[nodiscard]] scalar_type norm1( const vector_type &x ) const
    {
        return asum( x );
    }
    /// returns some weighted L1/l1 norm (problem dependent)
    [[nodiscard]] scalar_type norm_l1( const vector_type &x ) const
    {
        return norm1( x );
    }
    [[nodiscard]] scalar_type norm_inf( const vector_type &x ) const
    {
        scalar_type max_val = 0.0;
        for ( size_t j = 0; j < get_loc_size( x ); j++ )
        {
            const auto x_j = std::abs( get_value_at_point( j, x ) );
            max_val        = max_val < x_j ? x_j : max_val;
        }
        return max_val;
    }
    /// returns maximum of all elements absolute weighted values (problem
    /// dependent)
    [[nodiscard]] scalar_type norm_l_inf( const vector_type &x ) const
    {
        return norm_inf( x );
    }

    [[nodiscard]] scalar_type sum( const vector_type &x ) const
    {
        verify_max_loc_size( get_loc_size( x ) );
        for_each_inst_( sum_kernel{ vt_.get_raw_ptr( x ), vt_.get_raw_ptr( helper_ ) }, get_loc_size( x ) );
        return reduce_inst_( get_loc_size( x ), vt_.get_raw_ptr( helper_ ), scalar_type{ 0 } );
    }
    [[nodiscard]] scalar_type asum( const vector_type &x ) const
    {
        verify_max_loc_size( get_loc_size( x ) );
        for_each_inst_( asum_kernel{ vt_.get_raw_ptr( x ), vt_.get_raw_ptr( helper_ ) }, get_loc_size( x ) );
        return reduce_inst_( get_loc_size( x ), vt_.get_raw_ptr( helper_ ), scalar_type{ 0 } );
    }

    scalar_type normalize( vector_type &x ) const
    {
        auto norm_x = norm( x );
        if ( norm_x > 0.0 )
        {
            scale( static_cast<scalar_type>( 1.0 ) / norm_x, x );
        }
        return norm_x;
    }

    void set_value_at_point( scalar_type val_x, size_t at, vector_type &x ) const
    {
        memory_type::copy_from_host( sizeof( scalar_type ), &val_x, vt_.get_raw_ptr( x ) + at );
    }
    scalar_type get_value_at_point( size_t at, const vector_type &x ) const
    {
        scalar_type result;
        memory_type::copy_to_host( sizeof( scalar_type ), vt_.get_raw_ptr( x ) + at, &result );
        return result;
    }
    // calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar( const scalar_type scalar, vector_type &x ) const
    {
        for_each_inst_( assign_scalar_kernel{ scalar, vt_.get_raw_ptr( x ) }, get_loc_size( x ) );
    }
    // calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar( const scalar_type scalar, const scalar_type mul_x, vector_type &x ) const
    {
        for_each_inst_( add_mul_scalar_kernel{ scalar, mul_x, vt_.get_raw_ptr( x ) }, get_loc_size( x ) );
    }
    void scale( scalar_type scale, vector_type &x ) const
    {
        add_mul_scalar( static_cast<scalar_type>( 0.0 ), scale, x );
    }
    // copy: y := x
    void assign( const vector_type &x, vector_type &y ) const
    {
        for_each_inst_( assign_kernel{ vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x ) );
    }
    // calc: y := mul_x*x
    void assign_lin_comb( scalar_type mul_x, const vector_type &x, vector_type &y ) const
    {
        for_each_inst_(
            assign_lin_comb_1_kernel{ mul_x, vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x )
        );
    }

    // calc: z := mul_x*x + mul_y*y
    void assign_lin_comb(
        scalar_type mul_x, const vector_type &x, scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        for_each_inst_(
            assign_lin_comb_2_kernel{ mul_x, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( z ) },
            get_loc_size( x )
        );
    }
    // calc: result := mul_x*x + mul_y*y (alias for assign_lin_comb with different
    // argument order)
    void assign_mul(
        scalar_type mul_x, const vector_type &x, scalar_type mul_y, const vector_type &y, vector_type &result
    ) const
    {
        assign_lin_comb( mul_x, x, mul_y, y, result );
    }
    // calc: y := mul_x*x + y
    void add_lin_comb( scalar_type mul_x, const vector_type &x, vector_type &y ) const
    {
        for_each_inst_( add_lin_comb_1_kernel{ mul_x, vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x ) );
    }
    // calc: y := mul_x*x + mul_y*y
    void add_lin_comb( scalar_type mul_x, const vector_type &x, scalar_type mul_y, vector_type &y ) const
    {
        for_each_inst_(
            add_lin_comb_2_kernel{ mul_x, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ) }, get_loc_size( x )
        );
    }
    // calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_lin_comb(
        scalar_type mul_x, const vector_type &x, scalar_type mul_y, const vector_type &y, scalar_type mul_z,
        vector_type &z
    ) const
    {
        for_each_inst_(
            add_lin_comb_3_kernel{
                mul_x, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), mul_z, vt_.get_raw_ptr( z )
            },
            get_loc_size( x )
        );
    }

    void make_abs_copy( const vector_type &x, vector_type &y ) const
    {
        for_each_inst_( make_abs_copy_kernel{ vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x ) );
    }
    void make_abs( vector_type &x ) const
    {
        for_each_inst_( make_abs_kernel{ vt_.get_raw_ptr( x ) }, get_loc_size( x ) );
    }
    // y_j = max(x_j,y_j,sc)
    void max_pointwise( const scalar_type sc, const vector_type &x, vector_type &y ) const
    {
        for_each_inst_( max_pointwise_kernel{ sc, vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x ) );
    }
    void max_pointwise( const scalar_type sc, vector_type &y ) const
    {
        for_each_inst_( max_pointwise_kernel{ sc, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( y ) }, get_loc_size( y ) );
    }
    // y_j = min(x_j,y_j,sc)
    void min_pointwise( const scalar_type sc, const vector_type &x, vector_type &y ) const
    {
        for_each_inst_( min_pointwise_kernel{ sc, vt_.get_raw_ptr( x ), vt_.get_raw_ptr( y ) }, get_loc_size( x ) );
    }
    void min_pointwise( const scalar_type sc, vector_type &y ) const
    {
        for_each_inst_( min_pointwise_kernel{ sc, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( y ) }, get_loc_size( y ) );
    }
    // calc: x := x*mul_y*y
    void mul_pointwise( vector_type &x, const scalar_type mul_y, const vector_type &y ) const
    {
        for_each_inst_(
            mul_pointwise_kernel{
                scalar_type{ 1 }, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( x )
            },
            get_loc_size( x )
        );
    }
    // calc: z := mul_x*x*mul_y*y
    void mul_pointwise(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        for_each_inst_(
            mul_pointwise_kernel{ mul_x, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( z ) },
            get_loc_size( x )
        );
    }
    // calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const
    {
        for_each_inst_(
            div_pointwise_kernel{ mul_x, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( z ) },
            get_loc_size( x )
        );
    }
    // calc: x := x/(mul_y*y)
    void div_pointwise( vector_type &x, const scalar_type mul_y, const vector_type &y ) const
    {
        for_each_inst_(
            div_pointwise_kernel{
                scalar_type{ 1 }, vt_.get_raw_ptr( x ), mul_y, vt_.get_raw_ptr( y ), vt_.get_raw_ptr( x )
            },
            get_loc_size( x )
        );
    }

    // TODO:!
    /*std::pair<scalar_type, size_t> max_argmax_element(vector_type& y) const
{
  auto max_iterator = std::max_element(y.begin(), y.end());
  size_t argmax = std::distance(y.begin(), max_iterator);

  return {*max_iterator, argmax};
}

scalar_type max_element(vector_type& x)const
{
  auto ret = max_argmax_element(x);
  return ret.first;
}

size_t argmax_element(vector_type& x)const
{
  auto ret = max_argmax_element(x);
  return ret.second;
}*/

    void assign_random( vector_type &x, const scalar_type from = 0, const scalar_type to = 1 ) const
    {
        unsigned int seed = static_cast<unsigned int>( std::rand() );
        for_each_inst_( assign_random_kernel{ from, to - from, vt_.get_raw_ptr( x ), seed }, get_loc_size( x ) );
    }

    void
    assign_slices( const vector_type &x, const std::vector<std::pair<size_t, size_t>> slices, vector_type &y ) const
    {
        SCFD_TODO( "Implement assign_slices" );
    }

    void assign_skip_slices(
        const vector_type &x, const std::vector<std::pair<size_t, size_t>> skip_slices, vector_type &y
    ) const
    {
        SCFD_TODO( "Implement assign_skip_lices" );
    }

protected:
    void verify_max_loc_size( size_t loc_size ) const
    {
        if ( vt_.get_loc_size( helper_ ) < loc_size )
        {
            vt_.dealloc( helper_ );
            vt_.alloc( loc_size, helper_ );
        }
    }

    mutable VectorTraits vt_;
    mutable vector_type  helper_;
    for_each_type        for_each_inst_;
    reduce_type          reduce_inst_;
};

} // namespace operations

} // namespace nmfd

#endif
