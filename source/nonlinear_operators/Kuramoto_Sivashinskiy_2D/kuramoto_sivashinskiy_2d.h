#ifndef __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_2D_REFACTORED_H__
#define __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_2D_REFACTORED_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <common/scfd_backend_ext/complex.h>
#include <discretization/fourier/codecs/inversion_odd_field.h>
#include <discretization/fourier/dealiasing_policy.h>
#include <discretization/fourier/initialization/low_mode_odd_field.h>
#include <discretization/fourier/normalized_fft.h>
#include <discretization/fourier/operations/derivative.h>
#include <discretization/fourier/operations/pseudospectral_product.h>
#include <discretization/fourier/operations/vector_advection.h>
#include <discretization/fourier/periodic_grid.h>
#include <discretization/fourier/r2c_index_space.h>
#include <discretization/fourier/wavevector_table.h>
#include <nonlinear_operators/detail/linear_nonlinear_terms.h>
#include <scfd/utils/device_tag.h>
#include <symmetry/fourier/periodic_affine_action_2d.h>
#include <symmetry/generated_finite_group.h>

namespace nonlinear_operators
{

template <
    class VectorOperations, class FFTBackend,
    template <class, class> class StateCodec = discretization::fourier::codecs::inversion_odd_field_2d>
class kuramoto_sivashinskiy_2d
{
public:
    struct is_periodic_orbit_reprojected
    {
        static const bool value = false;
    };

    using vector_operations_type = VectorOperations;
    using backend_type           = typename vector_operations_type::backend_type;
    using fft_backend_type       = FFTBackend;
    using scalar_type            = typename vector_operations_type::scalar_type;
    using T                      = scalar_type;
    using vector_type            = typename vector_operations_type::vector_type;
    using T_vec                  = vector_type;
    using ordinal_type           = typename vector_operations_type::ordinal_type;
    using transform_type =
        discretization::fourier::normalized_r2c_transform<backend_type, fft_backend_type, scalar_type, 2>;
    using complex_type          = typename transform_type::complex_type;
    using complex_traits        = ::common::scfd_backend_ext::complex_value_traits<complex_type>;
    using extent_type           = discretization::common::structured_extent<2>;
    using grid_type             = discretization::fourier::periodic_grid<scalar_type, 2>;
    using index_space_type      = discretization::fourier::r2c_index_space_2d;
    using spectral_field_type   = typename transform_type::spectral_field_type;
    using physical_field_type   = typename transform_type::physical_field_type;
    using state_codec_type      = StateCodec<vector_operations_type, complex_type>;
    using wavevector_table_type = discretization::fourier::wavevector_table_2d<backend_type, scalar_type>;
    using dealiasing_type       = discretization::fourier::two_thirds_dealiasing_2d;
    using product_type          = discretization::fourier::operations::pseudospectral_product_2d<
                 backend_type, fft_backend_type, scalar_type, dealiasing_type>;
    using for_each_type                = typename vector_operations_type::for_each_type;
    using copy_type                    = typename vector_operations_type::copy_type;
    using finite_symmetry_element_type = symmetry::fourier::periodic_affine_element_2d;
    using finite_symmetry_group_type   = symmetry::generated_finite_group<finite_symmetry_element_type>;

    kuramoto_sivashinskiy_2d(
        const scalar_type a, const scalar_type b, const std::size_t nx, const std::size_t ny,
        vector_operations_type           *vector_operations,
        const std::array<scalar_type, 2> &lengths =
            { scalar_type( 2 ) * std::acos( scalar_type( -1 ) ), scalar_type( 2 ) * std::acos( scalar_type( -1 ) ) }
    )
        : vector_operations_( require_vector_operations( vector_operations ) ), physical_extent_( nx, ny ),
          index_space_( physical_extent_ ), lengths_( lengths ), grid_( physical_extent_, lengths_ ),
          transform_( physical_extent_ ), wavevectors_( grid_, index_space_ ),
          codec_( vector_operations_, index_space_ ), product_( &transform_, dealiasing_type( index_space_ ) ), a_( a ),
          b_( b ), u_hat_( index_space_.spectral_extent() ), u_gradient_x_( index_space_.spectral_extent() ),
          u_gradient_y_( index_space_.spectral_extent() ), u_gradient_sum_( index_space_.spectral_extent() ),
          nonlinear_hat_( index_space_.spectral_extent() ), output_hat_( index_space_.spectral_extent() ),
          u0_hat_( index_space_.spectral_extent() ), u0_gradient_sum_( index_space_.spectral_extent() ),
          du_hat_( index_space_.spectral_extent() ), du_gradient_sum_( index_space_.spectral_extent() ),
          jacobian_product_1_( index_space_.spectral_extent() ), jacobian_product_2_( index_space_.spectral_extent() ),
          preconditioner_hat_( index_space_.spectral_extent() ), physical_output_( physical_extent_ )
    {
        vector_operations_->init_vector( u0_state_ );
        vector_operations_->start_use_vector( u0_state_ );
        vector_operations_->assign_scalar( scalar_type( 0 ), u0_state_ );
    }

    kuramoto_sivashinskiy_2d( const kuramoto_sivashinskiy_2d & )            = delete;
    kuramoto_sivashinskiy_2d &operator=( const kuramoto_sivashinskiy_2d & ) = delete;

    ~kuramoto_sivashinskiy_2d()
    {
        vector_operations_->stop_use_vector( u0_state_ );
        vector_operations_->free_vector( u0_state_ );
    }

    std::size_t size() const
    {
        return codec_.state_size();
    }
    std::size_t physical_size() const
    {
        return index_space_.physical_size();
    }
    std::size_t complex_size() const
    {
        return index_space_.complex_size();
    }
    std::size_t nx() const
    {
        return index_space_.nx();
    }
    std::size_t ny() const
    {
        return index_space_.ny();
    }

    const extent_type &physical_extent() const
    {
        return physical_extent_;
    }

    const std::array<scalar_type, 2> &domain_lengths() const
    {
        return lengths_;
    }

    const auto &state_modes() const
    {
        return codec_.state_modes();
    }

    finite_symmetry_group_type finite_symmetry_group() const
    {
        finite_symmetry_group_type group( finite_symmetry_element_type::identity() );
        using inversion_odd_codec_type =
            discretization::fourier::codecs::inversion_odd_field_2d<vector_operations_type, complex_type>;
        if constexpr ( std::is_same<state_codec_type, inversion_odd_codec_type>::value )
        {
            group.add_generator( "inversion_odd_half_shift_x", finite_symmetry_element_type::half_shift( 0 ) );
            group.add_generator( "inversion_odd_half_shift_y", finite_symmetry_element_type::half_shift( 1 ) );
        }
        if ( axes_are_interchangeable() )
        {
            group.add_generator( "axis_swap", finite_symmetry_element_type::swap_axes() );
        }
        group.finalize();
        return group;
    }

    template <class FiniteActionRegistry>
    auto
    configure_finite_symmetry_actions( FiniteActionRegistry &registry, const finite_symmetry_group_type &group ) const
    {
        return symmetry::fourier::register_periodic_affine_group_2d<
            FiniteActionRegistry, finite_symmetry_group_type, state_codec_type, complex_type>(
            registry, group, &codec_, index_space_, lengths_
        );
    }

    template <class FiniteActionRegistry>
    auto configure_finite_symmetry_actions( FiniteActionRegistry &registry ) const
    {
        const auto group = finite_symmetry_group();
        return configure_finite_symmetry_actions( registry, group );
    }

    scalar_type linear_multiplier( const std::size_t ix, const std::size_t iy, const scalar_type lambda ) const
    {
        const scalar_type kx        = grid_.wave_number( 0, index_space_.signed_x_mode( ix ) );
        const scalar_type ky        = grid_.wave_number( 1, index_space_.y_mode( iy ) );
        const scalar_type k_squared = kx * kx + ky * ky;
        return linear_symbol<true>( k_squared, lambda, b_ );
    }

    void F( const vector_type &state, const scalar_type lambda, vector_type &output )
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::all>( state, lambda, output );
    }

    void linear_residual( const vector_type &state, const scalar_type lambda, vector_type &output )
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::linear>( state, lambda, output );
    }

    void nonlinear_residual( const vector_type &state, const scalar_type lambda, vector_type &output )
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::nonlinear>( state, lambda, output );
    }

    void set_linearization_point( const vector_type &state, const scalar_type lambda )
    {
        vector_operations_->assign( state, u0_state_ );
        lambda0_ = lambda;
        codec_.unpack( state, u0_hat_ );
        compute_gradient_sum( u0_hat_, u0_gradient_sum_ );
    }

    void jacobian_u( const vector_type &direction, vector_type &output )
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::all>( direction, output );
    }

    void linear_jacobian_u( const vector_type &direction, vector_type &output )
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::linear>( direction, output );
    }

    void nonlinear_jacobian_u( const vector_type &direction, vector_type &output )
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::nonlinear>( direction, output );
    }

    void jacobian_u_adjoint( const vector_type &cotangent, vector_type &output )
    {
        apply_spatial_jacobian_adjoint<detail::linear_nonlinear_terms::all>( cotangent, output );
    }

    void linear_jacobian_u_adjoint( const vector_type &cotangent, vector_type &output )
    {
        apply_spatial_jacobian_adjoint<detail::linear_nonlinear_terms::linear>( cotangent, output );
    }

    void nonlinear_jacobian_u_adjoint( const vector_type &cotangent, vector_type &output )
    {
        apply_spatial_jacobian_adjoint<detail::linear_nonlinear_terms::nonlinear>( cotangent, output );
    }

    void jacobian_alpha( vector_type &output )
    {
        jacobian_alpha( u0_state_, lambda0_, output );
    }

    void jacobian_alpha( const vector_type &state, const scalar_type &, vector_type &output )
    {
        codec_.unpack( state, u_hat_ );
        compute_nonlinearity( u_hat_, u_gradient_sum_, nonlinear_hat_ );
        assemble<detail::linear_nonlinear_terms::all, false>(
            u_hat_, nonlinear_hat_, scalar_type( 1 ), output_hat_
        );
        codec_.pack( output_hat_, output );
    }

    void preconditioner_jacobian_u( vector_type &rhs_to_solution ) const
    {
        preconditioner_jacobian_affine_u( rhs_to_solution, scalar_type( 1 ), scalar_type( 0 ) );
    }

    void solve_jacobian_system( vector_type &rhs_to_solution ) const
    {
        preconditioner_jacobian_u( rhs_to_solution );
    }

    void preconditioner_jacobian_affine_u(
        vector_type &rhs_to_solution, const scalar_type jacobian_scale, const scalar_type identity_shift
    ) const
    {
        codec_.unpack( rhs_to_solution, preconditioner_hat_ );
        complex_type      *values                  = preconditioner_hat_.data();
        const scalar_type *k_squared               = wavevectors_.k_squared().data();
        const scalar_type  lambda                  = lambda0_;
        const scalar_type  b                       = b_;
        const scalar_type  pole_relative_tolerance = std::sqrt( std::numeric_limits<scalar_type>::epsilon() );
        const ordinal_type count                   = static_cast<ordinal_type>( complex_size() );
        for_each_type      for_each;
        for_each(
            [=] __DEVICE_TAG__( const ordinal_type index ) {
                const scalar_type k2                = k_squared[index];
                const scalar_type jacobian_diagonal = jacobian_scale * ( -lambda * k2 + b * k2 * k2 );
                const scalar_type diagonal          = jacobian_diagonal + identity_shift;
                const scalar_type absolute_diagonal = diagonal < scalar_type( 0 ) ? -diagonal : diagonal;
                const scalar_type absolute_jacobian =
                    jacobian_diagonal < scalar_type( 0 ) ? -jacobian_diagonal : jacobian_diagonal;
                const scalar_type absolute_shift = identity_shift < scalar_type( 0 ) ? -identity_shift : identity_shift;
                const scalar_type scale          = absolute_jacobian + absolute_shift;
                const scalar_type threshold =
                    pole_relative_tolerance * ( scale > scalar_type( 1 ) ? scale : scalar_type( 1 ) );
                if ( absolute_diagonal > threshold )
                {
                    values[index] = complex_traits::make(
                        complex_traits::real( values[index] ) / diagonal,
                        complex_traits::imag( values[index] ) / diagonal
                    );
                }
            },
            count
        );
        for_each.wait();
        codec_.pack( preconditioner_hat_, rhs_to_solution );
    }

    void preconditioner_jacobian_affine_u_adjoint(
        vector_type &rhs_to_solution, const scalar_type jacobian_scale, const scalar_type identity_shift
    ) const
    {
        preconditioner_jacobian_affine_u( rhs_to_solution, jacobian_scale, identity_shift );
    }

    std::pair<scalar_type, scalar_type> preconditioner_jacobian_affine_diagonal_range(
        const scalar_type jacobian_scale, const scalar_type identity_shift
    ) const
    {
        scalar_type minimum = std::numeric_limits<scalar_type>::infinity();
        scalar_type maximum = scalar_type( 0 );
        visit_nonzero_modes( [&]( const scalar_type k_squared ) {
            const scalar_type diagonal =
                jacobian_scale * ( -lambda0_ * k_squared + b_ * k_squared * k_squared ) + identity_shift;
            const scalar_type absolute = diagonal < scalar_type( 0 ) ? -diagonal : diagonal;
            minimum                    = std::min( minimum, absolute );
            maximum                    = std::max( maximum, absolute );
        } );
        return { minimum, maximum };
    }

    scalar_type preconditioner_jacobian_affine_min_relative_diagonal(
        const scalar_type jacobian_scale, const scalar_type identity_shift
    ) const
    {
        scalar_type minimum = std::numeric_limits<scalar_type>::infinity();
        visit_nonzero_modes( [&]( const scalar_type k_squared ) {
            const scalar_type jacobian_diagonal =
                jacobian_scale * ( -lambda0_ * k_squared + b_ * k_squared * k_squared );
            const scalar_type diagonal = jacobian_diagonal + identity_shift;
            const scalar_type absolute = diagonal < scalar_type( 0 ) ? -diagonal : diagonal;
            const scalar_type absolute_jacobian =
                jacobian_diagonal < scalar_type( 0 ) ? -jacobian_diagonal : jacobian_diagonal;
            const scalar_type absolute_shift = identity_shift < scalar_type( 0 ) ? -identity_shift : identity_shift;
            const scalar_type scale          = absolute_jacobian + absolute_shift;
            minimum = std::min( minimum, scale > scalar_type( 0 ) ? absolute / scale : absolute );
        } );
        return minimum;
    }

    void physical_solution( const vector_type &state, vector_type &output )
    {
        if ( vector_operations_->get_size( output ) != physical_size() )
        {
            throw std::invalid_argument( "kuramoto_sivashinskiy_2d physical output size mismatch" );
        }
        codec_.unpack( state, u_hat_ );
        transform_.inverse( u_hat_, physical_output_ );
        copy_type()( static_cast<ordinal_type>( physical_size() ), physical_output_.data(), output.raw_ptr() );
    }

    void physical_solution( vector_type &state, vector_type &output )
    {
        physical_solution( static_cast<const vector_type &>( state ), output );
    }

    void project( vector_type & )
    {
    }

    void exact_solution( const scalar_type &, vector_type &output )
    {
        vector_operations_->assign_scalar( scalar_type( 0 ), output );
    }

    scalar_type check_solution_quality( const vector_type &state )
    {
        vector_type residual;
        vector_operations_->init_vector( residual );
        vector_operations_->start_use_vector( residual );
        F( state, lambda0_, residual );
        const scalar_type result = vector_operations_->norm_l2( residual );
        vector_operations_->stop_use_vector( residual );
        vector_operations_->free_vector( residual );
        return result;
    }

    void norm_bifurcation_diagram( const vector_type &state, std::vector<scalar_type> &result ) const
    {
        std::vector<scalar_type> host( std::min<std::size_t>( 2, size() ), scalar_type( 0 ) );
        if ( !host.empty() )
        {
            vector_operations_->get( state, host.data(), host.size() );
        }
        result.clear();
        result.push_back( vector_operations_->norm_l2( state ) );
        result.push_back( host.empty() ? scalar_type( 0 ) : host[0] );
        result.push_back( host.size() < 2 ? scalar_type( 0 ) : host[1] );
    }

    std::vector<std::string> norm_bifurcation_diagram_labels() const
    {
        return { "l2_norm", "state_0", "state_1" };
    }

    void randomize_vector( vector_type &output )
    {
        randomize_vector( output, lambda0_, random_profile_counter_++ );
    }

    void randomize_stability_vector( vector_type &output )
    {
        vector_operations_->assign_random(
            output,
            scalar_type( -1 ),
            scalar_type( 1 )
        );
    }

    void randomize_vector( vector_type &output, const scalar_type parameter, const std::uint64_t seed_id )
    {
        discretization::fourier::initialization::fill_low_mode_odd_physical_field_2d<backend_type, scalar_type>(
            physical_output_, nx(), ny(), parameter, b_, seed_id
        );
        transform_.forward( physical_output_, u_hat_ );
        codec_.pack( u_hat_, output );

        const scalar_type norm = vector_operations_->norm_l2( output );
        if ( !( norm > std::numeric_limits<scalar_type>::epsilon() ) )
        {
            throw std::runtime_error( "kuramoto_sivashinskiy_2d generated a zero deflation seed" );
        }
    }

    vector_operations_type *get_vec_ops_ref()
    {
        return vector_operations_;
    }
    const vector_operations_type *get_vec_ops_ref() const
    {
        return vector_operations_;
    }
    const state_codec_type &state_codec() const
    {
        return codec_;
    }

private:
    bool axes_are_interchangeable() const
    {
        if ( nx() != ny() )
        {
            return false;
        }
        const scalar_type scale = std::max( lengths_[0], lengths_[1] );
        return std::abs( lengths_[0] - lengths_[1] ) <=
               scalar_type( 64 ) * std::numeric_limits<scalar_type>::epsilon() * scale;
    }

    static vector_operations_type *require_vector_operations( vector_operations_type *vector_operations )
    {
        if ( vector_operations == nullptr )
        {
            throw std::invalid_argument( "kuramoto_sivashinskiy_2d requires vector operations" );
        }
        return vector_operations;
    }

    void compute_gradient_sum( const spectral_field_type &input, spectral_field_type &sum )
    {
        discretization::fourier::operations::derivative<backend_type>( input, wavevectors_, 0, u_gradient_x_ );
        discretization::fourier::operations::derivative<backend_type>( input, wavevectors_, 1, u_gradient_y_ );
        discretization::fourier::operations::add_spectra<backend_type>( u_gradient_x_, u_gradient_y_, sum );
    }

    void compute_nonlinearity(
        const spectral_field_type &input, spectral_field_type &gradient_sum, spectral_field_type &nonlinearity
    )
    {
        compute_gradient_sum( input, gradient_sum );
        product_.apply( input, gradient_sum, nonlinearity );
    }

    template <bool IncludeBiharmonic>
    __DEVICE_TAG__ static scalar_type linear_symbol(
        const scalar_type k_squared,
        const scalar_type lambda,
        const scalar_type biharmonic_scale
    )
    {
        scalar_type value = -lambda * k_squared;
        if constexpr ( IncludeBiharmonic )
        {
            value += biharmonic_scale * k_squared * k_squared;
        }
        return value;
    }

    template <detail::linear_nonlinear_terms Terms>
    void evaluate_spatial_residual(
        const vector_type &state,
        const scalar_type lambda,
        vector_type &output
    )
    {
        codec_.unpack( state, u_hat_ );
        if constexpr ( detail::includes_nonlinear<Terms>() )
        {
            compute_nonlinearity( u_hat_, u_gradient_sum_, nonlinear_hat_ );
        }
        assemble<Terms, true>( u_hat_, nonlinear_hat_, lambda, output_hat_ );
        codec_.pack( output_hat_, output );
    }

    template <detail::linear_nonlinear_terms Terms>
    void apply_spatial_jacobian( const vector_type &direction, vector_type &output )
    {
        codec_.unpack( direction, du_hat_ );
        if constexpr ( detail::includes_nonlinear<Terms>() )
        {
            compute_gradient_sum( du_hat_, du_gradient_sum_ );
            product_.apply( du_hat_, u0_gradient_sum_, jacobian_product_1_ );
            product_.apply( u0_hat_, du_gradient_sum_, jacobian_product_2_ );
            discretization::fourier::operations::add_spectra<backend_type>(
                jacobian_product_1_, jacobian_product_2_, nonlinear_hat_
            );
        }
        assemble<Terms, true>( du_hat_, nonlinear_hat_, lambda0_, output_hat_ );
        codec_.pack( output_hat_, output );
    }

    template <detail::linear_nonlinear_terms Terms>
    void apply_spatial_jacobian_adjoint(
        const vector_type &cotangent,
        vector_type &output )
    {
        codec_.pack_adjoint( cotangent, output_hat_ );
        if constexpr ( detail::includes_nonlinear<Terms>() )
        {
            product_.apply_left_adjoint(
                u0_gradient_sum_, output_hat_, jacobian_product_1_ );
            product_.apply_right_adjoint(
                u0_hat_, output_hat_, jacobian_product_2_ );
            discretization::fourier::operations::derivative_adjoint<backend_type>(
                jacobian_product_2_, wavevectors_, 0, u_gradient_x_ );
            discretization::fourier::operations::derivative_adjoint<backend_type>(
                jacobian_product_2_, wavevectors_, 1, u_gradient_y_ );
            discretization::fourier::operations::add_spectra<backend_type>(
                u_gradient_x_, u_gradient_y_, nonlinear_hat_ );
            discretization::fourier::operations::add_spectra<backend_type>(
                jacobian_product_1_, nonlinear_hat_, du_hat_ );
        }
        assemble<Terms, true>(
            output_hat_, du_hat_, lambda0_, u_hat_ );
        codec_.unpack_adjoint( u_hat_, output );
    }

public:
    // Public because NVCC requires enclosing functions of extended device lambdas to be accessible.
    template <detail::linear_nonlinear_terms Terms, bool IncludeBiharmonic>
    void assemble(
        const spectral_field_type &state,
        const spectral_field_type &nonlinearity,
        const scalar_type lambda,
        spectral_field_type &output
    ) const
    {
        const complex_type *state_values     = state.data();
        const complex_type *nonlinear_values = nonlinearity.data();
        complex_type       *output_values    = output.data();
        const scalar_type  *k_squared        = wavevectors_.k_squared().data();
        const scalar_type   a                = a_;
        const scalar_type   b                = b_;
        const ordinal_type  count            = static_cast<ordinal_type>( complex_size() );
        for_each_type       for_each;
        for_each(
            [state_values, nonlinear_values, output_values, k_squared, a, b, lambda]
            __DEVICE_TAG__( const ordinal_type index ) {
                scalar_type real_value = scalar_type( 0 );
                scalar_type imag_value = scalar_type( 0 );
                if constexpr ( detail::includes_linear<Terms>() )
                {
                    const scalar_type linear = linear_symbol<IncludeBiharmonic>( k_squared[index], lambda, b );
                    real_value += linear * complex_traits::real( state_values[index] );
                    imag_value += linear * complex_traits::imag( state_values[index] );
                }
                if constexpr ( detail::includes_nonlinear<Terms>() )
                {
                    const scalar_type nonlinear_scale = lambda * a;
                    real_value += nonlinear_scale * complex_traits::real( nonlinear_values[index] );
                    imag_value += nonlinear_scale * complex_traits::imag( nonlinear_values[index] );
                }
                output_values[index] = complex_traits::make( real_value, imag_value );
            },
            count
        );
        for_each.wait();
    }

private:
    template <class Function>
    void visit_nonzero_modes( Function &&function ) const
    {
        for ( std::size_t ix = 0; ix < index_space_.nx(); ++ix )
        {
            for ( std::size_t iy = 0; iy < index_space_.my(); ++iy )
            {
                if ( ix == 0 && iy == 0 )
                {
                    continue;
                }
                const scalar_type kx = grid_.wave_number( 0, index_space_.signed_x_mode( ix ) );
                const scalar_type ky = grid_.wave_number( 1, index_space_.y_mode( iy ) );
                function( kx * kx + ky * ky );
            }
        }
    }

    vector_operations_type    *vector_operations_;
    extent_type                physical_extent_;
    index_space_type           index_space_;
    std::array<scalar_type, 2> lengths_;
    grid_type                  grid_;
    mutable transform_type     transform_;
    wavevector_table_type      wavevectors_;
    mutable state_codec_type   codec_;
    mutable product_type       product_;
    scalar_type                a_;
    scalar_type                b_;
    scalar_type                lambda0_                = scalar_type( 0 );
    unsigned int               random_profile_counter_ = 0;

    vector_type                 u0_state_;
    spectral_field_type         u_hat_;
    spectral_field_type         u_gradient_x_;
    spectral_field_type         u_gradient_y_;
    spectral_field_type         u_gradient_sum_;
    spectral_field_type         nonlinear_hat_;
    spectral_field_type         output_hat_;
    spectral_field_type         u0_hat_;
    spectral_field_type         u0_gradient_sum_;
    spectral_field_type         du_hat_;
    spectral_field_type         du_gradient_sum_;
    spectral_field_type         jacobian_product_1_;
    spectral_field_type         jacobian_product_2_;
    mutable spectral_field_type preconditioner_hat_;
    physical_field_type         physical_output_;
};

} // namespace nonlinear_operators

#endif
