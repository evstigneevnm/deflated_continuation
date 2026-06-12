#ifndef __SYMMETRY_FOURIER_FOURIER_SLICE_1D_H__
#define __SYMMETRY_FOURIER_FOURIER_SLICE_1D_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <common/scfd_backend_ext/complex.h>
#include <symmetry/fourier/mode_access.h>
#include <symmetry/fourier/phase_conditions.h>
#include <symmetry/fourier/translation_generators.h>
#include <symmetry/slice_data.h>
#include <symmetry/slice_projector.h>

namespace symmetry
{
namespace fourier
{

template <class Complex>
class fourier_slice_1d
{
public:
    using complex_type = Complex;
    using traits_type = common::scfd_backend_ext::complex_value_traits<complex_type>;
    using real_type = typename traits_type::real_type;
    using generic_slice_data_type = symmetry::slice_data<real_type, 1>;

    struct slice_data : generic_slice_data_type
    {
        std::size_t mode = 0;
        real_type shift = real_type( 0 );
        real_type selected_abs = real_type( 0 );
        real_type selected_real_on_slice = real_type( 0 );
        real_type slice_matrix = real_type( 0 );
        real_type lsq_objective = real_type( 0 );
        std::vector<std::size_t> active_modes;

        std::size_t residual_group_order() const
        {
            return this->active() ? this->residual_group_order_value : std::size_t( 1 );
        }
    };

    struct projection_info
    {
        nmfd::operations::linalg::small_solve_status status =
            nmfd::operations::linalg::small_solve_status::success;
        real_type alpha = real_type( 0 );
        real_type phase_value = real_type( 0 );
        real_type slice_matrix = real_type( 0 );

        bool ok() const
        {
            return status == nmfd::operations::linalg::small_solve_status::success;
        }
    };

    explicit fourier_slice_1d(
        const real_type active_mode_tolerance = default_active_mode_tolerance(),
        const real_type singular_tolerance = real_type( 0 ),
        const real_type condition_tolerance = real_type( 0 )
    )
        : active_mode_tolerance_( active_mode_tolerance ),
          singular_tolerance_( singular_tolerance ),
          condition_tolerance_( condition_tolerance )
    {
    }

    static real_type default_active_mode_tolerance()
    {
        return std::sqrt( std::numeric_limits<real_type>::epsilon() );
    }

    slice_data choose_slice_data(
        const complex_type *spectrum, const std::size_t size, const std::size_t preferred_mode = 0
    ) const
    {
        if ( spectrum == nullptr )
            throw std::invalid_argument( "fourier_slice_1d::choose_slice_data got null spectrum" );

        slice_data data;
        data.group_dimension = 1;
        data.tolerance = active_mode_tolerance_;

        std::size_t selected_mode = 0;
        const_packed_positive_mode_view<complex_type> modes( spectrum, size );
        if ( preferred_mode != 0 && modes.has_mode( preferred_mode ) &&
             coefficient_abs( modes.mode( preferred_mode ) ) > active_mode_tolerance_ )
        {
            selected_mode = preferred_mode;
        }
        else
        {
            for ( std::size_t mode = 1; mode < size; ++mode )
            {
                if ( coefficient_abs( modes.mode( mode ) ) > active_mode_tolerance_ )
                {
                    selected_mode = mode;
                    break;
                }
            }
        }

        if ( selected_mode == 0 )
            return data;

        const complex_type coeff = modes.mode( selected_mode );
        const real_type real_part = traits_type::real( coeff );
        const real_type imag_part = traits_type::imag( coeff );
        const real_type magnitude = coefficient_abs( coeff );

        data.mode = selected_mode;
        data.active_rank = 1;
        data.residual_group_order_value = selected_mode;
        data.selected_abs = magnitude;
        data.selected_real_on_slice = magnitude;
        data.shift = -std::atan2( imag_part, real_part ) / static_cast<real_type>( selected_mode );
        data.set_shift( 0, data.shift );
        data.slice_matrix = static_cast<real_type>( selected_mode ) * magnitude;
        return data;
    }

    void apply_shift(
        const complex_type *source, complex_type *destination, const std::size_t size, const real_type shift
    ) const
    {
        apply_shift_1d( source, destination, size, shift );
    }

    std::vector<complex_type> apply_shift(
        const std::vector<complex_type> &source, const real_type shift
    ) const
    {
        std::vector<complex_type> destination( source.size() );
        apply_shift( source.data(), destination.data(), source.size(), shift );
        return destination;
    }

    void stabilize( const complex_type *source, complex_type *destination, const std::size_t size, slice_data &data ) const
    {
        data = choose_slice_data( source, size, data.mode );
        apply_shift( source, destination, size, data.shift );
    }

    std::vector<complex_type> stabilize( const std::vector<complex_type> &source, slice_data &data ) const
    {
        std::vector<complex_type> destination( source.size() );
        stabilize( source.data(), destination.data(), source.size(), data );
        return destination;
    }

    real_type phase_value( const slice_data &data, const complex_type *vector, const std::size_t size ) const
    {
        if ( !data.active() )
            return real_type( 0 );
        if ( vector == nullptr )
            throw std::invalid_argument( "fourier_slice_1d::phase_value got null vector" );
        if ( data.mode >= size )
            throw std::out_of_range( "fourier_slice_1d::phase_value mode exceeds vector size" );
        const_packed_positive_mode_view<complex_type> modes( vector, size );
        return scalar_phase_value( modes.mode( data.mode ) );
    }

    void translation_generator(
        const complex_type *state_on_slice, complex_type *generator, const std::size_t size
    ) const
    {
        translation_generator_1d( state_on_slice, generator, size );
    }

    std::vector<complex_type> translation_generator( const std::vector<complex_type> &state_on_slice ) const
    {
        std::vector<complex_type> generator( state_on_slice.size() );
        translation_generator( state_on_slice.data(), generator.data(), state_on_slice.size() );
        return generator;
    }

    projection_info project(
        const slice_data &data, const complex_type *generator, const complex_type *vector, complex_type *projected,
        const std::size_t size
    ) const
    {
        if ( generator == nullptr || vector == nullptr || projected == nullptr )
            throw std::invalid_argument( "fourier_slice_1d::project got null pointer" );

        projection_info info;
        if ( !data.active() )
        {
            for ( std::size_t i = 0; i < size; ++i )
                projected[i] = vector[i];
            return info;
        }

        info.phase_value = phase_value( data, vector, size );
        info.slice_matrix = data.slice_matrix;

        nmfd::operations::linalg::small_matrix<real_type, 1> S{ { data.slice_matrix } };
        nmfd::operations::linalg::small_vector<real_type, 1> rhs{ info.phase_value };
        const auto projection = solve_slice_projection( S, rhs, data.active_rank, singular_tolerance_, condition_tolerance_ );
        info.status = projection.status;
        if ( !info.ok() )
            return info;

        info.alpha = projection.alpha[0];
        for ( std::size_t i = 0; i < size; ++i )
        {
            projected[i] = traits_type::add(
                vector[i],
                scale_complex( generator[i], -info.alpha )
            );
        }
        return info;
    }

    std::vector<complex_type> project(
        const slice_data &data, const std::vector<complex_type> &generator, const std::vector<complex_type> &vector,
        projection_info &info
    ) const
    {
        if ( generator.size() != vector.size() )
            throw std::invalid_argument( "fourier_slice_1d::project vector sizes do not match" );
        std::vector<complex_type> projected( vector.size() );
        info = project( data, generator.data(), vector.data(), projected.data(), vector.size() );
        return projected;
    }

    real_type active_mode_tolerance() const
    {
        return active_mode_tolerance_;
    }

private:
    static real_type coefficient_abs( const complex_type &value )
    {
        return std::sqrt( traits_type::abs_sq( value ) );
    }

private:
    real_type active_mode_tolerance_;
    real_type singular_tolerance_;
    real_type condition_tolerance_;
};

} // namespace fourier
} // namespace symmetry

#endif
