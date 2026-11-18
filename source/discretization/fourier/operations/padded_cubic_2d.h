#ifndef DISCRETIZATION_FOURIER_PADDED_CUBIC_2D_H
#define DISCRETIZATION_FOURIER_PADDED_CUBIC_2D_H

#include <discretization/fourier/spectral_field.h>
#include <external_libraries/fft_facade.h>
#include <nmfd/operations/backend/complex.h>
#include <limits>
#include <scfd/utils/device_tag.h>

namespace discretization::fourier::operations
{
// Real nodal coordinates on [0,2*pi)^2, retaining |kx|,|ky| <= floor((n-1)/2).
// Keep fixed FFT input/output addresses (FFTW caches plans by address).
template <class Space, class FFTBackend>
class padded_cubic_2d
{
public:
    using scalar   = typename Space::scalar_type;
    using vector   = typename Space::vector_type;
    using backend  = typename Space::backend_type;
    using fft      = external_libraries::fft::r2c<FFTBackend, scalar>;
    using complex  = typename fft::complex_type;
    using traits   = nmfd::operations::backend::complex_value_traits<complex>;
    using field    = physical_field<backend, scalar, 2>;
    using spectrum = spectral_field<backend, complex, 2>;
    using extent   = typename field::extent_type;

    explicit padded_cubic_2d( int n )
        : n_( checked_size( n ) ), m_( 2 * n ), base_( n, n ), padded_( m_, m_ ), x_( extent( n, n ) ),
          y_( extent( m_, m_ ) ), z_( extent( n, n / 2 + 1 ) ), p_( extent( m_, m_ / 2 + 1 ) )
    {
    }

    extent padded_extent() const
    {
        return y_.extent();
    }

    void prepare_derivative( const vector &state, field &derivative )
    {
        if ( derivative.extent() != padded_extent() )
            throw std::invalid_argument( "Fourier derivative extent mismatch" );
        lift( state );
        auto y = y_.data(), d = derivative.data();
        each( y_.size(), [=] __DEVICE_TAG__( int i ) { d[i] = 3 * y[i] * y[i]; } );
    }

    // Out = L(k) u_hat + Q(k) P_K(cubic)_hat in physical coordinates.
    // A prepared multiplier replaces u^3 by 3*c_base^2*u on the padded grid.
    template <bool Nonlinear, class Symbol>
    void
    apply( const vector &u, vector &out, Symbol symbol, const field *derivative = nullptr, scalar excluded_gain = 0 )
    {
        if constexpr ( Nonlinear )
        {
            if ( derivative && derivative->extent() != padded_extent() )
                throw std::invalid_argument( "Fourier derivative extent mismatch" );
            lift( u );
            auto          y = y_.data();
            const scalar *d = derivative ? derivative->data() : nullptr;
            each( y_.size(), [=] __DEVICE_TAG__( int i ) { y[i] *= d ? d[i] : y[i] * y[i]; } );
            padded_.forward( y, p_.data() );
        }
        else
            forward( u );

        const int n = n_, m = m_, cutoff = ( n - 1 ) / 2;
        auto      z = z_.data(), p = p_.data();
        each( z_.size(), [=] __DEVICE_TAG__( int i ) {
            int k = i / ( n / 2 + 1 ), l = i % ( n / 2 + 1 );
            if ( k > n / 2 )
                k -= n;
            if ( k > cutoff || k < -cutoff || l > cutoff )
            {
                // Project the PDE, but allow a shifted inverse to act on the full nodal space.
                const scalar s = excluded_gain / ( scalar( n ) * n );
                z[i]           = traits::make( s * traits::real( z[i] ), s * traits::imag( z[i] ) );
                return;
            }
            scalar linear, nonlinear;
            symbol( k * k + l * l, linear, nonlinear );
            linear /= scalar( n ) * n;
            scalar re = linear * traits::real( z[i] ), im = linear * traits::imag( z[i] );
            if ( Nonlinear )
            {
                auto product = p[( k < 0 ? k + m : k ) * ( m / 2 + 1 ) + l];
                nonlinear /= scalar( m ) * m;
                re += nonlinear * traits::real( product );
                im += nonlinear * traits::imag( product );
            }
            z[i] = traits::make( re, im );
        } );
        base_.inverse( z, x_.data() );
        auto x = x_.data();
        each( x_.size(), [=] __DEVICE_TAG__( int i ) { out( i ) = x[i]; } );
    }

    // CUDA requires publicly nameable functor types for kernel launches.
    struct copy_input
    {
        vector              state;
        scalar             *x;
        __DEVICE_TAG__ void operator()( int i ) const
        {
            x[i] = state( i );
        }
    };
    struct embed
    {
        const complex      *z;
        complex            *p;
        int                 n, m;
        __DEVICE_TAG__ void operator()( int i ) const
        {
            int k = i / ( m / 2 + 1 ), l = i % ( m / 2 + 1 ), cutoff = ( n - 1 ) / 2;
            if ( k > m / 2 )
                k -= m;
            if ( k >= -cutoff && k <= cutoff && l <= cutoff )
            {
                auto         v = z[( k < 0 ? k + n : k ) * ( n / 2 + 1 ) + l];
                const scalar s = 1 / ( scalar( n ) * n );
                p[i]           = traits::make( s * traits::real( v ), s * traits::imag( v ) );
            }
            else
                p[i] = traits::make( 0, 0 );
        }
    };

private:
    static int checked_size( int n )
    {
        if ( n < 1 || n > std::numeric_limits<int>::max() / 4 / n )
            throw std::invalid_argument( "Invalid padded Fourier grid size" );
        return n;
    }
    template <class F>
    static void each( std::size_t count, F f )
    {
        typename Space::for_each_type run;
        run( f, count );
        run.wait();
    }
    void forward( const vector &state )
    {
        each( x_.size(), copy_input{ state, x_.data() } );
        base_.forward( x_.data(), z_.data() );
    }
    void lift( const vector &state )
    {
        forward( state );
        each( p_.size(), embed{ z_.data(), p_.data(), n_, m_ } );
        padded_.inverse( p_.data(), y_.data() );
    }
    int      n_, m_;
    fft      base_, padded_;
    field    x_, y_;
    spectrum z_, p_;
};
}
#endif
