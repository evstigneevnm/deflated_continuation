#ifndef PERIODIC_STOKES_H
#define PERIODIC_STOKES_H

#include <cmath>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <discretization/fourier/wavevector_table.h>

namespace nonlinear_operators::stokes
{
// R2C Fourier velocity: (Re ux, Im ux, Re uy, Im uy) per stored mode.
// As in the project's translation-equivariant Fourier space, Nyquist rows and
// columns are excluded. Pressure has zero mean; velocity means are retained.
// No FFT is needed for linear Stokes.
template <class Space>
class periodic_projector
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using modes_type  = discretization::fourier::wavevector_table_2d<typename Space::backend_type, scalar_type>;
    periodic_projector( std::shared_ptr<Space> operations, std::shared_ptr<const modes_type> modes )
        : operations_( std::move( operations ) ), modes_( std::move( modes ) )
    {
        if ( !operations_ || !modes_ || operations_->size() != 4 * modes_->index_space().complex_size() )
            throw std::invalid_argument( "Periodic Stokes velocity/mode size mismatch" );
    }
    bool apply( vector_type &velocity ) const
    {
        execute(
            mapping{
                velocity, velocity, modes_->kx().data(), modes_->ky().data(), false, modes_->index_space().nx(),
                modes_->index_space().my()
            }
        );
        return true;
    }
    // Store complex divergence in component zero; clear component one.
    void divergence( const vector_type &velocity, vector_type &out ) const
    {
        execute(
            mapping{
                velocity, out, modes_->kx().data(), modes_->ky().data(), true, modes_->index_space().nx(),
                modes_->index_space().my()
            }
        );
    }

    // CUDA requires accessible kernel argument types.
    struct mapping
    {
        vector_type         in, out;
        const scalar_type  *kx, *ky;
        bool                divergence;
        std::size_t         nx, my;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t mode ) const
        {
            const auto i = 4 * mode;
            if ( std::size_t( mode ) / my == nx / 2 || std::size_t( mode ) % my == my - 1 )
            {
                for ( int j = 0; j < 4; ++j )
                    out( i + j ) = 0;
                return;
            }
            const scalar_type x = kx[mode], y = ky[mode], k2 = x * x + y * y;
            const scalar_type ur = in( i ), ui = in( i + 1 ), vr = in( i + 2 ), vi = in( i + 3 );
            const scalar_type dr = x * ur + y * vr, di = x * ui + y * vi;
            if ( divergence )
            {
                out( i )     = -di;
                out( i + 1 ) = dr;
                out( i + 2 ) = out( i + 3 ) = 0;
            }
            else if ( k2 != 0 )
            {
                // Solve Delta q=div(u) with q(0)=0, then u <- u-grad(q).
                out( i )     = ur - x * dr / k2;
                out( i + 1 ) = ui - x * di / k2;
                out( i + 2 ) = vr - y * dr / k2;
                out( i + 3 ) = vi - y * di / k2;
            }
            else
            {
                out( i )     = ur;
                out( i + 1 ) = ui;
                out( i + 2 ) = vr;
                out( i + 3 ) = vi;
            }
        }
    };

private:
    void execute( mapping op ) const
    {
        typename Space::for_each_type each;
        each( op, modes_->index_space().complex_size() );
        each.wait();
    }
    std::shared_ptr<Space>            operations_;
    std::shared_ptr<const modes_type> modes_;
};

// Provisional equation u'=nu*Delta(u)-c.grad(u)+forcing (c=0 by default).
// A timestepper-supplied Leray
// projector supplies incompressibility at every stage. Diffusion commutes with
// that fixed projector. Forcing is a backend functor returning four real Fourier
// coefficients at (kx,ky,time); it must preserve real-field Hermitian symmetry.
template <class Space, class Forcing>
class periodic_stokes
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using modes_type  = typename periodic_projector<Space>::modes_type;
    struct jacobi_operator_type
    {
        std::shared_ptr<const modes_type> modes;
        scalar_type                       a = 0, b = 1, viscosity = 1;
        std::array<scalar_type, 2>        transport{ 0, 0 };
        struct mapping
        {
            vector_type         in, out;
            const scalar_type  *k2;
            scalar_type         a, b, viscosity;
            const scalar_type  *kx, *ky;
            scalar_type         cx, cy;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t mode ) const
            {
                const scalar_type diagonal = a + b * viscosity * k2[mode];
                const scalar_type skew     = b * ( cx * kx[mode] + cy * ky[mode] );
                for ( int component = 0; component < 2; ++component )
                {
                    const auto        i    = 4 * mode + 2 * component;
                    const scalar_type real = in( i ), imag = in( i + 1 );
                    out( i )     = diagonal * real - skew * imag;
                    out( i + 1 ) = diagonal * imag + skew * real;
                }
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Space::for_each_type each;
            each(
                mapping{
                    in, out, modes->k_squared().data(), a, b, viscosity, modes->kx().data(), modes->ky().data(),
                    transport[0], transport[1]
                },
                modes->index_space().complex_size()
            );
            each.wait();
        }
    };
    periodic_stokes(
        std::shared_ptr<Space> operations, std::shared_ptr<const modes_type> modes, scalar_type viscosity,
        Forcing forcing = {}, std::array<scalar_type, 2> transport = { 0, 0 }
    )
        : operations_( operations ? std::move( operations ) : throw std::invalid_argument( "Missing Stokes space" ) ),
          modes_( std::move( modes ) ), previous_( *operations_ ), rhs_( *operations_ ),
          jacobian_( std::make_shared<jacobi_operator_type>() ), viscosity_( viscosity ),
          forcing_( std::move( forcing ) ), transport_( transport )
    {
        periodic_projector<Space> validate( operations_, modes_ );
        if ( !std::isfinite( viscosity ) || viscosity <= 0 )
            throw std::invalid_argument( "Stokes requires positive finite viscosity" );
        if ( !std::isfinite( transport[0] ) || !std::isfinite( transport[1] ) )
            throw std::invalid_argument( "Nonfinite periodic transport velocity" );
        jacobian_->modes     = modes_;
        jacobian_->viscosity = viscosity;
        jacobian_->transport = transport;
        previous_.start_use();
        rhs_.start_use();
    }
    vector_type &previous_state()
    {
        return *previous_;
    }
    const vector_type &previous_state() const
    {
        return *previous_;
    }
    vector_type &known_rhs()
    {
        return *rhs_;
    }
    const vector_type &known_rhs() const
    {
        return *rhs_;
    }
    void set_time( scalar_type time )
    {
        time_ = time;
    }
    void set_affine_shift( scalar_type a, scalar_type b )
    {
        if ( !std::isfinite( a ) || !std::isfinite( b ) || b <= 0 )
            throw std::invalid_argument( "Invalid Stokes affine shift" );
        a_ = a;
        b_ = b;
    }
    void apply( const vector_type &state, vector_type &residual ) const
    {
        evaluate( state, residual, false );
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        evaluate( state, rate, true );
        return true;
    }
    // Partial derivative at fixed state. Only available for forcing functors
    // supplying an analytic time derivative; old forcing contracts are unchanged.
    template <class F = Forcing>
    auto evaluate_time_derivative( const vector_type &, vector_type &out ) const
        -> decltype( std::declval<const F &>().time_derivative( scalar_type{}, scalar_type{}, scalar_type{}, std::declval<scalar_type *>() ), bool() )
    {
        typename Space::for_each_type each;
        each(
            time_mapping{ out, modes_->kx().data(), modes_->ky().data(), time_, forcing_ },
            modes_->index_space().complex_size()
        );
        each.wait();
        return true;
    }
    void set_linearization_point( const vector_type & )
    {
        jacobian_->a = a_;
        jacobian_->b = b_;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian_;
    }

    // CUDA requires accessible kernel argument types.
    struct time_mapping
    {
        vector_type         out;
        const scalar_type  *kx, *ky;
        scalar_type         time;
        Forcing             forcing;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t mode ) const
        {
            scalar_type values[4];
            forcing.time_derivative( kx[mode], ky[mode], time, values );
            for ( int j = 0; j < 4; ++j )
                out( 4 * mode + j ) = values[j];
        }
    };
    struct mapping
    {
        vector_type         state, previous, rhs, out;
        const scalar_type  *kx, *ky, *k2;
        scalar_type         time, a, b, viscosity;
        Forcing             forcing;
        bool                rate;
        scalar_type         cx, cy;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t mode ) const
        {
            scalar_type force[4];
            forcing( kx[mode], ky[mode], time, force );
            const scalar_type frequency = cx * kx[mode] + cy * ky[mode];
            for ( int component = 0; component < 2; ++component )
            {
                const auto        i    = 4 * mode + 2 * component;
                const scalar_type real = state( i ), imag = state( i + 1 );
                const scalar_type fr = -viscosity * k2[mode] * real + frequency * imag + force[2 * component];
                const scalar_type fi = -viscosity * k2[mode] * imag - frequency * real + force[2 * component + 1];
                out( i )             = rate ? fr : a * ( real - previous( i ) ) - b * ( fr + rhs( i ) );
                out( i + 1 )         = rate ? fi : a * ( imag - previous( i + 1 ) ) - b * ( fi + rhs( i + 1 ) );
            }
        }
    };

private:
    void evaluate( const vector_type &state, vector_type &out, bool rate ) const
    {
        typename Space::for_each_type each;
        each(
            mapping{
                state, *previous_, *rhs_, out, modes_->kx().data(), modes_->ky().data(), modes_->k_squared().data(),
                time_, a_, b_, viscosity_, forcing_, rate, transport_[0], transport_[1]
            },
            modes_->index_space().complex_size()
        );
        each.wait();
    }
    std::shared_ptr<Space>                operations_;
    std::shared_ptr<const modes_type>     modes_;
    nmfd::detail::vector_wrap<Space>      previous_, rhs_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    scalar_type                           time_ = 0, a_ = 0, b_ = 1, viscosity_;
    Forcing                               forcing_;
    std::array<scalar_type, 2>            transport_;
};
}
#endif
