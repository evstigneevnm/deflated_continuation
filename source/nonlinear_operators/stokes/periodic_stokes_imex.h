#ifndef PERIODIC_STOKES_IMEX_H
#define PERIODIC_STOKES_IMEX_H

#include <array>
#include <utility>
#include <nonlinear_operators/stokes/periodic_stokes.h>
#include <nmfd/time_steppers/runge_kutta/imex_operator.h>

namespace nonlinear_operators::stokes
{
template <class T>
struct zero_spectral_forcing
{
    __DEVICE_TAG__ void operator()( T, T, T, T *out ) const
    {
        for ( int j = 0; j < 4; ++j )
            out[j] = 0;
    }
};

// Explicit Oseen transport plus forcing on the same Fourier velocity layout.
// Pressure-gradient removal is provided by the timestepper's projector, not
// hidden inside either partition. Zero transport gives the Stokes split.
template <class Space, class Forcing>
class periodic_transport_forcing
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using modes_type  = typename periodic_projector<Space>::modes_type;
    periodic_transport_forcing(
        std::shared_ptr<Space> operations, std::shared_ptr<const modes_type> modes,
        std::array<scalar_type, 2> transport, Forcing forcing
    )
        : operations_( std::move( operations ) ), modes_( std::move( modes ) ), transport_( transport ),
          forcing_( std::move( forcing ) )
    {
        periodic_projector<Space> validate( operations_, modes_ );
        if ( !std::isfinite( transport[0] ) || !std::isfinite( transport[1] ) )
            throw std::invalid_argument( "Nonfinite periodic transport velocity" );
    }
    void set_time( scalar_type time )
    {
        time_ = time;
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        typename Space::for_each_type each;
        each(
            mapping{
                state, rate, modes_->kx().data(), modes_->ky().data(), transport_[0], transport_[1], time_, forcing_
            },
            modes_->index_space().complex_size()
        );
        each.wait();
        return true;
    }
    // Accessible to CUDA's kernel template instantiation.
    struct mapping
    {
        vector_type         state, out;
        const scalar_type  *kx, *ky;
        scalar_type         cx, cy, time;
        Forcing             forcing;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t mode ) const
        {
            scalar_type force[4];
            forcing( kx[mode], ky[mode], time, force );
            const scalar_type frequency = cx * kx[mode] + cy * ky[mode];
            for ( int component = 0; component < 2; ++component )
            {
                const auto        i    = 4 * mode + 2 * component;
                const scalar_type real = state( i ), imag = state( i + 1 );
                out( i )     = frequency * imag + force[2 * component];
                out( i + 1 ) = -frequency * real + force[2 * component + 1];
            }
        }
    };

private:
    std::shared_ptr<Space>            operations_;
    std::shared_ptr<const modes_type> modes_;
    std::array<scalar_type, 2>        transport_;
    scalar_type                       time_ = 0;
    Forcing                           forcing_;
};

// Application-owned assembly only: shared geometry, independent partition
// contexts, no nonlinear/linear solver construction and no extra field buffers.
template <class Space, class Forcing>
class periodic_stokes_imex
{
public:
    using scalar_type        = typename Space::scalar_type;
    using modes_type         = typename periodic_projector<Space>::modes_type;
    using explicit_type      = periodic_transport_forcing<Space, Forcing>;
    using implicit_type      = periodic_stokes<Space, zero_spectral_forcing<scalar_type>>;
    using imex_operator_type = nmfd::time_steppers::runge_kutta::imex_operator<explicit_type, implicit_type>;
    periodic_stokes_imex(
        std::shared_ptr<Space> operations, std::shared_ptr<const modes_type> modes, scalar_type viscosity,
        Forcing forcing = {}, std::array<scalar_type, 2> transport = { 0, 0 }
    )
        : bundle_(
              std::make_shared<imex_operator_type>(
                  std::make_shared<explicit_type>( operations, modes, transport, std::move( forcing ) ),
                  std::make_shared<implicit_type>( operations, modes, viscosity )
              )
          )
    {
    }
    std::shared_ptr<imex_operator_type> get_imex_operator() const
    {
        return bundle_;
    }

private:
    std::shared_ptr<imex_operator_type> bundle_;
};
}
#endif
