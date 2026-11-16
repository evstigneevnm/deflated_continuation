#ifndef STOKES_VALIDATION_MANUFACTURED_FLOW_H
#define STOKES_VALIDATION_MANUFACTURED_FLOW_H

#include <cmath>
#include <memory>
#include <nonlinear_operators/stokes/periodic_stokes.h>

namespace stokes_validation
{
namespace stokes = nonlinear_operators::stokes;
// Periodic Stokes (c=0) or Oseen (constant c):
// u_t+c.grad(u)=nu*Delta(u)-grad(p)+f, div(u)=0, nu=.15.
// With X=x-cx*t, Y=y-cy*t, choose
// psi=exp(-t)*cos(X)*cos(Y)+.12*exp(-.7*t)*cos(3*X)*cos(2*Y),
// u=(d_y psi,-d_x psi)+(.125,-.25), p=(1+.2*sin(t))*cos(X)*cos(Y).
// Then f=u_t+c.grad(u)-nu*Delta(u)+grad(p). Two spatial eigenvalues,
// moving Fourier phases and time-dependent gradient forcing exercise both
// partitions. Reference correction removes pressure analytically, not by P.
struct manufactured_flow
{
    double              cx = 0, cy = 0;
    bool                pressure = true;
    __DEVICE_TAG__ void velocity( double x, double y, double t, double *out ) const
    {
        double amplitude = 0;
        if ( ( x == 1 || x == -1 ) && y == 1 )
            amplitude = .25 * exp( -t );
        if ( ( x == 3 || x == -3 ) && y == 2 )
            amplitude = .03 * exp( -.7 * t );
        const double phase = ( cx * x + cy * y ) * t;
        out[0]             = y * amplitude * sin( phase );
        out[1]             = y * amplitude * cos( phase );
        out[2]             = -x * amplitude * sin( phase );
        out[3]             = -x * amplitude * cos( phase );
        if ( x == 0 && y == 0 )
        {
            out[0] = .125;
            out[2] = -.25;
        }
    }
    __DEVICE_TAG__ void operator()( double x, double y, double t, double *out ) const
    {
        velocity( x, y, t, out );
        const double decay = y == 2 ? .7 : 1, multiplier = .15 * ( x * x + y * y ) - decay;
        for ( int j = 0; j < 4; ++j )
            out[j] *= multiplier;
        if ( x == 0 && y == 0 )
            out[0] = out[2] = 0;
        if ( pressure && ( x == 1 || x == -1 ) && y == 1 )
        {
            const double phase = ( cx * x + cy * y ) * t, p = .25 * ( 1 + .2 * sin( t ) );
            out[0] += x * p * sin( phase );
            out[1] += x * p * cos( phase );
            out[2] += y * p * sin( phase );
            out[3] += y * p * cos( phase );
        }
    }
};
template <class Space>
using modes_type = typename stokes::periodic_projector<Space>::modes_type;
template <class Space>
std::shared_ptr<modes_type<Space>> make_modes( std::size_t n )
{
    using namespace discretization::fourier;
    discretization::common::structured_extent<2> extent( n, n );
    return std::make_shared<modes_type<Space>>(
        periodic_grid<double, 2>( extent, { 2 * std::acos( -1. ), 2 * std::acos( -1. ) } ), r2c_index_space_2d( extent )
    );
}
template <class Space>
struct flow_mapping
{
    typename Space::vector_type out;
    const double               *kx, *ky;
    double                      time;
    manufactured_flow           flow;
    __DEVICE_TAG__ void         operator()( std::ptrdiff_t mode ) const
    {
        double value[4];
        flow.velocity( kx[mode], ky[mode], time, value );
        for ( int j = 0; j < 4; ++j )
            out( 4 * mode + j ) = value[j];
    }
};
template <class Space>
void exact_state(
    const modes_type<Space> &modes, typename Space::vector_type &out, double time, manufactured_flow flow
)
{
    typename Space::for_each_type each;
    each(
        flow_mapping<Space>{ out, modes.kx().data(), modes.ky().data(), time, flow }, modes.index_space().complex_size()
    );
    each.wait();
}

}
#endif
