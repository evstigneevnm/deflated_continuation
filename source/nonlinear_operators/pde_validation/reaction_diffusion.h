#ifndef PDE_VALIDATION_REACTION_DIFFUSION_H
#define PDE_VALIDATION_REACTION_DIFFUSION_H

#include "diffusion.h"

namespace nonlinear_operators::pde_validation
{
// Forced two-field Brusselator: u_t=Du*Delta_h(u)+1-3u+u^2*v+s_u,
// v_t=Dv*Delta_h(v)+2u-u^2*v+s_v. The nonconstant exact fields below
// define s independently through discrete modal eigenvalues. IMEX puts only
// diffusion in the implicit partition; both reactions and sources are explicit.
template <class Scalar>
struct reaction_diffusion
{
    static constexpr int                                             components = 2;
    discretization::finite_difference::periodic_laplacian_2d<Scalar> grid;
    Scalar                                                           du = .03, dv = .05, lambda1, lambda2;
    explicit reaction_diffusion( std::size_t n )
        : grid( n, n, 2 * std::acos( Scalar( -1 ) ), 2 * std::acos( Scalar( -1 ) ) ),
          lambda1( grid.eigenvalue( 1, 1 ) ), lambda2( grid.eigenvalue( 2, 1 ) )
    {
    }
    __DEVICE_TAG__ void exact( std::size_t i, Scalar t, Scalar *out ) const
    {
        const Scalar x = ( i / grid.ny ) * grid.hx, y = ( i % grid.ny ) * grid.hy;
        out[0] = 1 + .1 * std::exp( -t ) * std::cos( x ) * std::sin( y );
        out[1] = .8 + .07 * std::exp( -.5 * t ) * std::sin( 2 * x ) * std::cos( y );
    }
    template <partition Part, class Vector>
    __DEVICE_TAG__ void rate( const Vector &state, std::size_t i, Scalar t, Scalar *out ) const
    {
        out[0] = out[1] = 0;
        if constexpr ( Part != partition::explicit_part )
        {
            out[0] = du * grid.apply( state, i, 0, 2 );
            out[1] = dv * grid.apply( state, i, 1, 2 );
        }
        if constexpr ( Part != partition::implicit_part )
        {
            Scalar e[2];
            exact( i, t, e );
            const Scalar u = state( 2 * i ), v = state( 2 * i + 1 ), dx = -( e[0] - 1 ), dy = -.5 * ( e[1] - .8 );
            const Scalar r = 1 - 3 * e[0] + e[0] * e[0] * e[1], s = 2 * e[0] - e[0] * e[0] * e[1];
            out[0] += 1 - 3 * u + u * u * v + dx + du * lambda1 * ( e[0] - 1 ) - r;
            out[1] += 2 * u - u * u * v + dy + dv * lambda2 * ( e[1] - .8 ) - s;
        }
    }
    template <partition Part, class Vector>
    __DEVICE_TAG__ void jacobian( const Vector &state, const Vector &v, std::size_t i, Scalar, Scalar *out ) const
    {
        out[0] = out[1] = 0;
        if constexpr ( Part != partition::explicit_part )
        {
            out[0] = du * grid.apply( v, i, 0, 2 );
            out[1] = dv * grid.apply( v, i, 1, 2 );
        }
        if constexpr ( Part != partition::implicit_part )
        {
            const Scalar x = state( 2 * i ), y = state( 2 * i + 1 ), dx = v( 2 * i ), dy = v( 2 * i + 1 );
            out[0] += ( -3 + 2 * x * y ) * dx + x * x * dy;
            out[1] += ( 2 - 2 * x * y ) * dx - x * x * dy;
        }
    }
    template <partition Part>
    __DEVICE_TAG__ void time_derivative( std::size_t i, Scalar t, Scalar *out ) const
    {
        out[0] = out[1] = 0;
        if constexpr ( Part != partition::implicit_part )
        {
            Scalar e[2];
            exact( i, t, e );
            const Scalar dx = -( e[0] - 1 ), dy = -.5 * ( e[1] - .8 );
            out[0] = ( e[0] - 1 ) + du * lambda1 * dx - ( -3 + 2 * e[0] * e[1] ) * dx - e[0] * e[0] * dy;
            out[1] = .25 * ( e[1] - .8 ) + dv * lambda2 * dy - ( 2 - 2 * e[0] * e[1] ) * dx + e[0] * e[0] * dy;
        }
    }
};
}
#endif
