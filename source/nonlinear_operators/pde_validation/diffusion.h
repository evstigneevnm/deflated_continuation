#ifndef PDE_VALIDATION_DIFFUSION_H
#define PDE_VALIDATION_DIFFUSION_H

#include <discretization/finite_difference/periodic_laplacian.h>

namespace nonlinear_operators::pde_validation
{
enum class partition
{
    full,
    explicit_part,
    implicit_part
};

// u_t=nu*Delta_h(u)+s(t) on [0,2*pi]^2. The two exact spatial harmonics
// use discrete eigenvalues, so temporal tests contain no spatial truncation
// error. The source and its partial derivative preserve the mean 0.3.
template <class Scalar>
struct diffusion
{
    static constexpr int                                             components = 1;
    discretization::finite_difference::periodic_laplacian_2d<Scalar> grid;
    Scalar                                                           nu = .07, lambda1, lambda2;
    explicit diffusion( std::size_t n )
        : grid( n, n, 2 * std::acos( Scalar( -1 ) ), 2 * std::acos( Scalar( -1 ) ) ),
          lambda1( grid.eigenvalue( 1, 2 ) ), lambda2( grid.eigenvalue( 3, 1 ) )
    {
    }
    __DEVICE_TAG__ void harmonics( std::size_t i, Scalar t, Scalar &x, Scalar &y ) const
    {
        const Scalar xx = ( i / grid.ny ) * grid.hx, yy = ( i % grid.ny ) * grid.hy;
        x = std::exp( -t ) * std::cos( xx ) * std::sin( 2 * yy );
        y = .2 * std::exp( -2 * t ) * std::sin( 3 * xx ) * std::cos( yy );
    }
    __DEVICE_TAG__ void exact( std::size_t i, Scalar t, Scalar *out ) const
    {
        Scalar x, y;
        harmonics( i, t, x, y );
        out[0] = .3 + x + y;
    }
    template <partition Part, class Vector>
    __DEVICE_TAG__ void rate( const Vector &u, std::size_t i, Scalar t, Scalar *out ) const
    {
        Scalar x, y;
        harmonics( i, t, x, y );
        out[0] = 0;
        if constexpr ( Part != partition::explicit_part )
            out[0] += nu * grid.apply( u, i, 0, 1 );
        if constexpr ( Part != partition::implicit_part )
            out[0] += ( -1 + nu * lambda1 ) * x + ( -2 + nu * lambda2 ) * y;
    }
    template <partition Part, class Vector>
    __DEVICE_TAG__ void jacobian( const Vector &, const Vector &v, std::size_t i, Scalar, Scalar *out ) const
    {
        out[0] = 0;
        if constexpr ( Part != partition::explicit_part )
            out[0] = nu * grid.apply( v, i, 0, 1 );
    }
    template <partition Part>
    __DEVICE_TAG__ void time_derivative( std::size_t i, Scalar t, Scalar *out ) const
    {
        Scalar x, y;
        harmonics( i, t, x, y );
        out[0] = 0;
        if constexpr ( Part != partition::implicit_part )
            out[0] = ( 1 - nu * lambda1 ) * x + ( 4 - 2 * nu * lambda2 ) * y;
    }
};
}
#endif
