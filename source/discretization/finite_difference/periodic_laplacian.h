#ifndef DISCRETIZATION_FINITE_DIFFERENCE_PERIODIC_LAPLACIAN_H
#define DISCRETIZATION_FINITE_DIFFERENCE_PERIODIC_LAPLACIAN_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <scfd/utils/device_tag.h>
#include <discretization/common/structured_extent.h>

namespace discretization::finite_difference
{
// Five-point periodic Laplacian on cell-aligned, interleaved fields. The
// descriptor contains only scalars; stencil evaluation needs distinct in/out.
template <class Scalar>
struct periodic_laplacian_2d
{
    std::size_t   nx, ny;
    Scalar        hx, hy, ix2, iy2;
    static Scalar spacing( std::size_t n, Scalar length )
    {
        if ( n < 3 || !std::isfinite( length ) || length <= 0 )
            throw std::invalid_argument( "Periodic Laplacian requires a positive domain and at least 3x3 cells" );
        return length / n;
    }
    periodic_laplacian_2d( std::size_t x, std::size_t y, Scalar lx, Scalar ly )
        : nx( x ), ny( y ), hx( spacing( x, lx ) ), hy( spacing( y, ly ) ), ix2( 1 / ( hx * hx ) ),
          iy2( 1 / ( hy * hy ) )
    {
        common::structured_extent<2>( nx, ny ).size();
        if ( !std::isfinite( ix2 ) || !std::isfinite( iy2 ) || ix2 <= 0 || iy2 <= 0 )
            throw std::invalid_argument( "Unrepresentable periodic grid spacing" );
    }
    std::size_t size() const
    {
        return nx * ny;
    }
    template <class Vector>
    __DEVICE_TAG__ Scalar apply( const Vector &u, std::size_t i, int c, int components ) const
    {
        const auto   x = i / ny, y = i % ny;
        const auto   xm     = ( x ? x - 1 : nx - 1 ) * ny + y;
        const auto   xp     = ( x + 1 == nx ? 0 : x + 1 ) * ny + y;
        const auto   ym     = x * ny + ( y ? y - 1 : ny - 1 );
        const auto   yp     = x * ny + ( y + 1 == ny ? 0 : y + 1 );
        const Scalar center = u( components * i + c );
        return ix2 * ( u( components * xm + c ) - 2 * center + u( components * xp + c ) ) +
               iy2 * ( u( components * ym + c ) - 2 * center + u( components * yp + c ) );
    }
    // Positive eigenvalue of -Delta_h; used by independent manufactured forcing.
    Scalar eigenvalue( int kx, int ky ) const
    {
        const Scalar x = std::sin( Scalar( kx ) * std::acos( Scalar( -1 ) ) / nx );
        const Scalar y = std::sin( Scalar( ky ) * std::acos( Scalar( -1 ) ) / ny );
        return 4 * ( ix2 * x * x + iy2 * y * y );
    }
};
}
#endif
