#ifndef NONLINEAR_OPERATORS_CAHN_HILLIARD_MODEL_H
#define NONLINEAR_OPERATORS_CAHN_HILLIARD_MODEL_H

#include <memory>
#include <utility>
#include <discretization/finite_difference/periodic_laplacian.h>

namespace nonlinear_operators::cahn_hilliard
{
// Constant-mobility, unforced double-well CH on [0,2*pi]^2. Model data have
// no stage buffers or solver dependencies and are shared through const handles.
template <class Scalar>
struct model
{
    using scalar_type = Scalar;
    using grid_type   = discretization::finite_difference::periodic_laplacian_2d<Scalar>;
    grid_type grid;
    Scalar    mobility, epsilon_squared;
    explicit model( std::size_t n, Scalar m = Scalar( .5 ), Scalar epsilon = Scalar( .2 ) )
        : grid( n, n, 2 * std::acos( Scalar( -1 ) ), 2 * std::acos( Scalar( -1 ) ) ), mobility( m ),
          epsilon_squared( epsilon * epsilon )
    {
        if ( !std::isfinite( m ) || m <= 0 || !std::isfinite( epsilon ) || epsilon <= 0 ||
             !std::isfinite( epsilon_squared ) || epsilon_squared <= 0 )
            throw std::invalid_argument( "CH requires positive finite mobility and interface width" );
    }
    __DEVICE_TAG__ Scalar convex_potential( Scalar c, Scalar laplacian ) const
    {
        return c * c * c - epsilon_squared * laplacian;
    }
    __DEVICE_TAG__ Scalar physical_potential( Scalar c, Scalar laplacian ) const
    {
        return convex_potential( c, laplacian ) - c;
    }
    __DEVICE_TAG__ Scalar energy_density( Scalar c, Scalar laplacian ) const
    {
        const Scalar difference = c * c - 1;
        return difference * difference / 4 - epsilon_squared * c * laplacian / 2;
    }
    Scalar cell_area() const
    {
        return grid.hx * grid.hy;
    }
};

namespace detail
{
template <class Space>
std::shared_ptr<Space>
checked_space( std::shared_ptr<Space> ops, const std::shared_ptr<const model<typename Space::scalar_type>> &physics )
{
    if ( !ops || !physics || ops->size() != physics->grid.size() )
        throw std::invalid_argument( "CH concentration space/model mismatch" );
    return ops;
}
}
}
#endif
