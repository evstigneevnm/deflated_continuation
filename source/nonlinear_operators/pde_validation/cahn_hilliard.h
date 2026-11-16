#ifndef PDE_VALIDATION_CAHN_HILLIARD_H
#define PDE_VALIDATION_CAHN_HILLIARD_H

#include "diffusion.h"
#include <memory>
#include <nmfd/detail/vector_wrap.h>

namespace nonlinear_operators::pde_validation
{
// Linear mixed CH: c_t=m*Delta_h(mu), 0=mu-kappa*c+epsilon2*Delta_h(c).
// State and residual order are (c,mu) at each cell. M=diag(1,0); g_mu=I
// makes this index 1. Mean concentration is conserved. No rate provider or
// mass inverse is supplied; every implicit solve uses the complete system.
template <class Scalar>
struct cahn_hilliard_model
{
    static constexpr int                                             components = 2;
    discretization::finite_difference::periodic_laplacian_2d<Scalar> grid;
    Scalar mobility = .5, kappa = 1, epsilon2 = .01, lambda1, lambda2;
    explicit cahn_hilliard_model( std::size_t n )
        : grid( n, n, 2 * std::acos( Scalar( -1 ) ), 2 * std::acos( Scalar( -1 ) ) ),
          lambda1( grid.eigenvalue( 1, 1 ) ), lambda2( grid.eigenvalue( 2, 1 ) )
    {
    }
    __DEVICE_TAG__ void exact( std::size_t i, Scalar t, Scalar *out ) const
    {
        const Scalar x = ( i / grid.ny ) * grid.hx, y = ( i % grid.ny ) * grid.hy;
        const Scalar d1 = kappa + epsilon2 * lambda1, d2 = kappa + epsilon2 * lambda2;
        const Scalar u = .2 * std::exp( -mobility * lambda1 * d1 * t ) * std::cos( x ) * std::sin( y );
        const Scalar v = .1 * std::exp( -mobility * lambda2 * d2 * t ) * std::sin( 2 * x ) * std::cos( y );
        out[0]         = .3 + u + v;
        out[1]         = kappa * .3 + d1 * u + d2 * v;
    }
};
template <class Space>
class cahn_hilliard
{
public:
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    using model_type  = cahn_hilliard_model<scalar_type>;
    using wrap        = nmfd::detail::vector_wrap<Space>;
    struct jacobi_operator_type
    {
        std::shared_ptr<const model_type> model;
        scalar_type                       a = 0, b = 1;
        struct mapping
        {
            model_type          model;
            vector_type         in, out;
            scalar_type         a, b;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( 2 * i ) = a * in( 2 * i ) - b * model.mobility * model.grid.apply( in, i, 1, 2 );
                out( 2 * i + 1 ) =
                    in( 2 * i + 1 ) - model.kappa * in( 2 * i ) + model.epsilon2 * model.grid.apply( in, i, 0, 2 );
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Space::for_each_type each;
            each( mapping{ *model, in, out, a, b }, model->grid.size() );
            each.wait();
        }
    };
    cahn_hilliard( std::shared_ptr<Space> ops, std::shared_ptr<const model_type> model )
        : ops_( ops ? std::move( ops ) : throw std::invalid_argument( "Missing CH space" ) ),
          model_( std::move( model ) ), previous_( *ops_ ), rhs_( *ops_ ),
          jacobian_( std::make_shared<jacobi_operator_type>() )
    {
        if ( !model_ || ops_->size() != 2 * model_->grid.size() )
            throw std::invalid_argument( "CH space/model mismatch" );
        previous_.start_use();
        rhs_.start_use();
        jacobian_->model = model_;
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
    void set_time( scalar_type )
    {
    } // Autonomous; no hidden time-derivative interpretation.
    void set_affine_shift( scalar_type a, scalar_type b )
    {
        if ( !std::isfinite( a ) || !std::isfinite( b ) || b <= 0 )
            throw std::invalid_argument( "Invalid CH affine shift" );
        a_ = a;
        b_ = b;
    }
    struct mapping
    {
        model_type          model;
        vector_type         state, previous, rhs, out;
        scalar_type         a, b;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( 2 * i ) = a * ( state( 2 * i ) - previous( 2 * i ) ) -
                           b * ( model.mobility * model.grid.apply( state, i, 1, 2 ) + rhs( 2 * i ) );
            out( 2 * i + 1 ) =
                state( 2 * i + 1 ) - model.kappa * state( 2 * i ) + model.epsilon2 * model.grid.apply( state, i, 0, 2 );
        }
    };
    void apply( const vector_type &state, vector_type &out ) const
    {
        typename Space::for_each_type each;
        each( mapping{ *model_, state, *previous_, *rhs_, out, a_, b_ }, model_->grid.size() );
        each.wait();
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

private:
    std::shared_ptr<Space>                ops_;
    std::shared_ptr<const model_type>     model_;
    wrap                                  previous_, rhs_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    scalar_type                           a_ = 0, b_ = 1;
};
}
#endif
