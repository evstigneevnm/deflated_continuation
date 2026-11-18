#ifndef NONLINEAR_OPERATORS_CH_FOURIER_PROBLEM_H
#define NONLINEAR_OPERATORS_CH_FOURIER_PROBLEM_H

#include "../model.h"
#include <discretization/fourier/operations/padded_cubic_2d.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/runge_kutta/imex_operator.h>

namespace nonlinear_operators::cahn_hilliard::fourier
{
// F_hat=-m*k^2*(P(c^3)_hat+(epsilon^2*k^2-s)*c_hat), s=0 (convex) or 1 (full).
// The same symbols act on the padded cubic derivative in the shifted Jacobian.
template <class T>
struct rate_symbol
{
    T                   mobility, eps2, s;
    __DEVICE_TAG__ void operator()( int k2, T &linear, T &nonlinear ) const
    {
        nonlinear = -mobility * k2;
        linear    = nonlinear * ( eps2 * k2 - s );
    }
};
template <class T>
struct laplace_symbol
{
    T                   scale;
    __DEVICE_TAG__ void operator()( int k2, T &linear, T &nonlinear ) const
    {
        linear    = -scale * k2;
        nonlinear = 0;
    }
};
template <class S, class FFT>
using workspace = discretization::fourier::operations::padded_cubic_2d<S, FFT>;

template <class S, class FFT, bool Convex>
class problem
{
public:
    using scalar_type = typename S::scalar_type;
    using vector_type = typename S::vector_type;
    using physics     = model<scalar_type>;
    using wrap        = nmfd::detail::vector_wrap<S>;
    using work        = workspace<S, FFT>;
    struct jacobi_operator_type
    {
        std::shared_ptr<S>       ops;
        std::shared_ptr<work>    spatial;
        typename work::field     derivative;
        rate_symbol<scalar_type> symbol;
        scalar_type              a = 0, b = 1;
        jacobi_operator_type( std::shared_ptr<S> o, std::shared_ptr<work> w, rate_symbol<scalar_type> f )
            : ops( std::move( o ) ), spatial( std::move( w ) ), derivative( spatial->padded_extent() ), symbol( f )
        {
        }
        void apply( const vector_type &v, vector_type &out ) const
        {
            spatial->template apply<true>( v, out, symbol, &derivative );
            ops->add_mul( a, v, -b, out );
        }
    };
    problem( std::shared_ptr<S> ops, std::shared_ptr<const physics> m )
        : ops_( detail::checked_space( std::move( ops ), m ) ), previous_( *ops_ ), rhs_( *ops_ ),
          spatial_( std::make_shared<work>( m->grid.nx ) ),
          symbol_{ m->mobility, m->epsilon_squared, scalar_type( !Convex ) },
          jac_( std::make_shared<jacobi_operator_type>( ops_, spatial_, symbol_ ) )
    {
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
    void set_time( scalar_type )
    {
    }
    void set_affine_shift( scalar_type a, scalar_type b )
    {
        if ( !std::isfinite( a ) || !std::isfinite( b ) || b <= 0 )
            throw std::invalid_argument( "Invalid Fourier CH shift" );
        a_ = a;
        b_ = b;
    }
    bool evaluate_rate( const vector_type &x, vector_type &out ) const
    {
        spatial_->template apply<true>( x, out, symbol_ );
        return true;
    }
    void apply( const vector_type &x, vector_type &out ) const
    {
        evaluate_rate( x, out );
        auto                      w = *previous_, r = *rhs_;
        auto                      a = a_, b = b_;
        typename S::for_each_type each;
        each(
            [=] __DEVICE_TAG__( int i ) { out( i ) = a * ( x( i ) - w( i ) ) - b * ( out( i ) + r( i ) ); },
            ops_->size()
        );
        each.wait();
    }
    void set_linearization_point( const vector_type &x )
    {
        spatial_->prepare_derivative( x, jac_->derivative );
        jac_->a = a_;
        jac_->b = b_;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jac_;
    }

private:
    std::shared_ptr<S>                    ops_;
    wrap                                  previous_, rhs_;
    std::shared_ptr<work>                 spatial_;
    rate_symbol<scalar_type>              symbol_;
    std::shared_ptr<jacobi_operator_type> jac_;
    scalar_type                           a_ = 0, b_ = 1;
};

template <class S, class FFT>
class split_problem
{
public:
    using scalar_type            = typename S::scalar_type;
    using vector_type            = typename S::vector_type;
    using physics                = model<scalar_type>;
    using implicit_operator_type = problem<S, FFT, true>;
    struct explicit_operator_type
    {
        explicit_operator_type( int n, scalar_type m ) : spatial( n ), symbol{ -m }
        {
        }
        workspace<S, FFT>           spatial;
        laplace_symbol<scalar_type> symbol;
        void                        set_time( scalar_type )
        {
        }
        bool evaluate_rate( const vector_type &x, vector_type &out )
        {
            spatial.template apply<false>( x, out, symbol );
            return true;
        }
    };
    using imex_operator_type =
        nmfd::time_steppers::runge_kutta::imex_operator<explicit_operator_type, implicit_operator_type>;
    split_problem( std::shared_ptr<S> ops, std::shared_ptr<const physics> m )
    {
        auto implicit = std::make_shared<implicit_operator_type>( std::move( ops ), m );
        bundle_       = std::make_shared<imex_operator_type>(
            std::make_shared<explicit_operator_type>( m->grid.nx, m->mobility ), implicit
        );
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
