#ifndef NONLINEAR_OPERATORS_CH_FOURIER_DIAGNOSTICS_H
#define NONLINEAR_OPERATORS_CH_FOURIER_DIAGNOSTICS_H

#include "problem.h"
#include "../energy_diagnostics.h"
#include "../physical_diagnostics.h"

namespace nonlinear_operators::cahn_hilliard::fourier
{
template <class S, class FFT>
class diagnostics
{
public:
    using T           = typename S::scalar_type;
    using vector      = typename S::vector_type;
    using physics     = model<T>;
    using report_type = typename physical_diagnostics<S>::report_type;
    using step_report = typename energy_diagnostics<S>::report_type;
    diagnostics( std::shared_ptr<S> ops, std::shared_ptr<const physics> m )
        : ops_( detail::checked_space( std::move( ops ), m ) ), m_( std::move( m ) ), w_( m_->grid.nx ), mu_( *ops_ ),
          lap_( *ops_ ), defect_( *ops_ )
    {
        mu_.start_use();
        lap_.start_use();
        defect_.start_use();
    }

    report_type evaluate( const vector &x )
    {
        // <c,P_K(c^3)> integrates c^4 exactly on the retained subspace.
        w_.template apply<true>( x, *mu_, [] __DEVICE_TAG__( int, T &l, T &q ) {
            l = 0;
            q = 1;
        } );
        w_.template apply<false>( x, *lap_, laplace_symbol<T>{ 1 } );
        const T area   = m_->cell_area();
        const T energy = area * ( ( ops_->size() + ops_->scalar_prod( x, *mu_ ) - 2 * ops_->scalar_prod( x, x ) ) / 4 -
                                  m_->epsilon_squared * ops_->scalar_prod( x, *lap_ ) / 2 );
        ops_->add_mul( -1, x, -m_->epsilon_squared, *lap_, 1, *mu_ );
        const T d    = dissipation();
        const T mass = area * ops_->sum( x );
        if ( !std::isfinite( energy ) || !std::isfinite( d ) || !std::isfinite( mass ) )
            throw std::overflow_error( "Nonfinite Fourier CH diagnostics" );
        return { energy, mass, d };
    }
    step_report evaluate_step( const vector &before, const vector &after, T h )
    {
        const auto old = evaluate( before ), now = evaluate( after );
        // mu_* = mu(after)+after-before for convex-split Euler only.
        ops_->add_mul( 1, after, -1, before, 1, *mu_ );
        const T d = dissipation();
        ops_->assign_mul( 1, after, -1, before, *defect_ );
        ops_->add_lin_comb( -h * m_->mobility, *lap_, *defect_ );
        const T work = m_->cell_area() * ops_->scalar_prod( *mu_, *defect_ );
        const T roundoff =
            256 * std::numeric_limits<T>::epsilon() *
            ( 1 + std::abs( old.energy ) + std::abs( now.energy ) + std::abs( h * d ) + std::abs( work ) );
        return {
            old.energy,
            now.energy,
            old.mass,
            now.mass,
            d,
            work,
            now.energy - old.energy + h * d,
            ops_->norm( *defect_ ) / std::sqrt( T( ops_->size() ) ),
            roundoff
        };
    }

private:
    T dissipation()
    {
        w_.template apply<false>( *mu_, *lap_, laplace_symbol<T>{ 1 } );
        return -m_->mobility * m_->cell_area() * ops_->scalar_prod( *mu_, *lap_ );
    }
    std::shared_ptr<S>             ops_;
    std::shared_ptr<const physics> m_;
    workspace<S, FFT>              w_;
    nmfd::detail::vector_wrap<S>   mu_, lap_, defect_;
};
}
#endif
