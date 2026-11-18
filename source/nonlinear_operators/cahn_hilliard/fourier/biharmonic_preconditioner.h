#ifndef NONLINEAR_OPERATORS_CH_FOURIER_BIHARMONIC_PRECONDITIONER_H
#define NONLINEAR_OPERATORS_CH_FOURIER_BIHARMONIC_PRECONDITIONER_H

#include "problem.h"

namespace nonlinear_operators::cahn_hilliard::fourier
{
// Right inverse of the principal part a*I + b*m*eps^2*Delta^2 of a forward
// CH stage. GMRES still applies the full frozen Jacobian to verify its residual.
template <class Problem>
class biharmonic_preconditioner
{
    using jacobian = typename Problem::jacobi_operator_type;
    using scalar   = typename Problem::scalar_type;
    using vector   = typename Problem::vector_type;
    std::shared_ptr<const jacobian> op_;

public:
    struct inverse_symbol
    {
        scalar              a, biharmonic;
        __DEVICE_TAG__ void operator()( int k2, scalar &linear, scalar &nonlinear ) const
        {
            linear    = 1 / ( a + biharmonic * k2 * k2 );
            nonlinear = 0;
        }
    };
    void set_operator( std::shared_ptr<const jacobian> op )
    {
        op_ = std::move( op );
    }
    bool apply( vector &x ) const
    {
        if ( !op_ || !( op_->a > 0 ) || !( op_->b > 0 ) )
            return false;
        inverse_symbol symbol{ op_->a, op_->b * op_->symbol.mobility * op_->symbol.eps2 };
        // The rate is projected out on excluded Nyquist lines: J there is a*I.
        op_->spatial->template apply<false>( x, x, symbol, nullptr, 1 / op_->a );
        return true;
    }
};
}
#endif
