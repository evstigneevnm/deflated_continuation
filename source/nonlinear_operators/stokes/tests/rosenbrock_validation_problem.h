#ifndef STOKES_TESTS_ROSENBROCK_VALIDATION_PROBLEM_H
#define STOKES_TESTS_ROSENBROCK_VALIDATION_PROBLEM_H
#include "manufactured_flow.h"
#include "implicit_rosenbrock_projection_problem.h"

namespace stokes_validation
{
// Benchmark-owned W approximation B=factor*J. Keep the physical residual,
// rate and analytic partial time derivative unchanged. Only the prepared
// shifted operator is copied (shared geometry and a few scalars, no fields).
template <class Space>
class rosenbrock_problem : public stokes::periodic_stokes<Space, manufactured_flow>
{
    using base = stokes::periodic_stokes<Space, manufactured_flow>;

public:
    using typename base::jacobi_operator_type;
    using typename base::vector_type;
    rosenbrock_problem(
        std::shared_ptr<Space> ops, std::shared_ptr<modes_type<Space>> modes, manufactured_flow flow, double factor = 1
    )
        : base( ops, modes, .15, flow, std::array<double, 2>{ flow.cx, flow.cy } ), factor_( factor ),
          jacobian_( std::make_shared<jacobi_operator_type>() )
    {
        if ( !std::isfinite( factor ) || factor < 0 )
            throw std::invalid_argument( "Invalid W Jacobian factor" );
    }
    void set_linearization_point( const vector_type &state )
    {
        base::set_linearization_point( state );
        *jacobian_ = *base::get_jacobi_operator();
        jacobian_->b *= factor_;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian_;
    }

private:
    double                                factor_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
};
}
#endif
