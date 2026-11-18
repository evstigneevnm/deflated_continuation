#ifndef CH_IMEX_TEST_SUPPORT_H
#define CH_IMEX_TEST_SUPPORT_H

#include "pde_test_support.h"
#include <nonlinear_operators/cahn_hilliard/convex_split_problem.h>
#include <nonlinear_operators/cahn_hilliard/energy_diagnostics.h>

namespace ch_tests
{
using namespace pde_tests;
namespace ch     = nonlinear_operators::cahn_hilliard;
using model_type = ch::model<double>;
template <class Vector>
struct profile_mapping
{
    model_type          model;
    Vector              out;
    double              mean, amplitude;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        const double x = ( i / model.grid.ny ) * model.grid.hx, y = ( i % model.grid.ny ) * model.grid.hy;
        out( i ) = mean + amplitude * ( std::cos( x ) + .4 * std::sin( y ) + .25 * std::cos( 2 * x ) * std::cos( y ) );
    }
};
template <class S>
void initialize(
    const S &ops, const model_type &model, typename S::vector_type &out, double mean = .2, double amplitude = .12
)
{
    execute( ops, model.grid.size(), profile_mapping<typename S::vector_type>{ model, out, mean, amplitude } );
}

template <class S, class Problem = ch::convex_split_problem<S>>
struct context
{
    using bundle_type   = typename Problem::imex_operator_type;
    using implicit_type = typename bundle_type::implicit_operator_type;
    using newton_type   = newton<S, implicit_type>;
    using control_type  = ts::integration::time_step_adaptation_constant<S>;
    using step_type     = ts::runge_kutta::imex_time_step<S, Problem, control_type, typename newton_type::solver>;
    std::shared_ptr<Problem>                      problem;
    std::shared_ptr<control_type>                 control;
    std::shared_ptr<typename newton_type::linear> linear;
    std::shared_ptr<typename newton_type::solver> solver;
    std::unique_ptr<step_type>                    step;
    context( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model, double h )
        : problem( std::make_shared<Problem>( ops, model ) ),
          control( std::make_shared<control_type>( typename control_type::params{ h } ) ),
          linear( std::make_shared<typename newton_type::linear>( ops ) ), solver( newton_type::make( ops, linear ) ),
          step(
              std::make_unique<step_type>( ops, problem, control, solver, typename step_type::params{ "IMEX_EULER" } )
          )
    {
    }
};
template <class S, class Problem>
double apply( context<S, Problem> &ctx, const typename S::vector_type &in, typename S::vector_type &out, double time )
{
    ctx.step->set_time( time );
    ctx.step->set_target_time( time + ctx.control->get_dt() );
    ctx.step->apply( in, out );
    check( ctx.step->get_status() == single_step_status::converged, "Convex-split CH stage failed" );
    return ctx.step->get_dt();
}
template <class S, class Problem = ch::convex_split_problem<S>>
void trajectory(
    std::shared_ptr<S> ops, std::shared_ptr<const model_type> model, double h, unsigned count,
    typename S::vector_type &result
)
{
    context<S,Problem> ctx( ops, model, h );
    buffer<S>  state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    initialize( *ops, *model, *state );
    double time = 0;
    for ( unsigned k = 0; k < count; ++k )
    {
        time += apply( ctx, *state, *out, time );
        ctx.step->finalize( adaptation_status::accepted, time, *out );
        ops->assign( *out, *state );
    }
    ops->assign( *state, result );
}

// Independent mixed solve of the SAME discrete convex-split Euler equation,
// with unknowns (c_new,mu_star), not an inverse of the singular mass matrix.
template <class S>
class mixed_reference
{
public:
    using vector_type = typename S::vector_type;
    using wrap_type   = buffer<S>;
    struct jacobi_operator_type
    {
        std::shared_ptr<S>                ops;
        std::shared_ptr<const model_type> model;
        wrap_type                         point;
        double                            a = 1, b = 1;
        jacobi_operator_type( std::shared_ptr<S> o, std::shared_ptr<const model_type> m )
            : ops( std::move( o ) ), model( std::move( m ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            model_type          model;
            vector_type         point, in, out;
            double              a, b;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( 2 * i )     = a * in( 2 * i ) - b * model.mobility * model.grid.apply( in, i, 1, 2 );
                out( 2 * i + 1 ) = in( 2 * i + 1 ) - 3 * point( 2 * i ) * point( 2 * i ) * in( 2 * i ) +
                                   model.epsilon_squared * model.grid.apply( in, i, 0, 2 );
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            execute( *ops, model->grid.size(), mapping{ *model, *point, in, out, a, b } );
        }
    };
    mixed_reference( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model )
        : ops_( std::move( ops ) ), model_( std::move( model ) ), previous_( *ops_ ), rhs_( *ops_ ),
          jacobian_( std::make_shared<jacobi_operator_type>( ops_, model_ ) )
    {
        check( ops_->size() == 2 * model_->grid.size(), "Mixed CH reference size mismatch" );
        previous_.start_use();
        rhs_.start_use();
    }
    vector_type &previous_state()
    {
        return *previous_;
    }
    vector_type &known_rhs()
    {
        return *rhs_;
    }
    const vector_type &previous_state() const
    {
        return *previous_;
    }
    const vector_type &known_rhs() const
    {
        return *rhs_;
    }
    void set_time( double )
    {
    }
    void set_affine_shift( double a, double b )
    {
        a_ = a;
        b_ = b;
    }
    struct mapping
    {
        model_type          model;
        vector_type         in, previous, out;
        double              a, b;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const double c   = in( 2 * i );
            out( 2 * i )     = a * ( c - previous( 2 * i ) ) - b * model.mobility * model.grid.apply( in, i, 1, 2 );
            out( 2 * i + 1 ) = in( 2 * i + 1 ) - c * c * c + previous( 2 * i ) +
                               model.epsilon_squared * model.grid.apply( in, i, 0, 2 );
        }
    };
    void apply( const vector_type &in, vector_type &out ) const
    {
        execute( *ops_, model_->grid.size(), mapping{ *model_, in, *previous_, out, a_, b_ } );
    }
    void set_linearization_point( const vector_type &in )
    {
        ops_->assign( in, *jacobian_->point );
        jacobian_->a = a_;
        jacobian_->b = b_;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian_;
    }

private:
    std::shared_ptr<S>                    ops_;
    std::shared_ptr<const model_type>     model_;
    wrap_type                             previous_, rhs_;
    std::shared_ptr<jacobi_operator_type> jacobian_;
    double                                a_ = 1, b_ = 1;
};
}
#endif
