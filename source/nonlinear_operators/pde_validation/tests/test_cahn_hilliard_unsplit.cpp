#include "cahn_hilliard_imex_test_support.h"
#include <nonlinear_operators/cahn_hilliard/unsplit_problem.h>
using namespace ch_tests;

// The SAME nonlinear, unforced periodic CH equation as the convex-split test:
// c_t=m*Delta_h(c^3-c-epsilon^2*Delta_h(c)). Concentration alone has identity
// mass. The continuous semidiscrete flow conserves mass and dissipates energy;
// no convex-split discrete energy inequality is imposed on unsplit methods.
template <class Vector>
struct thermodynamics_mapping
{
    model_type          model;
    Vector              state, out;
    bool                energy;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        const double c = state( i ), lap = model.grid.apply( state, i, 0, 1 );
        out( i ) = energy ? ( c * c - 1 ) * ( c * c - 1 ) / 4 - model.epsilon_squared * c * lap / 2
                          : c * c * c - c - model.epsilon_squared * lap;
    }
};
template <class S>
void operator_contract( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model )
{
    ch::unsplit_problem<S>      p( ops, model );
    ch::convex_split_problem<S> split( ops, model );
    auto                        bundle = split.get_imex_operator();
    buffer<S> x( *ops ), v( *ops ), f( *ops ), g( *ops ), residual( *ops ), jv( *ops ), saved( *ops ), plus( *ops ),
        minus( *ops ), scratch( *ops ), mu( *ops );
    for ( auto *value : { &x, &v, &f, &g, &residual, &jv, &saved, &plus, &minus, &scratch, &mu } )
        value->start_use();
    initialize( *ops, *model, *x );
    initialize( *ops, *model, *v, .1, .3 );
    check( p.evaluate_rate( *x, *f ), "Unsplit CH rate evaluation failed" );
    bundle->implicit_part->evaluate_rate( *x, *g );
    bundle->explicit_part->evaluate_rate( *x, *residual );
    ops->add_lin_comb( 1., *residual, *g );
    check( difference( *ops, *f, *g, *scratch ) < 1e-12, "Full CH rate differs from summed split rates" );
    check( std::abs( model->cell_area() * ops->sum( *f ) ) < 1e-11, "Physical CH rate changes mass" );
    execute( *ops, ops->size(), thermodynamics_mapping<typename S::vector_type>{ *model, *x, *mu, false } );
    check(
        model->cell_area() * ops->scalar_prod( *mu, *f ) < 0, "Semidiscrete physical CH flow did not dissipate energy"
    );

    p.set_time( .37 );
    p.set_affine_shift( .6, .8 );
    ops->assign_scalar( .17, p.previous_state() );
    ops->assign_scalar( .03, p.known_rhs() );
    p.apply( *x, *residual );
    ops->assign_mul( .6, *x, -.8, *f, *g );
    ops->add_mul_scalar( -.6 * .17 - .8 * .03, 1., *g );
    check( difference( *ops, *g, *residual, *scratch ) < 1e-12, "Unsplit CH shifted residual has wrong sign/context" );
    p.set_linearization_point( *x );
    auto j = p.get_jacobi_operator();
    j->apply( *v, *jv );
    ops->assign( *jv, *saved );
    ops->assign_mul( 1., *x, 1e-6, *v, *plus );
    ops->assign_mul( 1., *x, -1e-6, *v, *minus );
    execute( *ops, ops->size(), thermodynamics_mapping<typename S::vector_type>{ *model, *plus, *scratch, true } );
    const double ep = model->cell_area() * ops->sum( *scratch );
    execute( *ops, ops->size(), thermodynamics_mapping<typename S::vector_type>{ *model, *minus, *scratch, true } );
    const double em = model->cell_area() * ops->sum( *scratch );
    check(
        std::abs( ( ep - em ) / 2e-6 - model->cell_area() * ops->scalar_prod( *mu, *v ) ) < 1e-8,
        "CH physical potential differs from discrete energy gradient"
    );
    p.apply( *plus, *f );
    p.apply( *minus, *g );
    ops->add_mul( -1., *g, 1., *f );
    ops->scale( 5e5, *f );
    check( difference( *ops, *f, *jv, *scratch ) < 1e-8, "Unsplit CH Jacobian finite-difference mismatch" );

    p.evaluate_rate( *x, *f );
    p.set_time( .89 );
    p.set_affine_shift( -.4, .2 );
    ops->assign_scalar( 4., p.previous_state() );
    ops->assign_scalar( 12., p.known_rhs() );
    p.apply( *plus, *g );
    p.evaluate_rate( *x, *g );
    check( difference( *ops, *f, *g, *scratch ) == 0, "Physical CH rate depends on affine context" );
    j->apply( *v, *jv );
    check( difference( *ops, *saved, *jv, *scratch ) == 0, "Trial CH residual/rate changed prepared Jacobian" );
    ops->scale( .25, *v );
    j->apply( *v, *jv );
    ops->scale( .25, *saved );
    check( difference( *ops, *saved, *jv, *scratch ) < 1e-13, "Unsplit CH Jv is not linear" );
}

void invalid_construction_and_shifts( std::shared_ptr<guarded_space> ops, std::shared_ptr<const model_type> model )
{
    using P = ch::unsplit_problem<guarded_space>;
    for ( unsigned missing : { 0u, 1u, 2u } )
    {
        bool rejected = false;
        try
        {
            auto space = missing == 0   ? nullptr
                         : missing == 2 ? std::make_shared<guarded_space>( ops->size() + 1 )
                                        : ops;
            P    invalid( space, missing == 1 ? nullptr : model );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        check( rejected, "Unsplit CH accepted missing or mismatched dependencies" );
    }
    P p( ops, model );
    for ( double b : { 0., -.1, std::numeric_limits<double>::infinity() } )
    {
        bool rejected = false;
        try
        {
            p.set_affine_shift( 1, b );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        check( rejected, "Unsplit CH accepted an invalid RHS coefficient" );
    }
    bool rejected = false;
    try
    {
        p.set_affine_shift( std::numeric_limits<double>::quiet_NaN(), 1 );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    check( rejected, "Unsplit CH accepted a nonfinite mass coefficient" );
}

// At a homogeneous equilibrium c0, a Fourier perturbation has growth rate
// sigma=m*lambda_h*((3*c0^2-1)-epsilon^2*lambda_h), lambda_h<=0 for Delta_h.
// This independently detects
// omission of the concave -1 derivative, including stable and unstable modes.
template <class Vector>
struct fourier_mapping
{
    model_type          model;
    Vector              out;
    unsigned            kx, ky;
    double              factor;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = factor * std::cos( kx * ( i / model.grid.ny ) * model.grid.hx ) *
                   std::cos( ky * ( i % model.grid.ny ) * model.grid.hy );
    }
};
template <class S>
void dispersion( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model )
{
    ch::unsplit_problem<S> p( ops, model );
    buffer<S>              x( *ops ), v( *ops ), jv( *ops ), expected( *ops ), scratch( *ops );
    for ( auto *value : { &x, &v, &jv, &expected, &scratch } )
        value->start_use();
    for ( double mean : { .2, 1. } )
    {
        ops->assign_scalar( mean, *x );
        p.set_affine_shift( .7, .3 );
        p.set_linearization_point( *x );
        for ( unsigned k : { 0u, 1u, 3u } )
        {
            const double lambda = -model->grid.eigenvalue( k, k );
            const double sigma  = model->mobility * lambda * ( 3 * mean * mean - 1 - model->epsilon_squared * lambda );
            execute( *ops, ops->size(), fourier_mapping<typename S::vector_type>{ *model, *v, k, k, 1. } );
            execute(
                *ops, ops->size(), fourier_mapping<typename S::vector_type>{ *model, *expected, k, k, .7 - .3 * sigma }
            );
            p.get_jacobi_operator()->apply( *v, *jv );
            check( difference( *ops, *expected, *jv, *scratch ) < 1e-11, "Unsplit CH discrete dispersion mismatch" );
        }
    }
}

template <engine E, class S>
struct unsplit_context
{
    using problem_type = ch::unsplit_problem<S>;
    using newton_type  = newton<S, problem_type>;
    using control_type = ts::integration::time_step_adaptation_constant<S>;
    using solver_type =
        std::conditional_t<E == engine::rosenbrock, typename newton_type::linear, typename newton_type::solver>;
    using step_type = std::conditional_t<
        E == engine::dirk, ts::runge_kutta::implicit_time_step<S, problem_type, control_type, solver_type>,
        std::conditional_t<
            E == engine::bdf, ts::bdf::implicit_time_step<S, problem_type, control_type, solver_type>,
            ts::rosenbrock::time_step<S, problem_type, control_type, solver_type>>>;
    std::shared_ptr<problem_type>                 problem;
    std::shared_ptr<control_type>                 control;
    std::shared_ptr<typename newton_type::linear> linear;
    step_type                                     step;
    static std::shared_ptr<solver_type>
    make_solver( std::shared_ptr<S> ops, std::shared_ptr<typename newton_type::linear> linear )
    {
        if constexpr ( E == engine::rosenbrock )
            return linear;
        else
            return newton_type::make( ops, linear );
    }
    unsplit_context( std::shared_ptr<S> ops, std::shared_ptr<const model_type> model, double h, const char *method )
        : problem( std::make_shared<problem_type>( ops, model ) ),
          control( std::make_shared<control_type>( typename control_type::params{ h } ) ),
          linear( std::make_shared<typename newton_type::linear>( ops ) ),
          step( ops, problem, control, make_solver( ops, linear ), typename step_type::params{ method } )
    {
    }
};

template <engine E, class S>
void integrate_unsplit(
    std::shared_ptr<S> ops, std::shared_ptr<const model_type> model, const char *method, double end, unsigned count,
    typename S::vector_type &result, double amplitude = .12
)
{
    const double          h = end / count;
    unsplit_context<E, S> ctx( ops, model, h, method );
    buffer<S>             state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    initialize( *ops, *model, *state, .2, amplitude );
    const double mass   = model->cell_area() * ops->sum( *state );
    std::size_t  warmup = 0;
    for ( unsigned k = 0; k < count; ++k )
    {
        const double time = k * h, next = ( k + 1 == count ? end : ( k + 1 ) * h );
        ctx.step.set_time( time );
        ctx.step.set_target_time( next );
        ctx.step.apply( *state, *out );
        check( ctx.step.get_status() == single_step_status::converged, "Unsplit CH stage failed" );
        check( ops->is_valid_number( *out ), "Unsplit CH produced nonfinite state" );
        check( std::abs( ctx.step.get_dt() - ( next - time ) ) < 1e-13 * h, "Unsplit CH endpoint mismatch" );
        check( std::abs( model->cell_area() * ops->sum( *out ) - mass ) < 2e-10, "Unsplit CH integration lost mass" );
        ctx.step.finalize( adaptation_status::accepted, next, *out );
        ops->assign( *out, *state );
        if ( k == 0 )
            warmup = guarded_memory::allocation_count;
        check( warmup == guarded_memory::allocation_count, "Unsplit CH allocated backend vectors after warmup" );
    }
    ops->assign( *state, result );
}

template <engine E>
void accuracy(
    std::shared_ptr<guarded_space> ops, std::shared_ptr<const model_type> model, const char *method,
    unsigned expected_order, const guarded_space::vector_type &reference, double reference_difference
)
{
    buffer<guarded_space> coarse( *ops ), fine( *ops ), scratch( *ops ), initial( *ops );
    for ( auto *value : { &coarse, &fine, &scratch, &initial } )
        value->start_use();
    integrate_unsplit<E>( ops, model, method, .2, 5, *coarse );
    integrate_unsplit<E>( ops, model, method, .2, 10, *fine );
    const double ec    = difference( *ops, *coarse, reference, *scratch ),
                 ef    = difference( *ops, *fine, reference, *scratch );
    const double order = std::log2( ec / ef );
    check( reference_difference < .1 * ef, "Unsplit CH reference disagreement exceeds accuracy budget" );
    std::cout << "ORDER UNSPLIT " << method << " observed=" << order << " expected=" << expected_order
              << " coarse_error=" << ec << " fine_error=" << ef << '\n';
    check(
        ef > 1e-11 && order > expected_order - .5 && order < expected_order + .5, "Unsplit CH temporal order regression"
    );
    integrate_unsplit<E>( ops, model, method, .2, 2, *fine, 0 );
    initialize( *ops, *model, *initial, .2, 0 );
    check( difference( *ops, *fine, *initial, *scratch ) < 1e-12, "Unsplit CH homogeneous equilibrium moved" );
}

void rollback_and_ownership( std::shared_ptr<guarded_space> ops, std::shared_ptr<const model_type> model )
{
    using context_type = unsplit_context<engine::dirk, guarded_space>;
    buffer<guarded_space> initial( *ops ), state( *ops ), out( *ops ), sentinel( *ops ), first( *ops ), scratch( *ops );
    for ( auto *value : { &initial, &state, &out, &sentinel, &first, &scratch } )
        value->start_use();
    initialize( *ops, *model, *initial );
    for ( bool alias : { false, true } )
    {
        context_type ctx( ops, model, .02, "IE" );
        ctx.linear->fail_on = 1;
        ops->assign( *initial, *state );
        ops->assign_scalar( 7., *out );
        ops->assign( *out, *sentinel );
        ctx.step.set_time( .3 );
        ctx.step.set_target_time( .32 );
        ctx.step.apply( *state, alias ? *state : *out );
        check( ctx.step.get_status() != single_step_status::converged, "Unsplit CH accepted a failed linear solve" );
        check( difference( *ops, *initial, *state, *scratch ) == 0, "Unsplit CH failure changed committed input" );
        check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Unsplit CH failure published output" );
        ctx.linear->fail_on = 0;
        for ( unsigned repeat = 0; repeat < 2; ++repeat )
        {
            ctx.step.reset();
            ctx.control->reset();
            ctx.step.set_time( .3 );
            ctx.step.set_target_time( .32 );
            ctx.step.apply( *initial, *out );
            check( ctx.step.get_status() == single_step_status::converged, "Unsplit CH did not recover after reset" );
            if ( repeat == 0 )
                ops->assign( *out, *first );
            else
                check( difference( *ops, *out, *first, *scratch ) < 1e-12, "Unsplit CH restart retained trial data" );
            ctx.step.finalize( adaptation_status::accepted, .32, *out );
        }
    }
    using P = ch::unsplit_problem<guarded_space>;
    std::weak_ptr<const typename P::jacobi_operator_type> weak;
    {
        auto p = std::make_shared<P>( ops, model );
        p->set_affine_shift( 1, .1 );
        p->set_linearization_point( *initial );
        auto j = p->get_jacobi_operator();
        weak   = j;
        j->apply( *initial, *first );
        p.reset();
        j->apply( *initial, *out );
        check( difference( *ops, *first, *out, *scratch ) == 0, "Retained CH Jacobian lost its prepared state" );
    }
    check( weak.expired(), "Unsplit CH ownership cycle" );
}

int main()
try
{
    backend_type::init_device();
    {
        auto model = std::make_shared<const model_type>( 8 );
        auto ops   = std::make_shared<guarded_space>( model->grid.size() );
        operator_contract( ops, model );
        dispersion( ops, model );
        invalid_construction_and_shifts( ops, model );
        rollback_and_ownership( ops, model );
        buffer<guarded_space> reference( *ops ), refined( *ops ), scratch( *ops );
        for ( auto *value : { &reference, &refined, &scratch } )
            value->start_use();
        integrate_unsplit<engine::rosenbrock>( ops, model, "ROS3", .2, 200, *reference );
        integrate_unsplit<engine::rosenbrock>( ops, model, "ROS3", .2, 400, *refined );
        const double reference_difference = difference( *ops, *reference, *refined, *scratch );
        check( reference_difference < 1e-10, "Unsplit CH reference is unresolved" );
        accuracy<engine::dirk>( ops, model, "IE", 1, *refined, reference_difference );
        accuracy<engine::dirk>( ops, model, "CN", 2, *refined, reference_difference );
        accuracy<engine::dirk>( ops, model, "SDIRK2(1)2", 2, *refined, reference_difference );
        accuracy<engine::bdf>( ops, model, "BDF2", 2, *refined, reference_difference );
        accuracy<engine::rosenbrock>( ops, model, "ROS3", 3, *refined, reference_difference );
    }
    {
        auto                  model = std::make_shared<const model_type>( 64 );
        auto                  ops   = std::make_shared<guarded_space>( model->grid.size() );
        buffer<guarded_space> result( *ops );
        result.start_use();
        integrate_unsplit<engine::dirk>( ops, model, "IE", 3e-6, 3, *result );
        integrate_unsplit<engine::bdf>( ops, model, "BDF2", 3e-6, 3, *result );
        integrate_unsplit<engine::rosenbrock>( ops, model, "ROS3", 3e-6, 3, *result );
    }
    finish();
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
