#ifndef PDE_VALIDATION_TEST_SUPPORT_H
#define PDE_VALIDATION_TEST_SUPPORT_H

#include "implicit_test_support.h"
#include "projection_transfer_guard.h"
#include <nonlinear_operators/pde_validation/ode_problem.h>
#include <nonlinear_operators/pde_validation/cahn_hilliard.h>
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>
#include <nmfd/time_steppers/bdf/implicit_time_step.h>
#include <nmfd/time_steppers/rosenbrock/time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

namespace pde_tests
{
namespace pde       = nonlinear_operators::pde_validation;
namespace ts        = nmfd::time_steppers;
using space_type    = nmfd::operations::scfd_vector_space<backend_type, double>;
using guarded_space = nmfd::operations::scfd_vector_space<guarded_backend, double>;
using ts::adaptation_status;
using ts::single_step_status;
inline std::size_t checks = 0;
inline void        check( bool value, const char *message )
{
    ++checks;
    if ( !value )
        throw std::runtime_error( message );
}
template <class S>
using buffer = nmfd::detail::vector_wrap<S>;
template <class S, class Mapping>
void execute( const S &, std::size_t n, Mapping mapping )
{
    typename S::for_each_type each;
    each( mapping, n );
    each.wait();
}
template <class Model, class Vector>
struct exact_mapping
{
    Model               model;
    Vector              out;
    double              time;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        double value[Model::components];
        model.exact( i, time, value );
        for ( int c = 0; c < Model::components; ++c )
            out( Model::components * i + c ) = value[c];
    }
};
template <class S, class M>
void exact( const S &ops, const M &model, typename S::vector_type &out, double t )
{
    execute( ops, model.grid.size(), exact_mapping<M, typename S::vector_type>{ model, out, t } );
}
template <class S>
double difference(
    const S &ops, const typename S::vector_type &x, const typename S::vector_type &y, typename S::vector_type &scratch
)
{
    ops.assign_mul( 1., x, -1., y, scratch );
    return ops.norm( scratch ) / std::sqrt( double( ops.size() ) );
}

template <class Space, class Problem>
struct linear_solver
{
    using monitor = nmfd::solvers::monitor_krylov<Space, scfd::utils::log_std>;
    using core_type =
        nmfd::solvers::gmres<Space, monitor, scfd::utils::log_std, typename Problem::jacobi_operator_type>;
    using vector_type = typename Space::vector_type;
    core_type                         core;
    std::size_t                       calls = 0, fail_on = 0;
    static typename core_type::params parameters()
    {
        typename core_type::params p;
        p.basis_size            = 32;
        p.batch_size            = 1;
        p.monitor.max_iters_num = 300;
        p.monitor.abs_tol       = 5e-14;
        p.monitor.rel_tol       = 2e-13;
        return p;
    }
    explicit linear_solver( std::shared_ptr<Space> ops ) : core( ops, nullptr, parameters() )
    {
    }
    void set_operator( std::shared_ptr<const typename Problem::jacobi_operator_type> op )
    {
        core.set_operator( std::move( op ) );
    }
    bool solve( const vector_type &rhs, vector_type &out )
    {
        return ++calls != fail_on && core.solve( rhs, out );
    }
};
template <class Space, class P>
struct newton
{
    using linear    = linear_solver<Space, P>;
    using iteration = nmfd::solvers::newton_iteration<Space, P, linear>;
    using strategy  = nmfd::solvers::default_convergence_strategy<Space, scfd::utils::log_std, P>;
    using solver    = nmfd::solvers::nonlinear_solver<Space, scfd::utils::log_std, P, iteration>;
    static std::shared_ptr<solver> make( std::shared_ptr<Space> ops, std::shared_ptr<linear> lin )
    {
        typename strategy::params p;
        p.abs_tol       = 3e-12;
        p.rel_tol       = 1e-13;
        p.max_iters_num = 12;
        return std::make_shared<solver>(
            ops, nullptr, std::make_shared<iteration>( ops, lin ), std::make_shared<strategy>( ops, nullptr, p )
        );
    }
};

// Central differences check the actual shifted equation at nonzero RHS/time.
// Subsequent context/rate calls must not change its already prepared operator.
template <class S, class P, class M>
void contract( std::shared_ptr<S> ops, std::shared_ptr<P> p, const M &model )
{
    buffer<S> x( *ops ), v( *ops ), plus( *ops ), minus( *ops ), r1( *ops ), r2( *ops ), jv( *ops ), saved( *ops ),
        scratch( *ops );
    for ( auto *b : { &x, &v, &plus, &minus, &r1, &r2, &jv, &saved, &scratch } )
        b->start_use();
    exact( *ops, model, *x, .37 );
    exact( *ops, model, *v, .73 );
    ops->assign_scalar( .1, p->previous_state() );
    ops->assign_scalar( .03, p->known_rhs() );
    p->set_time( .41 );
    p->set_affine_shift( .6, .8 );
    p->set_linearization_point( *x );
    p->get_jacobi_operator()->apply( *v, *jv );
    ops->assign( *jv, *saved );
    ops->assign_mul( 1., *x, 1e-6, *v, *plus );
    ops->assign_mul( 1., *x, -1e-6, *v, *minus );
    p->apply( *plus, *r1 );
    p->apply( *minus, *r2 );
    ops->add_mul( -1., *r2, 1., *r1 );
    ops->scale( 5e5, *r1 );
    check( difference( *ops, *r1, *jv, *scratch ) < 1e-8, "PDE shifted Jacobian finite difference mismatch" );
    p->set_time( .82 );
    p->set_affine_shift( -.4, .2 );
    p->apply( *plus, *r1 );
    p->get_jacobi_operator()->apply( *v, *jv );
    check( difference( *ops, *saved, *jv, *scratch ) == 0, "PDE trial call corrupted frozen Jacobian" );
}
template <class S, class M>
void ode_contract( std::shared_ptr<S> ops, std::shared_ptr<const M> model )
{
    using P = pde::ode_problem<S, M>;
    auto p  = std::make_shared<P>( ops, model );
    contract( ops, p, *model );
    buffer<S> x( *ops ), f( *ops ), g( *ops ), e( *ops ), i( *ops ), d( *ops ), scratch( *ops );
    for ( auto *b : { &x, &f, &g, &e, &i, &d, &scratch } )
        b->start_use();
    exact( *ops, *model, *x, .41 );
    p->set_time( .41 );
    p->evaluate_rate( *x, *f );
    p->set_affine_shift( 0, 1 );
    ops->assign_scalar( .17, p->known_rhs() );
    p->evaluate_rate( *x, *g );
    check( difference( *ops, *f, *g, *scratch ) == 0, "PDE rate depends on affine/RHS context" );
    pde::imex_problem<S, M> split( ops, model );
    auto                    bundle = split.get_imex_operator();
    check( bundle == split.get_imex_operator(), "IMEX factory did not retain its bundle" );
    bundle->explicit_part->set_time( .41 );
    bundle->implicit_part->set_time( .41 );
    bundle->explicit_part->evaluate_rate( *x, *e );
    bundle->implicit_part->evaluate_rate( *x, *i );
    ops->add_lin_comb( 1., *i, *e );
    check( difference( *ops, *f, *e, *scratch ) < 2e-14, "PDE split changed the full equation" );
    p->evaluate_time_derivative( *x, *d );
    p->set_time( .41 + 1e-5 );
    p->evaluate_rate( *x, *f );
    p->set_time( .41 - 1e-5 );
    p->evaluate_rate( *x, *g );
    ops->add_mul( -1., *g, 1., *f );
    ops->scale( 5e4, *f );
    check( difference( *ops, *f, *d, *scratch ) < 1e-8, "PDE partial time derivative uses trajectory derivative" );
    contract( ops, bundle->implicit_part, *model );
}

// Exercise publication and fresh-trajectory semantics with a real coupled PDE
// problem. The failure injector is in test assembly, not the spatial operator.
template <class S, class P, class M>
void lifecycle( std::shared_ptr<S> ops, std::shared_ptr<P> p, const M &model )
{
    using C    = ts::integration::time_step_adaptation_constant<S>;
    using N    = newton<S, P>;
    using Step = ts::runge_kutta::implicit_time_step<S, P, C, typename N::solver, true>;
    buffer<S> x( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), first( *ops ), scratch( *ops );
    for ( auto *b : { &x, &initial, &out, &sentinel, &first, &scratch } )
        b->start_use();
    for ( bool alias : { false, true } )
    {
        auto control = std::make_shared<C>( typename C::params{ .04 } );
        auto lin     = std::make_shared<typename N::linear>( ops );
        lin->fail_on = 1;
        Step step( ops, p, control, N::make( ops, lin ), typename Step::params{ "SDIRK2(1)2" } );
        exact( *ops, model, *x, .3 );
        ops->assign( *x, *initial );
        ops->assign_scalar( 7., *out );
        ops->assign( *out, *sentinel );
        step.set_time( .3 );
        step.set_target_time( .5 );
        step.apply( *x, alias ? *x : *out );
        check( step.get_status() != single_step_status::converged, "PDE accepted a failed linear correction" );
        check( difference( *ops, *x, *initial, *scratch ) == 0, "PDE failure changed input" );
        check( difference( *ops, *out, *sentinel, *scratch ) == 0, "PDE failure published output" );
        bool refused = false;
        try
        {
            step.get_continuous_integration();
        }
        catch ( const std::logic_error & )
        {
            refused = true;
        }
        check( refused, "PDE failure published a dense view" );
        lin->fail_on = 0;
        step.reset();
        control->reset();
        step.set_time( .3 );
        step.set_target_time( .5 );
        step.apply( *x, *out );
        check( step.get_status() == single_step_status::converged, "PDE correction did not recover" );
        ops->assign( *out, *first );
        auto         view = step.get_continuous_integration();
        const double t    = view.evaluate( .5, *scratch );
        step.finalize( adaptation_status::accepted_modified, t, *scratch );
        refused = false;
        try
        {
            view.evaluate( .5, *out );
        }
        catch ( const std::logic_error & )
        {
            refused = true;
        }
        check( refused, "PDE shortened commit retained a stale dense view" );
        step.reset();
        control->reset();
        step.set_time( .3 );
        step.set_target_time( .5 );
        step.apply( *initial, *out );
        check( difference( *ops, *first, *out, *scratch ) < 1e-12, "PDE restart retained stale stage context" );
    }
}

template <class S, class M>
void constraints( const S &, const M &, const typename S::vector_type &, typename S::vector_type & )
{
}
template <class Vector>
struct ch_checks
{
    pde::cahn_hilliard_model<double> model;
    Vector                           in, out;
    bool                             mean;
    __DEVICE_TAG__ void              operator()( std::ptrdiff_t i ) const
    {
        out( 2 * i ) =
            mean ? in( 2 * i )
                 : in( 2 * i + 1 ) - model.kappa * in( 2 * i ) + model.epsilon2 * model.grid.apply( in, i, 0, 2 );
        out( 2 * i + 1 ) = 0;
    }
};
template <class S>
void constraints(
    const S &ops, const pde::cahn_hilliard_model<double> &model, const typename S::vector_type &state,
    typename S::vector_type &scratch
)
{
    execute( ops, model.grid.size(), ch_checks<typename S::vector_type>{ model, state, scratch, false } );
    check(
        ops.norm( scratch ) / std::sqrt( double( model.grid.size() ) ) < 2e-10, "Mixed CH algebraic constraint drift"
    );
    execute( ops, model.grid.size(), ch_checks<typename S::vector_type>{ model, state, scratch, true } );
    check( std::abs( ops.sum( scratch ) / model.grid.size() - .3 ) < 2e-11, "Mixed CH lost mean concentration" );
}

enum class engine
{
    explicit_rk,
    dirk,
    imex,
    bdf,
    rosenbrock
};
struct report
{
    double   error = 0, dense_error = 0;
    unsigned steps = 0, order = 0;
};
template <engine E, bool Adaptive, bool Dense, class S, class M, class P>
report integrate(
    std::shared_ptr<S> ops, std::shared_ptr<const M> model, std::shared_ptr<P> problem, const char *method, double h,
    double tolerance = 1e-6, double end = .62
)
{
    using C = std::conditional_t<
        Adaptive, ts::integration::time_step_adaptation_scipy<S>, ts::integration::time_step_adaptation_constant<S>>;
    using IP = std::conditional_t<
        E == engine::imex, pde::ode_problem<S, M, pde::partition::implicit_part>,
        std::conditional_t<E == engine::explicit_rk, pde::ode_problem<S, M>, P>>;
    using N      = newton<S, IP>;
    using L      = typename N::linear;
    using Solver = std::conditional_t<E == engine::rosenbrock, L, typename N::solver>;
    using Step   = std::conditional_t<
          E == engine::explicit_rk, ts::runge_kutta::explicit_time_step<S, P, C, Dense>,
          std::conditional_t<
              E == engine::dirk, ts::runge_kutta::implicit_time_step<S, P, C, Solver, Dense>,
              std::conditional_t<
                  E == engine::imex, ts::runge_kutta::imex_time_step<S, P, C, Solver, Dense>,
                  std::conditional_t<
                      E == engine::bdf, ts::bdf::implicit_time_step<S, P, C, Solver, Dense>,
                      ts::rosenbrock::time_step<S, P, C, Solver, Dense>>>>>;
    auto control = [&] {
        typename C::params cp;
        if constexpr ( Adaptive )
        {
            cp.initial_step       = std::abs( h );
            cp.maximum_step       = std::abs( h );
            cp.relative_tolerance = tolerance;
            cp.absolute_tolerance = .01 * tolerance;
            return std::make_shared<C>( ops, cp );
        }
        else
        {
            cp.step_size = std::abs( h );
            return std::make_shared<C>( cp );
        }
    }();
    auto lin = [&] {
        if constexpr ( E == engine::explicit_rk )
            return std::shared_ptr<L>{};
        else
            return std::make_shared<L>( ops );
    }();
    typename Step::params sp{ method };
    if constexpr ( E == engine::bdf )
        sp.adaptive = Adaptive;
    if constexpr ( E == engine::rosenbrock )
    {
        sp.autonomous = std::is_same_v<M, pde::cahn_hilliard_model<double>>;
        if constexpr ( std::is_same_v<M, pde::cahn_hilliard_model<double>> )
        {
            sp.mass_form  = true;
            sp.index1_dae = true;
        }
    }
    auto step = [&] {
        if constexpr ( E == engine::explicit_rk )
            return std::make_unique<Step>( ops, problem, control, sp );
        else if constexpr ( E == engine::rosenbrock )
            return std::make_unique<Step>( ops, problem, control, lin, sp );
        else
            return std::make_unique<Step>( ops, problem, control, N::make( ops, lin ), sp );
    }();
    buffer<S> state( *ops ), out( *ops ), reference( *ops ), sample( *ops ), scratch( *ops );
    for ( auto *b : { &state, &out, &reference, &sample, &scratch } )
        b->start_use();
    double time = h > 0 ? .3 : .62;
    exact( *ops, *model, *state, time );
    report      result;
    std::size_t allocated = 0;
    while ( time != end && result.steps < 3000 )
    {
        step->set_time( time );
        step->set_target_time( end );
        step->apply( *state, *out );
        check( step->get_status() == single_step_status::converged, "PDE step failed" );
        constraints( *ops, *model, *out, *scratch );
        if constexpr ( Dense )
        {
            auto         view = step->get_continuous_integration();
            const double t    = view.evaluate( .37, *sample );
            exact( *ops, *model, *reference, t );
            result.dense_error = std::max( result.dense_error, difference( *ops, *sample, *reference, *scratch ) );
        }
        if constexpr ( E == engine::bdf )
            result.order = std::max( result.order, step->get_order() );
        time = step->get_dt() == end - time ? end : time + step->get_dt();
        step->finalize( adaptation_status::accepted, time, *out );
        ops->assign( *out, *state );
        if constexpr ( std::is_same_v<typename S::backend_type, guarded_backend> )
        {
            if ( result.steps == 0 )
                allocated = guarded_memory::allocation_count;
            check( allocated == guarded_memory::allocation_count, "PDE allocated backend workspace after warmup" );
        }
        ++result.steps;
    }
    check( time == end, "PDE trajectory exhausted its step budget" );
    exact( *ops, *model, *reference, time );
    result.error = difference( *ops, *state, *reference, *scratch );
    std::cout << "RESULT " << method << " n=" << model->grid.nx << " steps=" << result.steps << " h=" << h
              << " tolerance=" << tolerance << " error=" << result.error << " dense_error=" << result.dense_error
              << '\n';
    return result;
}
template <engine E, bool Adaptive, bool Dense, class S, class M>
report ode_run(
    std::shared_ptr<S> ops, std::shared_ptr<const M> model, const char *method, double h, double tol = 1e-6,
    double end = .62
)
{
    if constexpr ( E == engine::explicit_rk )
        return integrate<E, Adaptive, Dense>(
            ops, model, std::make_shared<pde::rate_problem<S, M>>( ops, model ), method, h, tol, end
        );
    else if constexpr ( E == engine::imex )
        return integrate<E, Adaptive, Dense>(
            ops, model, std::make_shared<pde::imex_problem<S, M>>( ops, model ), method, h, tol, end
        );
    else
        return integrate<E, Adaptive, Dense>(
            ops, model, std::make_shared<pde::ode_problem<S, M>>( ops, model ), method, h, tol, end
        );
}
template <engine E, class M>
void ode_case(
    std::shared_ptr<space_type> ops, std::shared_ptr<const M> model, const char *fixed, const char *adaptive, int order
)
{
    const auto   a        = ode_run<E, false, true>( ops, model, fixed, .08 );
    const auto   b        = ode_run<E, false, true>( ops, model, fixed, .04 );
    const double observed = std::log2( a.error / b.error );
    std::cout << "ORDER " << fixed << " observed=" << observed << " expected=" << order << '\n';
    check( observed > order - .45 && b.error < .005 && b.dense_error < .005, "PDE temporal order/accuracy regression" );
    const auto loose = ode_run<E, true, true>( ops, model, adaptive, .16, 1e-3 );
    const auto tight = ode_run<E, true, true>( ops, model, adaptive, .16, E == engine::explicit_rk ? 1e-11 : 1e-7 );
    check( tight.error < loose.error * .6, "PDE adaptive tolerance refinement failed" );
    ode_run<E, false, true>( ops, model, fixed, -.04, 1e-6, .3 );
}
template <class M>
void ode_suite()
{
    auto model = std::make_shared<const M>( 8 );
    auto ops   = std::make_shared<space_type>( M::components * model->grid.size() );
    ode_contract( ops, model );
    lifecycle( ops, std::make_shared<pde::ode_problem<space_type, M>>( ops, model ), *model );
    ode_case<engine::explicit_rk>( ops, model, "RK4", "RK45", 4 );
    ode_case<engine::dirk>( ops, model, "SDIRK2(1)2", "ARK324L2SA", 2 );
    ode_case<engine::imex>( ops, model, "IMEX_ARK324L2SA", "IMEX_ARK324L2SA", 3 );
    ode_case<engine::bdf>( ops, model, "BDF2", "BDF", 2 );
    ode_case<engine::rosenbrock>( ops, model, "ROS34PW2", "ROS34PW2", 3 );
    ode_run<engine::dirk, false, true>( ops, model, "ESDIRK2(1)3", .04 );
    auto large  = std::make_shared<const M>( 64 );
    auto device = std::make_shared<guarded_space>( M::components * large->grid.size() );
    ode_run<engine::explicit_rk, false, true>( device, large, "RK4", 1e-6, 1e-6, .300002 );
    ode_run<engine::dirk, false, true>( device, large, "SDIRK2(1)2", 1e-6, 1e-6, .300002 );
    ode_run<engine::imex, false, false>( device, large, "IMEX_ARS222", 1e-6, 1e-6, .300002 );
    ode_run<engine::bdf, false, true>( device, large, "BDF2", 1e-6, 1e-6, .300002 );
    ode_run<engine::rosenbrock, false, true>( device, large, "ROS34PW2", 1e-6, 1e-6, .300002 );
}
inline void finish()
{
    check( guarded_memory::allocations.empty(), "PDE leaked guarded workspace" );
    std::cout << "PASS " << checks << " PDE contract checks; bulk host transfer bytes=0\n";
}
}
#endif
