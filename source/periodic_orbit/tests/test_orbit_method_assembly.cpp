#include "ode_orbit_support.h"
#include "rossler_periodic_problem.h"
#include "lorenz_periodic_problem.h"
#include "vaidyanathan_periodic_problem.h"
using namespace section_tests;

// Check the benchmark's identity-mass stage adapter independently of orbit
// correction: time-dependent rates, shifted Jv and the previous-state derivative.
struct time_dependent
{
    inline static constexpr std::array<double, 2> seed{ 1.2, -.7 };
    __DEVICE_TAG__ double                         rhs( std::ptrdiff_t i, const double *u, double t ) const
    {
        return t * u[i];
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *, const double *v, double t ) const
    {
        return t * v[i];
    }
};
template <class Model>
void check()
{
    auto                                       ops = std::make_shared<operations>( Model::seed.size() );
    orbit_benchmarks::stage_problem<Model>     problem( ops );
    orbit_benchmarks::stage_sensitivity<Model> sensitivity{ ops };
    auto x = buffer( ops ), v = buffer( ops ), plus = buffer( ops ), minus = buffer( ops );
    auto fp = buffer( ops ), fm = buffer( ops ), exact = buffer( ops ), numerical = buffer( ops );
    ode_orbit_tests::fill_profile( *x, Model::seed );
    ops->assign_scalar( .3, *v );
    ops->assign_scalar( .02, problem.known_rhs() );
    const double epsilon = 1e-5;
    for ( double time : { .4, 1.7 } )
        for ( const auto weights : { std::pair<double, double>{ 1, .01 }, { .01, 1 } } )
        {
            const auto a = weights.first, b = weights.second;
            problem.set_time( time );
            problem.set_affine_shift( a, b );
            ops->assign_lin_comb( .9, *x, problem.previous_state() );
            problem.set_linearization_point( *x );
            const auto allocated = guarded_memory::allocation_calls;
            problem.get_jacobi_operator()->apply( *v, *exact );
            problem.get_jacobi_operator()->apply( *v, *exact );
            require( guarded_memory::allocation_calls == allocated, "Benchmark stage Jv allocated vectors" );
            ops->assign_lin_comb( 1, *x, epsilon, *v, *plus );
            ops->assign_lin_comb( 1, *x, -epsilon, *v, *minus );
            problem.apply( *plus, *fp );
            problem.apply( *minus, *fm );
            ops->assign_lin_comb( .5 / epsilon, *fp, -.5 / epsilon, *fm, *numerical );
            const auto scale = 1 + ops->norm( *exact );
            ops->add_lin_comb( -1, *exact, *numerical );
            require( ops->norm( *numerical ) < 1e-7 * scale, "Benchmark shifted Jacobian finite difference" );
            periodic_orbit::dirk_stage_context<operations> context{ time, a, b, *x, problem.previous_state() };
            require( sensitivity.stage_tangent_rhs( context, *v, *exact ), "Benchmark stage tangent RHS failed" );
            // R is exactly affine in w; a larger displacement avoids subtracting
            // nearly equal large residuals without introducing truncation error.
            const double previous_epsilon = 1e-3;
            ops->add_lin_comb( previous_epsilon, *v, problem.previous_state() );
            problem.apply( *x, *fp );
            ops->add_lin_comb( -2 * previous_epsilon, *v, problem.previous_state() );
            problem.apply( *x, *fm );
            ops->assign_lin_comb( -.5 / previous_epsilon, *fp, .5 / previous_epsilon, *fm, *numerical );
            ops->add_lin_comb( -1, *exact, *numerical );
            const auto previous_error = ops->norm( *numerical );
            require( previous_error < 1e-7, "Benchmark previous-state tangent has wrong sign/weight" );
            require( problem.evaluate_rate( *x, *fp ), "Benchmark physical rate unavailable" );
            problem.set_affine_shift( 0, 1 );
            problem.apply( *x, *fm );
            ops->add_lin_comb( 1, *fp, *fm );
            ops->add_lin_comb( 1, problem.known_rhs(), *fm );
            require( ops->norm( *fm ) < 1e-12, "Benchmark physical rate depends on affine shift/RHS" );
            require( sensitivity.rate_jacobian_apply( time, *x, *v, *exact ), "Benchmark rate Jv failed" );
            problem.evaluate_rate( *plus, *fp );
            problem.evaluate_rate( *minus, *fm );
            ops->assign_lin_comb( .5 / epsilon, *fp, -.5 / epsilon, *fm, *numerical );
            const auto rate_scale = 1 + ops->norm( *exact );
            ops->add_lin_comb( -1, *exact, *numerical );
            require( ops->norm( *numerical ) < 1e-7 * rate_scale, "Benchmark rate Jacobian finite difference" );
            problem.set_affine_shift( a, b );
            problem.set_linearization_point( *x );
            auto linear = orbit_benchmarks::stage_solvers<Model>::make_linear( ops );
            linear->set_operator( problem.get_jacobi_operator() );
            ops->assign_scalar( 0, *exact );
            require( linear->solve( *v, *exact ), "Benchmark reference stage linear solve failed" );
            for ( const double amplitude : { 1e-12, 1e-16 } )
            {
                ops->assign_lin_comb( amplitude, *v, *fp );
                ops->assign_scalar( 0, *fm );
                require( linear->solve( *fp, *fm ), "Benchmark tiny stage direction failed" );
                ops->assign_lin_comb( 1 / amplitude, *fm, -1, *exact, *numerical );
                require(
                    ops->norm( *numerical ) < 1e-9 * ( 1 + ops->norm( *exact ) ),
                    "Benchmark stage inverse depends on direction magnitude"
                );
            }
        }
}
void test()
{
    check<time_dependent>();
    check<ode_orbit_tests::rossler_periodic>();
    check<ode_orbit_tests::lorenz_periodic>();
    check<ode_orbit_tests::vaidyanathan_periodic>();
    require( guarded_memory::host_bytes == 0, "Benchmark stage adapter transferred vectors" );
    require( guarded_memory::live_allocations == 0, "Benchmark stage adapter leaked workspace" );
}
int main()
{
    return run( test );
}
