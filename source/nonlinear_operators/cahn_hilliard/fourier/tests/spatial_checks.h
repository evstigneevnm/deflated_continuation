#ifndef CH_FOURIER_SPATIAL_CHECKS_H
#define CH_FOURIER_SPATIAL_CHECKS_H

#include "benchmark_types.h"
#include "../biharmonic_preconditioner.h"
#include "cahn_hilliard_imex_test_support.h"

namespace ch_fourier_tests
{
using namespace ch_tests;
using S     = guarded_space;
using split = ch::fourier::split_problem<S, ch_fft_backend>;
using full  = ch::fourier::problem<S, ch_fft_backend, false>;

// For c=d+A*cos(phi) at the cutoff, only the constant and first harmonic
// of c^3 survive projection. An aliased cube (or a truncated square) fails.
inline void modes( int n )
{
    auto      m   = std::make_shared<const model_type>( n );
    auto      ops = std::make_shared<S>( n * n );
    full      problem( ops, m );
    buffer<S> x( *ops ), v( *ops ), out( *ops ), expected( *ops ), scratch( *ops ), jv( *ops ), plus( *ops ),
        minus( *ops );
    for ( auto b : { &x, &v, &out, &expected, &scratch, &jv, &plus, &minus } )
        b->start_use();
    const int    k = ( n - 1 ) / 2, k2 = k * k + 4;
    const double h = m->grid.hx, A = .3, d = .2, eps2 = m->epsilon_squared, mobility = m->mobility;
    auto         u = *x, dir = *v, ref = *expected;
    execute( *ops, n * n, [=] __DEVICE_TAG__( int i ) {
        const double wave = std::cos( h * ( k * ( i / n ) - 2 * ( i % n ) ) );
        u( i )            = d + A * wave;
        dir( i )          = .1 + std::sin( h * ( i % n ) );
        ref( i )          = -mobility * k2 * A * ( 3 * d * d + .75 * A * A - 1 + eps2 * k2 ) * wave;
    } );
    problem.evaluate_rate( *x, *out );
    check( difference( *ops, *out, *expected, *scratch ) < 2e-13, "Fourier CH retained cubic mode mismatch" );
    check( std::abs( ops->sum( *out ) ) < 1e-12, "Fourier CH mean rate is nonzero" );
    ops->assign_scalar( .17, problem.previous_state() );
    ops->assign_scalar( .2, problem.known_rhs() );
    problem.set_affine_shift( .7, .3 );
    problem.set_linearization_point( *x );
    ops->assign_mul( 1, *x, 1e-5, *v, *plus );
    ops->assign_mul( 1, *x, -1e-5, *v, *minus );
    problem.apply( *plus, *out );
    problem.apply( *minus, *expected );
    ops->add_mul( -1, *expected, 1, *out );
    ops->scale( .5e5, *out );
    problem.set_affine_shift( 1, .1 ); // Trial residuals and shifts must not change the prepared Jv.
    problem.get_jacobi_operator()->apply( *v, *jv );
    check( difference( *ops, *jv, *out, *scratch ) < 3e-9, "Fourier shifted Jacobian mismatch" );
    if ( n % 2 == 0 )
    {
        execute( *ops, n * n, [=] __DEVICE_TAG__( int i ) { u( i ) = d + ( ( i / n ) % 2 ? 1. : -1. ); } );
        problem.evaluate_rate( *x, *out );
        check( ops->norm( *out ) < 1e-12, "Excluded Nyquist mode contaminated retained rates" );
    }
}

inline double
nested_difference( std::shared_ptr<S> ops, int n, int nf, const S::vector_type &coarse, const S::vector_type &fine )
{
    buffer<S> work( *ops );
    work.start_use();
    auto diff = *work;
    execute( *ops, n * n, [=] __DEVICE_TAG__( int i ) {
        diff( i ) = coarse( i ) - fine( ( i / n ) * ( nf / n ) * nf + ( i % n ) * ( nf / n ) );
    } );
    return ops->norm( *work ) / n;
}

// At c=0 the convex Jacobian equals the shifted biharmonic principal part.
// Exercise the full nodal space, including the mean and an excluded Nyquist line.
inline void shifted_inverse()
{
    using problem_type                                       = split::implicit_operator_type;
    constexpr int                                        n   = 16;
    auto                                                 m   = std::make_shared<const model_type>( n );
    auto                                                 ops = std::make_shared<S>( n * n );
    problem_type                                         problem( ops, m );
    ch::fourier::biharmonic_preconditioner<problem_type> inverse;
    buffer<S> zero( *ops ), rhs( *ops ), correction( *ops ), action( *ops ), scratch( *ops );
    for ( auto v : { &zero, &rhs, &correction, &action, &scratch } )
        v->start_use();
    ops->assign_scalar( 0, *zero );
    auto       r = *rhs;
    const auto h = m->grid.hx;
    execute( *ops, n * n, [=] __DEVICE_TAG__( int i ) {
        r( i ) = .2 + std::cos( h * ( 7 * ( i / n ) + 4 * ( i % n ) ) ) + ( ( i / n ) % 2 ? -.3 : .3 );
    } );
    check( !inverse.apply( *rhs ), "Unbound Fourier inverse reported success" );
    for ( double a : { .75, 2. } )
    {
        problem.set_affine_shift( a, .1 );
        problem.set_linearization_point( *zero );
        auto jac = problem.get_jacobi_operator();
        inverse.set_operator( jac );
        ops->assign( *rhs, *correction );
        check( inverse.apply( *correction ), "Fourier inverse failed on positive shift" );
        jac->apply( *correction, *action );
        check( difference( *ops, *rhs, *action, *scratch ) < 1e-13, "Shifted Fourier inverse residual mismatch" );
    }
    problem.set_affine_shift( 0, 1 );
    problem.set_linearization_point( *zero );
    check( !inverse.apply( *correction ), "Fourier inverse accepted a singular mean mode" );
}

inline void verify()
{
    shifted_inverse();
    modes( 8 );
    modes( 9 );
    // Same nonlinear profile and final physical time. Resolve the reference
    // in space and time independently before measuring FD spatial convergence.
    const double end = .005;
    auto         m   = std::make_shared<const model_type>( 32 );
    auto         ops = std::make_shared<S>( 32 * 32 );
    buffer<S>    ref( *ops ), time_coarse( *ops );
    ref.start_use();
    time_coarse.start_use();
    trajectory<S, split>( ops, m, end / 32, 32, *ref );
    trajectory<S, split>( ops, m, end / 16, 16, *time_coarse );
    const double temporal = nested_difference( ops, 32, 32, *time_coarse, *ref );
    double       spatial = 0, prev = 0;
    for ( int n : { 8, 16, 32 } )
    {
        auto      cm = std::make_shared<const model_type>( n );
        auto      co = std::make_shared<S>( n * n );
        buffer<S> state( *co );
        state.start_use();
        if ( n == 16 )
        {
            trajectory<S, split>( co, cm, end / 32, 32, *state );
            spatial = nested_difference( co, n, 32, *state, *ref );
        }
        trajectory<S>( co, cm, end / 32, 32, *state );
        const double rms = nested_difference( co, n, 32, *state, *ref );
        std::cout << "FD vs Fourier: N=" << n << " RMS=" << rms;
        if ( prev > 0 )
        {
            const double p = std::log2( prev / rms );
            std::cout << " order=" << p;
            check( p > 1.5 && p < 2.5, "FD/Fourier spatial refinement failed" );
        }
        std::cout << '\n';
        prev = rms;
    }
    std::cout << "Fourier reference refinement: spatial RMS=" << spatial << " temporal RMS=" << temporal << '\n';
    check( std::max( spatial, temporal ) < .1 * prev, "Fourier reference not sufficiently resolved" );
}
}
#endif
