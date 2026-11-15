#include "implicit_projection_problem.h"
#include "projection_transfer_guard.h"
#include <nonlinear_operators/stokes/periodic_stokes.h>

using namespace projection_tests;
namespace stokes = nonlinear_operators::stokes;

// On [0,2*pi]^2, u=(d_y psi,-d_x psi), with
// psi=exp(-t)*cos(x)*cos(y)+.12*exp(-.7*t)*cos(3*x)*cos(2*y).
// Add a constant velocity (.125,-.25). The manufactured pressure is
// p=(1+.2*sin(t))*cos(x)*cos(y), with mean-zero gauge. The forcing is
// u_t-nu*Delta(u)+grad(p), nu=.15. This is an exact unsteady periodic Stokes
// solution with two spatial eigenvalues and genuinely non-autonomous forcing.
// Coefficients are normalized Fourier series values, not unnormalized FFT data.
struct manufactured_forcing
{
    double                     viscosity = .15;
    bool                       pressure  = true;
    __DEVICE_TAG__ static void velocity( double x, double y, double t, double *out )
    {
        double amplitude = 0;
        if ( ( x == 1 || x == -1 ) && y == 1 )
            amplitude = .25 * exp( -t );
        if ( ( x == 3 || x == -3 ) && y == 2 )
            amplitude = .03 * exp( -.7 * t );
        out[0] = out[2] = 0;
        out[1]          = y * amplitude;
        out[3]          = -x * amplitude;
        if ( x == 0 && y == 0 )
        {
            out[0] = .125;
            out[2] = -.25;
        }
    }
    __DEVICE_TAG__ void operator()( double x, double y, double t, double *out ) const
    {
        velocity( x, y, t, out );
        const double decay      = y == 2 ? .7 : 1;
        const double multiplier = viscosity * ( x * x + y * y ) - decay;
        for ( int j = 0; j < 4; ++j )
            out[j] *= multiplier;
        if ( x == 0 && y == 0 )
            out[0] = out[2] = 0;
        if ( pressure && ( x == 1 || x == -1 ) && y == 1 )
        {
            const double p = .25 * ( 1 + .2 * sin( t ) );
            out[1] += x * p;
            out[3] += y * p;
        }
    }
};

template <class O>
using modes_type = typename stokes::periodic_projector<O>::modes_type;
template <class O>
std::shared_ptr<modes_type<O>> make_modes( std::size_t n )
{
    using namespace discretization::fourier;
    discretization::common::structured_extent<2> extent( n, n );
    return std::make_shared<modes_type<O>>(
        periodic_grid<double, 2>( extent, { 2 * std::acos( -1. ), 2 * std::acos( -1. ) } ), r2c_index_space_2d( extent )
    );
}
template <class O>
struct stokes_exact_mapping
{
    typename O::vector_type out;
    const double           *kx, *ky;
    double                  time;
    __DEVICE_TAG__ void     operator()( std::ptrdiff_t mode ) const
    {
        double value[4];
        manufactured_forcing::velocity( kx[mode], ky[mode], time, value );
        for ( int j = 0; j < 4; ++j )
            out( 4 * mode + j ) = value[j];
    }
};
template <class O>
void exact_state( const modes_type<O> &modes, typename O::vector_type &out, double time )
{
    typename O::for_each_type each;
    each(
        stokes_exact_mapping<O>{ out, modes.kx().data(), modes.ky().data(), time }, modes.index_space().complex_size()
    );
    each.wait();
}

template <class O, bool Dense, bool Adaptive>
double integrate( std::shared_ptr<O> ops, std::shared_ptr<modes_type<O>> modes, const char *method, double dt )
{
    using P = stokes::periodic_stokes<O, manufactured_forcing>;
    using C = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S = runge_kutta::implicit_time_step<
        O, P, C, typename implicit_newton<O, P>::solver, Dense, stokes::periodic_projector<O>>;
    using R        = runge_kutta::implicit_time_step<O, P, C, typename implicit_newton<O, P>::solver, Dense>;
    auto p         = std::make_shared<P>( ops, modes, .15 );
    auto reduced   = std::make_shared<P>( ops, modes, .15, manufactured_forcing{ .15, false } );
    auto projector = std::make_shared<stokes::periodic_projector<O>>( ops, modes );
    auto control   = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step = cp.maximum_step = dt;
            cp.relative_tolerance             = 1e-6;
            cp.absolute_tolerance             = 1e-9;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ dt } );
    };
    S         step( ops, p, control(), implicit_newton<O, P>::make( ops ), typename S::params{ method }, projector );
    R         reference( ops, reduced, control(), implicit_newton<O, P>::make( ops ), typename R::params{ method } );
    buffer<O> state( *ops ), ref_state( *ops ), out( *ops ), ref_out( *ops ), expected( *ops ), sample( *ops ),
        ref_sample( *ops ), scratch( *ops );
    double       time   = .3;
    const double target = .5;
    exact_state<O>( *modes, *state, time );
    ops->assign( *state, *ref_state );
    std::size_t warmed_allocations = 0;
    for ( int attempt = 0; time < target && attempt < 1000; ++attempt )
    {
        step.set_time( time );
        step.set_target_time( target );
        reference.set_time( time );
        reference.set_target_time( target );
        step.apply( *state, *out );
        reference.apply( *ref_state, *ref_out );
        check( step.get_status() == single_step_status::converged, "Manufactured Stokes DIRK failed" );
        check( reference.get_status() == single_step_status::converged, "Reduced Stokes DIRK failed" );
        check( std::abs( step.get_dt() - reference.get_dt() ) < 2e-10, "Stokes projected embedded estimate mismatch" );
        check(
            difference( *ops, *out, *ref_out, *scratch ) < 2e-10,
            "Stokes stage projection differs from pressure elimination"
        );
        projector->divergence( *out, *scratch );
        check( ops->norm( *scratch ) < 2e-12, "Stokes endpoint divergence" );
        const double endpoint = std::min( target, time + step.get_dt() );
        if constexpr ( Dense )
        {
            auto view     = step.get_continuous_integration();
            auto ref_view = reference.get_continuous_integration();
            for ( double theta : { 0., .25, .6, 1. } )
            {
                const double sampled_time = view.evaluate( theta, *sample );
                ref_view.evaluate( theta, *ref_sample );
                check(
                    difference( *ops, *sample, *ref_sample, *scratch ) < 3e-10,
                    "Stokes dense pressure elimination mismatch"
                );
                projector->divergence( *sample, *scratch );
                check( ops->norm( *scratch ) < 2e-12, "Stokes dense divergence" );
                exact_state<O>( *modes, *expected, sampled_time );
                check(
                    difference( *ops, *sample, *expected, *scratch ) < .006, "Stokes interior manufactured accuracy"
                );
            }
        }
        step.finalize( adaptation_status::accepted, endpoint, *out );
        reference.finalize( adaptation_status::accepted, std::min( target, time + reference.get_dt() ), *ref_out );
        ops->assign( *out, *state );
        ops->assign( *ref_out, *ref_state );
        time = endpoint;
        if constexpr ( std::is_same_v<typename O::backend_type, guarded_backend> )
        {
            if ( attempt == 0 )
                warmed_allocations = guarded_memory::allocation_count;
            check(
                guarded_memory::allocation_count == warmed_allocations, "Stokes allocated backend data after warmup"
            );
        }
    }
    check( time == target, "Stokes integration exhausted its budget" );
    exact_state<O>( *modes, *expected, time );
    return difference( *ops, *state, *expected, *scratch );
}

template <class O>
struct gradient_mapping
{
    typename O::vector_type out;
    const double           *kx, *ky;
    __DEVICE_TAG__ void     operator()( std::ptrdiff_t mode ) const
    {
        const double x = kx[mode], y = ky[mode], p = 1 / ( 1 + x * x + y * y );
        out( 4 * mode ) = out( 4 * mode + 2 ) = 0;
        out( 4 * mode + 1 )                   = x * p;
        out( 4 * mode + 3 )                   = y * p;
    }
};
template <class O>
void projection_checks( std::shared_ptr<O> ops, std::shared_ptr<modes_type<O>> modes )
{
    stokes::periodic_projector<O> project( ops, modes );
    buffer<O>                     state( *ops ), expected( *ops ), scratch( *ops );
    exact_state<O>( *modes, *state, .3 );
    ops->assign( *state, *expected );
    project.apply( *state );
    check(
        difference( *ops, *state, *expected, *scratch ) < 1e-14, "Stokes projector changed solenoidal velocity or mean"
    );
    typename O::for_each_type each;
    each( gradient_mapping<O>{ *state, modes->kx().data(), modes->ky().data() }, modes->index_space().complex_size() );
    each.wait();
    project.apply( *state );
    check( ops->norm( *state ) < 2e-13, "Periodic Poisson solve did not remove gradient forcing" );
    ops->assign_scalar( 1., *state );
    project.apply( *state );
    ops->assign( *state, *expected );
    project.apply( *state );
    check( difference( *ops, *state, *expected, *scratch ) < 2e-13, "Stokes projector not idempotent" );
    project.divergence( *state, *scratch );
    check( ops->norm( *scratch ) < 3e-11, "All-mode Stokes projection divergence" );
}

template <class O>
void jacobian_checks( std::shared_ptr<O> ops, std::shared_ptr<modes_type<O>> modes )
{
    using P = stokes::periodic_stokes<O, manufactured_forcing>;
    P         p( ops, modes, .15 );
    buffer<O> state( *ops ), direction( *ops ), plus( *ops ), minus( *ops ), residual( *ops ), jacobian( *ops ),
        scratch( *ops );
    exact_state<O>( *modes, *state, .3 );
    ops->assign_scalar( .2, p.previous_state() );
    ops->assign_scalar( .1, p.known_rhs() );
    ops->assign_scalar( .7, *direction );
    p.set_time( .4 );
    p.set_affine_shift( .7, .2 );
    p.set_linearization_point( *state );
    auto frozen = p.get_jacobi_operator();
    frozen->apply( *direction, *jacobian );
    ops->assign_lin_comb( 1., *state, 1e-5, *direction, *plus );
    ops->assign_lin_comb( 1., *state, -1e-5, *direction, *minus );
    p.apply( *plus, *residual );
    p.apply( *minus, *scratch );
    ops->assign_lin_comb( 5e4, *residual, -5e4, *scratch, *plus );
    check( difference( *ops, *plus, *jacobian, *scratch ) < 1e-8, "Stokes shifted Jacobian finite difference" );
    p.evaluate_rate( *state, *plus );
    p.set_time( .4 );
    p.set_affine_shift( 13., .9 );
    p.evaluate_rate( *state, *minus );
    check( difference( *ops, *plus, *minus, *scratch ) == 0, "Stokes physical rate depends on affine shift" );
    frozen->apply( *direction, *residual );
    check( difference( *ops, *residual, *jacobian, *scratch ) == 0, "Stokes rate changed prepared Jacobian" );
}

int main()
try
{
    backend_type::init_device();
    {
        auto modes = make_modes<space_type>( 12 );
        auto ops   = std::make_shared<space_type>( 4 * modes->index_space().complex_size() );
        projection_checks( ops, modes );
        jacobian_checks( ops, modes );
        for ( const char *method : { "IE", "IM", "CN", "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
        {
            const double coarse = integrate<space_type, true, false>( ops, modes, method, .04 );
            const double fine   = integrate<space_type, true, false>( ops, modes, method, .02 );
            const auto   table  = runge_kutta::make_butcher_table( method );
            check( coarse / fine > std::pow( 2., table.order() - .4 ), "Stokes temporal order regression" );
            std::cout << method << " Stokes error " << fine << ", refinement " << coarse / fine << '\n';
            if ( table.is_embedded() )
                integrate<space_type, true, true>( ops, modes, method, .04 );
        }
    }
    // Constructor wave-number uploads precede the guard. The complete large
    // solve, including manufactured initialization/verification, stays resident.
    // A tagged backend copy registry permits device copies and scalar reductions.
    {
        using O                         = counted_space<guarded_backend>;
        guarded_memory::check_transfers = false;
        auto modes                      = make_modes<O>( 128 );
        guarded_memory::check_transfers = true;
        auto ops                        = std::make_shared<O>( 4 * modes->index_space().complex_size() );
        projection_checks( ops, modes );
        const double error = integrate<O, true, true>( ops, modes, "ARK324L2SA", .02 );
        check( error < 2e-7, "Large manufactured Stokes accuracy" );
        std::cout << ops->size() << " Stokes unknowns, guarded host vector transfers: zero\n";
    }
    check( guarded_memory::allocations.empty(), "Stokes backend workspace leaked" );
    std::cout << checks << " periodic Stokes projection checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
