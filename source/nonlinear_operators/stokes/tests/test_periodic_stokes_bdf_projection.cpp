#include "implicit_bdf_projection_test.h"
#include "projection_transfer_guard.h"
#include "manufactured_flow.h"
#include <nonlinear_operators/stokes/periodic_stokes.h>

using namespace projection_tests;
using namespace stokes_validation;
namespace stokes = nonlinear_operators::stokes;

// Periodic divergence-free manufactured velocity with two spatial harmonics,
// preserved mean flow and an independently prescribed pressure gradient. BDF
// solves the full diffusion/advection equation, then removes its normal part.
template <class Space, bool Adaptive, bool Dense>
double integrate(
    std::shared_ptr<Space> ops, std::shared_ptr<modes_type<Space>> modes, const char *method, double h,
    manufactured_flow flow, double tolerance = 1e-6
)
{
    using P = stokes::periodic_stokes<Space, manufactured_flow>;
    using C = std::conditional_t<Adaptive, adaptive_control<Space>, fixed_control<Space>>;
    using S = bdf::implicit_time_step<
        Space, P, C, typename implicit_newton<Space, P>::solver, Dense, stokes::periodic_projector<Space>>;
    auto p          = std::make_shared<P>( ops, modes, .15, flow, std::array<double, 2>{ flow.cx, flow.cy } );
    auto projector  = std::make_shared<stokes::periodic_projector<Space>>( ops, modes );
    auto controller = [&] {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step       = std::abs( h );
            cp.maximum_step       = .04;
            cp.relative_tolerance = tolerance;
            cp.absolute_tolerance = tolerance * .01;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ std::abs( h ) } );
    }();
    typename S::params sp{ method };
    sp.adaptive = Adaptive;
    S             step( ops, p, controller, implicit_newton<Space, P>::make( ops ), sp, projector );
    buffer<Space> state( *ops ), out( *ops ), reference( *ops ), sample( *ops ), scratch( *ops );
    double        time = h > 0 ? .3 : .5, end = h > 0 ? .5 : .3;
    exact_state<Space>( *modes, *state, time, flow );
    std::size_t  allocations = 0, count = 0;
    unsigned int highest = 0;
    while ( time != end && count < 5000 )
    {
        step.set_time( time );
        step.set_target_time( end );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Stokes/Oseen BDF solve failed" );
        highest = std::max( highest, step.get_order() );
        projector->divergence( *out, *scratch );
        check( ops->norm( *scratch ) < 3e-12, "BDF endpoint divergence" );
        if constexpr ( Dense )
        {
            auto view = step.get_continuous_integration();
            for ( double theta : { 0., .3, .7, 1. } )
            {
                const double t = view.evaluate( theta, *sample );
                projector->divergence( *sample, *scratch );
                check( ops->norm( *scratch ) < 3e-12, "BDF dense divergence" );
                exact_state<Space>( *modes, *reference, t, flow );
                check( difference( *ops, *sample, *reference, *scratch ) < .01, "BDF manufactured interior error" );
            }
        }
        time = step.get_dt() == end - time ? end : time + step.get_dt();
        step.finalize( adaptation_status::accepted, time, *out );
        ops->assign( *out, *state );
        if constexpr ( std::is_same_v<typename Space::backend_type, guarded_backend> )
        {
            if ( count == 0 )
                allocations = guarded_memory::allocation_count;
            check( allocations == guarded_memory::allocation_count, "BDF PDE allocated vectors after warmup" );
        }
        ++count;
    }
    check( time == end, "BDF PDE step budget exhausted" );
    if ( std::string( method ) == "BDF2" )
        check( highest == 2, "Projected PDE BDF2 kept restarting" );
    exact_state<Space>( *modes, *reference, time, flow );
    return difference( *ops, *state, *reference, *scratch );
}

// Check the new unsplit Oseen Jacobian at nonzero time and general affine
// coefficients. No mass, transport or forcing term is inferred by the solver.
void jacobian()
{
    using Space = space_type;
    using P     = stokes::periodic_stokes<Space, manufactured_flow>;
    auto modes  = make_modes<Space>( 12 );
    auto ops    = std::make_shared<Space>( 4 * modes->index_space().complex_size() );
    auto p = std::make_shared<P>( ops, modes, .15, manufactured_flow{ .7, -.3 }, std::array<double, 2>{ .7, -.3 } );
    buffer<Space> state( *ops ), direction( *ops ), trial( *ops ), base( *ops ), value( *ops ), action( *ops ),
        scratch( *ops );
    exact_state<Space>( *modes, *state, .3, { .7, -.3 } );
    exact_state<Space>( *modes, *direction, .7, {} );
    ops->assign( *state, p->previous_state() );
    ops->assign_scalar( .02, p->known_rhs() );
    p->set_time( .41 );
    p->set_affine_shift( .4, .8 );
    p->apply( *state, *base );
    p->set_linearization_point( *state );
    p->get_jacobi_operator()->apply( *direction, *action );
    ops->assign_mul( 1., *state, 1e-5, *direction, *trial );
    p->apply( *trial, *value );
    ops->add_mul( -1., *base, 1., *value );
    ops->scale( 1e5, *value );
    check(
        difference( *ops, *value, *action, *scratch ) < 1e-8, "Unsplit Oseen Jacobian differs from residual derivative"
    );
}

int main()
try
{
    backend_type::init_device();
    jacobian();
    {
        auto modes = make_modes<space_type>( 12 );
        auto ops   = std::make_shared<space_type>( 4 * modes->index_space().complex_size() );
        for ( auto flow : { manufactured_flow{}, manufactured_flow{ .7, -.3 } } )
        {
            for ( const char *method : { "BDF1", "BDF2" } )
            {
                double coarse = integrate<space_type, false, false>( ops, modes, method, .04, flow );
                double fine   = integrate<space_type, false, true>( ops, modes, method, .02, flow );
                check(
                    coarse / fine > std::pow( 2., std::string( method ) == "BDF1" ? .65 : 1.6 ),
                    "BDF PDE order regression"
                );
                integrate<space_type, false, true>( ops, modes, method, -.02, flow );
            }
            double coarse = integrate<space_type, true, true>( ops, modes, "BDF", .02, flow, 1e-4 );
            double fine   = integrate<space_type, true, true>( ops, modes, "BDF", .02, flow, 1e-7 );
            check( fine < coarse * .5, "Adaptive BDF PDE tolerance refinement failed" );
            integrate<space_type, true, true>( ops, modes, "BDF", -.02, flow );
        }
    }
    {
        using Space                     = counted_space<guarded_backend>;
        guarded_memory::check_transfers = false;
        auto modes                      = make_modes<Space>( 128 );
        guarded_memory::check_transfers = true;
        auto ops                        = std::make_shared<Space>( 4 * modes->index_space().complex_size() );
        // The generic controller weights every storage slot, including inactive
        // Fourier modes. Tighten its local tolerance for this sparse large grid;
        // physical error is still measured independently, without N dilution.
        check(
            integrate<Space, true, true>( ops, modes, "BDF", .01, { .7, -.3 }, 1e-9 ) < 3e-5, "Large BDF Oseen accuracy"
        );
        std::cout << ops->size() << " BDF Oseen components; guarded vector transfers: zero\n";
    }
    check( guarded_memory::allocations.empty(), "BDF PDE storage leaked" );
    std::cout << checks << " periodic Stokes/Oseen BDF projection checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
