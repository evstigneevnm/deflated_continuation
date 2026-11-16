#include "implicit_imex_projection_problem.h"
#include "projection_transfer_guard.h"
#include "manufactured_flow.h"
#include <nonlinear_operators/stokes/periodic_stokes_imex.h>

using namespace projection_tests;
namespace stokes = nonlinear_operators::stokes;
using namespace stokes_validation;
template <class Space, bool Dense, bool Adaptive>
double integrate(
    std::shared_ptr<Space> ops, std::shared_ptr<modes_type<Space>> modes, const char *method, double dt,
    manufactured_flow flow
)
{
    using P = stokes::periodic_stokes_imex<Space, manufactured_flow>;
    using I = typename P::implicit_type;
    using C = std::conditional_t<Adaptive, adaptive_control<Space>, fixed_control<Space>>;
    using S = runge_kutta::imex_time_step<
        Space, P, C, typename implicit_newton<Space, I>::solver, Dense, stokes::periodic_projector<Space>>;
    using R               = runge_kutta::imex_time_step<Space, P, C, typename implicit_newton<Space, I>::solver, Dense>;
    auto reduced_flow     = flow;
    reduced_flow.pressure = false;
    auto p                = std::make_shared<P>( ops, modes, .15, flow, std::array<double, 2>{ flow.cx, flow.cy } );
    auto reduced   = std::make_shared<P>( ops, modes, .15, reduced_flow, std::array<double, 2>{ flow.cx, flow.cy } );
    auto projector = std::make_shared<stokes::periodic_projector<Space>>( ops, modes );
    auto control   = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step = cp.maximum_step = std::abs( dt );
            cp.relative_tolerance             = 1e-6;
            cp.absolute_tolerance             = 1e-9;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ std::abs( dt ) } );
    };
    S step( ops, p, control(), implicit_newton<Space, I>::make( ops ), typename S::params{ method }, projector );
    R reference( ops, reduced, control(), implicit_newton<Space, I>::make( ops ), typename R::params{ method } );
    buffer<Space> state( *ops ), ref_state( *ops ), out( *ops ), ref_out( *ops ), expected( *ops ), sample( *ops ),
        ref_sample( *ops ), scratch( *ops );
    double       time   = dt > 0 ? .3 : .5;
    const double target = dt > 0 ? .5 : .3;
    exact_state<Space>( *modes, *state, time, flow );
    ops->assign( *state, *ref_state );
    std::size_t warmed_allocations = 0;
    for ( int attempt = 0; time != target && attempt < 1000; ++attempt )
    {
        step.set_time( time );
        step.set_target_time( target );
        reference.set_time( time );
        reference.set_target_time( target );
        step.apply( *state, *out );
        reference.apply( *ref_state, *ref_out );
        check( step.get_status() == single_step_status::converged, "Manufactured Stokes/Oseen IMEX failed" );
        check( reference.get_status() == single_step_status::converged, "Pressure-eliminated IMEX failed" );
        check( std::abs( step.get_dt() - reference.get_dt() ) < 2e-10, "Projected PDE combined estimator differs" );
        check(
            difference( *ops, *out, *ref_out, *scratch ) < 3e-10,
            "IMEX projection differs from analytical pressure elimination"
        );
        projector->divergence( *out, *scratch );
        check( ops->norm( *scratch ) < 2e-12, "Stokes/Oseen endpoint divergence" );
        const auto endpoint     = step.get_dt() == target - time ? target : time + step.get_dt();
        const auto ref_endpoint = reference.get_dt() == target - time ? target : time + reference.get_dt();
        if constexpr ( Dense )
        {
            auto view = step.get_continuous_integration();
            auto ref  = reference.get_continuous_integration();
            for ( double theta : { 0., .25, .6, 1. } )
            {
                const auto sampled_time = view.evaluate( theta, *sample );
                ref.evaluate( theta, *ref_sample );
                check(
                    difference( *ops, *sample, *ref_sample, *scratch ) < 4e-10,
                    "Projected PDE dense pressure elimination mismatch"
                );
                projector->divergence( *sample, *scratch );
                check( ops->norm( *scratch ) < 2e-12, "Stokes/Oseen dense divergence" );
                exact_state<Space>( *modes, *expected, sampled_time, flow );
                check(
                    difference( *ops, *sample, *expected, *scratch ) < 2e-5, "Manufactured Oseen interior accuracy"
                );
            }
        }
        step.finalize( adaptation_status::accepted, endpoint, *out );
        reference.finalize( adaptation_status::accepted, ref_endpoint, *ref_out );
        ops->assign( *out, *state );
        ops->assign( *ref_out, *ref_state );
        time = endpoint;
        if constexpr ( std::is_same_v<typename Space::backend_type, guarded_backend> )
        {
            if ( attempt == 0 )
                warmed_allocations = guarded_memory::allocation_count;
            check( warmed_allocations == guarded_memory::allocation_count, "IMEX PDE allocated fields after warmup" );
        }
    }
    check( time == target, "Stokes/Oseen integration exhausted its budget" );
    exact_state<Space>( *modes, *expected, time, flow );
    return difference( *ops, *state, *expected, *scratch );
}

int main()
try
{
    backend_type::init_device();
    {
        auto modes = make_modes<space_type>( 12 );
        auto ops   = std::make_shared<space_type>( 4 * modes->index_space().complex_size() );
        for ( auto flow : { manufactured_flow{}, manufactured_flow{ .7, -.3 } } )
        {
            for ( const char *method :
                  { "IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS233", "IMEX_ARS222", "IMEX_ARK324L2SA" } )
            {
                const double coarse = integrate<space_type, false, false>( ops, modes, method, .04, flow );
                const double fine   = integrate<space_type, false, false>( ops, modes, method, .02, flow );
                const auto   table  = runge_kutta::make_imex_butcher_table( method );
                check(
                    coarse / fine > std::pow( 2., table.order - .4 ), "Manufactured Stokes/Oseen IMEX order regression"
                );
                std::cout << ( flow.cx == 0 ? "Stokes " : "Oseen " ) << method << " error " << fine << ", refinement "
                          << coarse / fine << '\n';
            }
            integrate<space_type, false, false>( ops, modes, "IMEX_ARK324L2SA", -.02, flow );
            integrate<space_type, true, false>( ops, modes, "IMEX_ARK324L2SA", .02, flow );
            integrate<space_type, true, false>( ops, modes, "IMEX_ARK324L2SA", -.02, flow );
            integrate<space_type, true, true>( ops, modes, "IMEX_ARK324L2SA", .04, flow );
        }
    }
    {
        using Space                     = counted_space<guarded_backend>;
        guarded_memory::check_transfers = false;
        auto modes                      = make_modes<Space>( 128 );
        guarded_memory::check_transfers = true;
        auto       ops                  = std::make_shared<Space>( 4 * modes->index_space().complex_size() );
        const auto error = integrate<Space, true, true>( ops, modes, "IMEX_ARK324L2SA", .01, { .7, -.3 } );
        check( error < 3e-7, "Large manufactured Oseen accuracy" );
        std::cout << ops->size() << " Oseen components, guarded host vector transfers: zero\n";
    }
    check( guarded_memory::allocations.empty(), "IMEX PDE workspace leaked" );
    std::cout << checks << " periodic Stokes/Oseen IMEX projection checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
