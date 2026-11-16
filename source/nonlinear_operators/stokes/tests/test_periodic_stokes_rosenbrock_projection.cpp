#include "rosenbrock_validation_problem.h"
#include "projection_transfer_guard.h"
using namespace projection_tests;
using namespace stokes_validation;

// Two resolved Fourier harmonics, time-dependent pressure-gradient forcing,
// preserved mean velocity and optional constant transport. Compare projected
// increments with analytical pressure elimination, not a second projector.
template <class Space, bool Adaptive, bool Dense>
double integrate(
    std::shared_ptr<Space> ops, std::shared_ptr<modes_type<Space>> modes, const char *method, double h,
    manufactured_flow flow, double factor = 1, double tolerance = 1e-6
)
{
    using P               = rosenbrock_problem<Space>;
    using L               = projection_linear_solver<Space, P>;
    using C               = std::conditional_t<Adaptive, adaptive_control<Space>, rosenbrock_recording_control<Space>>;
    using S               = rosenbrock::time_step<Space, P, C, L, Dense, stokes::periodic_projector<Space>>;
    using R               = rosenbrock::time_step<Space, P, C, L, Dense>;
    auto reduced_flow     = flow;
    reduced_flow.pressure = false;
    auto p                = std::make_shared<P>( ops, modes, flow, factor ),
         r                = std::make_shared<P>( ops, modes, reduced_flow, factor );
    auto controller       = [&] {
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
            return std::make_shared<C>( ops, std::abs( h ) );
    };
    auto               c = controller(), rc = controller();
    typename S::params sp{ method };
    sp.autonomous = false;
    typename R::params rp{ method };
    rp.autonomous           = false;
    auto          projector = std::make_shared<stokes::periodic_projector<Space>>( ops, modes );
    S             step( ops, p, c, std::make_shared<L>( ops ), sp, projector );
    R             reference( ops, r, rc, std::make_shared<L>( ops ), rp );
    buffer<Space> state( *ops ), ref( *ops ), out( *ops ), expected( *ops ), sample( *ops ), rsample( *ops ),
        scratch( *ops ), analytic( *ops );
    double time = h > 0 ? .3 : .5, end = h > 0 ? .5 : .3;
    exact_state<Space>( *modes, *state, time, flow );
    ops->assign( *state, *ref );
    std::size_t count = 0, allocations = 0;
    while ( time != end && count < 5000 )
    {
        step.set_time( time );
        step.set_target_time( end );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Stokes/Oseen Rosenbrock failed" );
        if constexpr ( !Adaptive )
        {
            reference.set_time( time );
            reference.set_target_time( end );
            reference.apply( *ref, *expected );
            check( reference.get_status() == single_step_status::converged, "Pressure-eliminated Rosenbrock failed" );
            check( difference( *ops, *out, *expected, *scratch ) < 4e-11, "Rosenbrock pressure elimination mismatch" );
            check(
                difference( *ops, *c->error, *rc->error, *scratch ) < 4e-11, "Projected PDE embedded error mismatch"
            );
        }
        projector->divergence( *out, *scratch );
        check( ops->norm( *scratch ) < 3e-12, "Rosenbrock PDE endpoint divergence" );
        if constexpr ( Dense )
        {
            auto view = step.get_continuous_integration();
            for ( double theta : { 0., .25, .6, 1. } )
            {
                const double t = view.evaluate( theta, *sample );
                if constexpr ( !Adaptive )
                {
                    reference.get_continuous_integration().evaluate( theta, *rsample );
                    check(
                        difference( *ops, *sample, *rsample, *scratch ) < 5e-11,
                        "Projected PDE dense pressure elimination mismatch"
                    );
                }
                projector->divergence( *sample, *scratch );
                check( ops->norm( *scratch ) < 3e-12, "Rosenbrock PDE dense divergence" );
                exact_state<Space>( *modes, *analytic, t, flow );
                check( difference( *ops, *sample, *analytic, *scratch ) < .001, "Rosenbrock PDE interior accuracy" );
            }
        }
        time = step.get_dt() == end - time ? end : time + step.get_dt();
        step.finalize( adaptation_status::accepted, time, *out );
        ops->assign( *out, *state );
        if constexpr ( !Adaptive )
        {
            reference.finalize( adaptation_status::accepted, time, *expected );
            ops->assign( *expected, *ref );
        }
        if constexpr ( std::is_same_v<typename Space::backend_type, guarded_backend> )
        {
            if ( count == 0 )
                allocations = guarded_memory::allocation_count;
            check( allocations == guarded_memory::allocation_count, "Rosenbrock PDE allocated fields after warmup" );
        }
        ++count;
    }
    check( time == end, "Rosenbrock PDE step budget exhausted" );
    exact_state<Space>( *modes, *analytic, time, flow );
    return difference( *ops, *state, *analytic, *scratch );
}

struct no_derivative_forcing
{
    __DEVICE_TAG__ void operator()( double, double, double, double *out ) const
    {
        for ( int i = 0; i < 4; ++i )
            out[i] = 0;
    }
};
void partial_time_derivative()
{
    using Space = space_type;
    using Plain = stokes::periodic_stokes<Space, no_derivative_forcing>;
    static_assert( !nmfd::time_steppers::detail::has_time_derivative_evaluation<Plain, Space::vector_type>::value );
    auto          modes = make_modes<Space>( 12 );
    auto          ops   = std::make_shared<Space>( 4 * modes->index_space().complex_size() );
    buffer<Space> state( *ops ), left( *ops ), right( *ops ), derivative( *ops ), scratch( *ops );
    for ( auto flow : { manufactured_flow{}, manufactured_flow{ .7, -.3 } } )
    {
        rosenbrock_problem<Space> p( ops, modes, flow );
        exact_state<Space>( *modes, *state, .3, flow );
        for ( double t : { .3, .71, 1.2 } )
        {
            p.set_time( t );
            p.set_affine_shift( .4, .8 );
            p.set_linearization_point( *state );
            check( p.evaluate_time_derivative( *state, *derivative ), "Missing analytic forcing time derivative" );
            const auto prepared = *p.get_jacobi_operator();
            p.set_time( t - 1e-5 );
            p.evaluate_rate( *state, *left );
            p.set_time( t + 1e-5 );
            p.evaluate_rate( *state, *right );
            ops->add_mul( -1., *left, 1., *right );
            ops->scale( 5e4, *right );
            check(
                difference( *ops, *right, *derivative, *scratch ) < 1e-9,
                "Forcing derivative is not the partial derivative at fixed state"
            );
            check(
                p.get_jacobi_operator()->a == prepared.a && p.get_jacobi_operator()->b == prepared.b,
                "Rate/time derivative changed frozen operator"
            );
        }
    }
}
int main()
try
{
    backend_type::init_device();
    partial_time_derivative();
    {
        auto modes = make_modes<space_type>( 12 );
        auto ops   = std::make_shared<space_type>( 4 * modes->index_space().complex_size() );
        for ( auto flow : { manufactured_flow{}, manufactured_flow{ .7, -.3 } } )
        {
            for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
            {
                const double order = std::string( method ) == "ROS2" ? 2 : 3;
                for ( double factor : { 1., .5 } )
                {
                    if ( factor != 1 && std::string( method ) != "ROS34PW2" )
                        continue;
                    const double coarse = integrate<space_type, false, false>( ops, modes, method, .04, flow, factor );
                    const double fine   = integrate<space_type, false, true>( ops, modes, method, .02, flow, factor );
                    check( coarse / fine > std::pow( 2., order - .4 ), "Manufactured projected Rosenbrock PDE order" );
                    integrate<space_type, false, true>( ops, modes, method, -.02, flow, factor );
                    const double a = integrate<space_type, true, true>( ops, modes, method, .04, flow, factor, 1e-4 );
                    const double b = integrate<space_type, true, true>( ops, modes, method, .04, flow, factor, 1e-8 );
                    check( b < a * .5, "Rosenbrock PDE tolerance refinement failed" );
                }
            }
        }
    }
    {
        using Space                     = counted_space<guarded_backend>;
        guarded_memory::check_transfers = false;
        auto modes                      = make_modes<Space>( 128 );
        guarded_memory::check_transfers = true;
        auto ops                        = std::make_shared<Space>( 4 * modes->index_space().complex_size() );
        check(
            integrate<Space, true, true>( ops, modes, "ROS34PW2", .01, { .7, -.3 }, .5, 1e-9 ) < 3e-6,
            "Large Rosenbrock Oseen accuracy"
        );
        std::cout << ops->size() << " Oseen components; guarded vector transfers: zero\n";
    }
    check( guarded_memory::allocations.empty(), "Rosenbrock PDE storage leaked" );
    std::cout << checks << " periodic Stokes/Oseen Rosenbrock projection checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
