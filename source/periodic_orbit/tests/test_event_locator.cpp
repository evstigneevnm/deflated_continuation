#include "section_test_support.h"
using namespace section_tests;

// Exact manufactured dense trajectory u(t)=t. It isolates scalar root
// refinement from integration error and exercises bounded diagnostic exits.
struct linear_dense
{
    std::shared_ptr<operations> ops;
    double                      start = 0, dt = 1;
    bool                        invalid = false;
    double                      evaluate( double theta, X &state ) const
    {
        const double time = start + theta * dt;
        ops->assign_scalar( invalid ? std::numeric_limits<double>::infinity() : time, state );
        return time;
    }
};
struct diagnostic_policy : policy
{
    using policy::policy;
    bool fail = false;
    bool reconstruct( double t, X &state )
    {
        return !fail && policy::reconstruct( t, state );
    }
};
struct diagnostic_event
{
    plane::evaluator underlying;
    bool             invalid = false;
    double           value( double t, const X &state )
    {
        return invalid ? std::numeric_limits<double>::quiet_NaN() : underlying.value( t, state );
    }
};

struct counted_space : operations
{
    using operations::operations;
    mutable std::size_t finite_checks = 0;
    bool is_valid_number( const X &state ) const
    {
        ++finite_checks;
        return operations::is_valid_number( state );
    }
};

struct dense_step
{
    linear_dense dense;
    const linear_dense &get_continuous_integration() const { return dense; }
};

using counted_identity = periodic_orbit::identity_state_policy<counted_space>;
struct corrupting_policy : counted_identity
{
    std::shared_ptr<counted_space> space;
    unsigned calls = 0;
    explicit corrupting_policy( std::shared_ptr<counted_space> s ) : counted_identity( s ), space( s ) {}
    bool reconstruct( double, X &state )
    {
        ++calls;
        space->assign_scalar( std::numeric_limits<double>::infinity(), state );
        return true;
    }
};

void test_sample_validation()
{
    auto ops = std::make_shared<counted_space>( 2 );
    auto identity = std::make_shared<counted_identity>( ops );
    auto stepper = std::make_shared<dense_step>( dense_step{ linear_dense{ ops } } );
    auto out = buffer( ops );
    fill( *ops, *out, 7, 8 );
    periodic_orbit::event_params<double> params;
    params.subdivisions = 4;
    periodic_orbit::section_manager<counted_space, dense_step, plane> manager( ops, stepper, identity, params );
    periodic_orbit::flight_request<double, plane> request{ 0., 1., make_plane( ops, 2, 2, 1, 0 ) };
    auto scan = [&]( auto &scanner ) {
        scanner.reset( request );
        double end = request.end_time;
        scanner.set_time_interval( request.start_time, end );
        ops->finite_checks = 0;
        require( !scanner.apply( *out, *out ) && end == request.end_time, "Unexpected step modification" );
    };
    scan( manager );
    require( manager.get_status() == decltype(manager)::integration_status::running &&
             ops->finite_checks == params.subdivisions + 1, "Identity scan must check each dense sample once" );
    stepper->dense.invalid = true;
    scan( manager );
    require( manager.report().status == status::invalid_state && ops->finite_checks == 1, "Invalid identity sample" );
    stepper->dense.invalid = false;
    auto custom = std::make_shared<corrupting_policy>( ops );
    periodic_orbit::section_manager<counted_space, dense_step, plane, corrupting_policy>
        custom_manager( ops, stepper, custom, params );
    scan( custom_manager );
    require( custom_manager.report().status == status::invalid_state && custom->calls == 1 && ops->finite_checks == 2,
             "Scanner must validate the reconstructed state even for an identity-derived policy" );
    near_state( ops, *out, 7, 8, 1e-15 );

    auto target = make_plane( ops, .25, .25, 1, 0 );
    plane::evaluation_workspace work( ops );
    auto event = target->bind( work );
    periodic_orbit::event_locator<counted_space> locator( ops );
    periodic_orbit::event_bracket<double> bracket{ 0, 1, 0, 1, -.25, .75 };
    ops->finite_checks = 0;
    const auto result = locator.locate( stepper->dense, bracket, event, *identity, *out );
    require( result.succeeded() && result.time == .25 && ops->finite_checks == result.iterations,
             "Identity refinement must check each dense sample once" );
    ops->finite_checks = 0;
    require( locator.locate( stepper->dense, bracket, event, *custom, *out ).status == status::invalid_state &&
             custom->calls == 2 && ops->finite_checks == 2, "Locator must reject a nonfinite reconstructed state" );
    stepper->dense.invalid = true;
    require( locator.locate( stepper->dense, bracket, event, *custom, *out ).status == status::invalid_state &&
             custom->calls == 2, "Invalid input must not reach reconstruction" );
    near_state( ops, *out, .25, .25, 1e-15 );
}

void test()
{
    test_sample_validation();
    auto ops = std::make_shared<operations>( 2 );
    auto out = buffer( ops );
    fill( *ops, *out, 7, 8 );
    auto                                      target = make_plane( ops, .3, .3, 1, 0 );
    plane::evaluation_workspace               work( ops );
    diagnostic_event                          event{ target->bind( work ) };
    diagnostic_policy                         p( ops );
    periodic_orbit::event_locator<operations> locator( ops );
    linear_dense                              dense{ ops };
    periodic_orbit::event_bracket<double>     bracket{ 0, 1, 0, 1, -.3, .7 };
    auto                                      result = locator.locate( dense, bracket, event, p, *out );
    require( result.succeeded() && std::abs( result.time - .3 ) < 1e-12, "Exact dense root" );
    near_state( ops, *out, .3, .3, 1e-12 );
    dense.start = 1;
    dense.dt    = -1;
    bracket     = { 1, -1, 0, 1, .7, -.3 };
    require( locator.locate( dense, bracket, event, p, *out ).succeeded(), "Backward bracket refinement" );
    fill( *ops, *out, 7, 8 );
    p.fail = true;
    require(
        locator.locate( dense, bracket, event, p, *out ).status == status::reconstruction_failure,
        "Locator policy failure"
    );
    near_state( ops, *out, 7, 8, 1e-15 );
    p.fail        = false;
    dense.invalid = true;
    require(
        locator.locate( dense, bracket, event, p, *out ).status == status::invalid_state, "Locator nonfinite state"
    );
    dense.invalid = false;
    event.invalid = true;
    require(
        locator.locate( dense, bracket, event, p, *out ).status == status::nonfinite_event, "Locator nonfinite event"
    );
    event.invalid       = false;
    bracket.lower_value = bracket.upper_value;
    require(
        locator.locate( dense, bracket, event, p, *out ).status == status::invalid_input, "Unbracketed contact accepted"
    );
    near_state( ops, *out, 7, 8, 1e-15 );
}
int main()
{
    return run( test );
}
