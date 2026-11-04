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
void test()
{
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
