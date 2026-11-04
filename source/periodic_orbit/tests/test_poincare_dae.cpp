#include "section_test_support.h"
using namespace section_tests;

// Lifted oscillator: x'=-y, y'=x, 0=z-x^2. The algebraic state is
// reconstructed by the problem-owned policy before every event evaluation.
// This tests the section boundary, not an automatic inversion of singular mass.
struct lifted_oscillator
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        const auto base = i - i % 3;
        return i % 3 == 0 ? -u[base + 1] : i % 3 == 1 ? u[base] : -2 * u[base] * u[base + 1];
    }
};
struct constraint_kernel
{
    const double       *input;
    double             *output;
    bool                reconstruct;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        const auto base = i - i % 3;
        if ( reconstruct )
        {
            if ( i % 3 == 2 )
                output[i] = input[base] * input[base];
        }
        else
            output[i] = i % 3 == 2 ? input[i] - input[base] * input[base] : 0;
    }
};
struct constraint_policy
{
    std::shared_ptr<operations> ops;
    wrap                        residual;
    explicit constraint_policy( std::shared_ptr<operations> o ) : ops( std::move( o ) ), residual( buffer( ops ) )
    {
    }
    bool validate_initial( double, const X &x )
    {
        backend::for_each_type{}( constraint_kernel{ x.raw_ptr(), ( *residual ).raw_ptr(), false }, x.size() );
        return ops->norm( *residual ) < 1e-10;
    }
    bool reconstruct( double, X &x )
    {
        backend::for_each_type{}( constraint_kernel{ x.raw_ptr(), x.raw_ptr(), true }, x.size() );
        return true;
    }
};
void test()
{
    auto ops = std::make_shared<operations>( 3 );
    auto u = buffer( ops ), out = buffer( ops ), center = buffer( ops ), normal = buffer( ops );
    fill( *ops, *u, 1, 0, 1, 3 );
    fill( *ops, *center, 0, 1, 0, 3 );
    fill( *ops, *normal, 1, 0, 0, 3 );
    auto target = std::make_shared<plane>( ops, *center, *normal );
    auto s      = make_step<lifted_oscillator>( ops, .03 );
    auto p      = std::make_shared<constraint_policy>( ops );
    periodic_orbit::section_map<operations, step<lifted_oscillator>, plane, constraint_policy> flight( ops, s, p );
    periodic_orbit::flight_request<double, plane>                                              request;
    request.start_time     = 2;
    request.end_time       = 4;
    request.target         = target;
    request.rule.direction = periodic_orbit::crossing_direction::decreasing;
    const auto report      = flight.evaluate( *u, request, *out );
    require( report.succeeded() && std::abs( report.time - ( 2 + pi / 2 ) ) < 1e-9, "Constrained event time" );
    require( p->validate_initial( report.time, *out ), "Event state violates algebraic constraint" );
    fill( *ops, *u, 1, 0, 0, 3 );
    require(
        flight.evaluate( *u, request, *out ).status == status::invalid_state, "Inconsistent initial constraint accepted"
    );
}
int main()
{
    return run( test );
}
