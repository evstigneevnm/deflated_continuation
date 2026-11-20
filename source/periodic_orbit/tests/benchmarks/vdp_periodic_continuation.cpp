#include "ode_continuation_benchmark.h"

namespace ode_orbit_tests
{
// x'=y, y'=mu*(1-x^2)*y-x. Independent attracting cycle at mu=1:
// SciPy DOP853 from (2,0) to t=120, rtol=1e-13, atol=1e-14.
// Use the final x=0 increasing event and the interval from its predecessor.
struct vdp_branch
{
    static constexpr const char                  *name             = "vdp_periodic";
    static constexpr double                       anchor_parameter = 1, period = 6.6632868593231365;
    inline static constexpr std::array<double, 2> seed{ 0., 2.1727136926225405 };
    double                                        mu = anchor_parameter;
    __DEVICE_TAG__ double                         rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i == 0 ? u[1] : mu * ( 1 - u[0] * u[0] ) * u[1] - u[0];
    }
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const double *u, const double *v, double ) const
    {
        return i == 0 ? v[1] : ( -1 - 2 * mu * u[0] * u[1] ) * v[0] + mu * ( 1 - u[0] * u[0] ) * v[1];
    }
    vdp_branch with_parameter( double value ) const
    {
        auto copy = *this;
        copy.mu   = value;
        return copy;
    }
    __DEVICE_TAG__ double parameter_rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i == 0 ? 0. : ( 1 - u[0] * u[0] ) * u[1];
    }
};
}

int main( int argc, char **argv )
{
    using namespace ode_orbit_tests;
    return section_tests::run( [&] {
        const vdp_branch model{};
        const auto       settings     = branch_settings::read( argc, argv, model.name );
        auto             make_flights = [&] {
            return continuation_fixture(
                model, model.seed, model.period, settings.sections, settings.step, settings.start, settings.budget
            );
        };
        auto make_solver = [&]( auto &flights ) {
            return orbit_benchmarks::branch_solver(
                flights, settings.arclength_step, settings.newton_tolerance, settings.iterations
            );
        };
        branch_case checks( make_flights, make_solver, settings, origin_reference<2> );
        auto        anchor = checks.correct_anchor( model.mu );
        checks.trace( *anchor, model.mu, -1 );
        checks.trace( *anchor, model.mu, +1 );
        checks.write_report();
    } );
}
