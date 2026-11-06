#include "section_test_support.h"
#include <scfd/utils/log_std.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/time_steppers/bdf/implicit_time_step.h>

using namespace section_tests;
namespace ts = nmfd::time_steppers;

struct shifted_rotation
{
    double              a, b;
    X                   in, out;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = a * in( i ) + ( i % 2 ? -b * in( i - 1 ) : b * in( i + 1 ) );
    }
};

// u'=1 in independent components, u(t)=t. Every BDF order reproduces
// this trajectory exactly; a transverse plane has a known crossing time.
// The generic section manager, not the BDF engine, owns event location.
struct drift_problem
{
    struct jacobian
    {
        std::shared_ptr<operations> ops;
        double                      a = 1, b = 1;
        bool                        oscillator = false;
        void                        apply( const X &in, X &out ) const
        {
            if ( oscillator )
                backend::for_each_type{}( shifted_rotation{ a, b, in, out }, out.size() );
            else
                ops->assign_lin_comb( a, in, out );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<operations> ops;
    wrap                        previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    double                      a = 1, b = 1;
    bool                        oscillator = false;
    explicit drift_problem( std::shared_ptr<operations> o )
        : ops( std::move( o ) ), previous( buffer( ops ) ), rhs( buffer( ops ) ),
          jacobi( std::make_shared<jacobian>( jacobian{ ops } ) )
    {
    }
    void set_time( double )
    {
    }
    void set_affine_shift( double mass, double force )
    {
        a = mass;
        b = force;
    }
    X &previous_state()
    {
        return *previous;
    }
    const X &previous_state() const
    {
        return *previous;
    }
    X &known_rhs()
    {
        return *rhs;
    }
    const X &known_rhs() const
    {
        return *rhs;
    }
    void apply( const X &in, X &out ) const
    {
        if ( oscillator )
        {
            backend::for_each_type{}( shifted_rotation{ a, b, in, out }, out.size() );
            ops->add_lin_comb( -a, *previous, out );
        }
        else
            ops->assign_lin_comb( a, in, -a, *previous, out );
        ops->add_lin_comb( -b, *rhs, out );
        if ( !oscillator )
            ops->add_mul_scalar( -b, 1, out );
    }
    void set_linearization_point( const X & )
    {
        jacobi->a          = a;
        jacobi->b          = b;
        jacobi->oscillator = oscillator;
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
};

using log_type    = scfd::utils::log_std;
using monitor     = nmfd::solvers::monitor_krylov<operations, log_type>;
using linear      = nmfd::solvers::gmres<operations, monitor, log_type, drift_problem::jacobian>;
using iteration   = nmfd::solvers::newton_iteration<operations, drift_problem, linear>;
using solver      = nmfd::solvers::nonlinear_solver<operations, log_type, drift_problem, iteration>;
using convergence = nmfd::solvers::default_convergence_strategy<operations, log_type, drift_problem>;
using single_step = ts::bdf::implicit_time_step<operations, drift_problem, adaptive, solver, true>;

std::shared_ptr<solver> make_solver( const std::shared_ptr<operations> &ops )
{
    linear::params lp;
    lp.basis_size      = 4;
    lp.batch_size      = 1;
    lp.monitor.abs_tol = 1e-13;
    lp.monitor.rel_tol = 1e-11;
    convergence::params cp;
    cp.abs_tol = 1e-12;
    cp.rel_tol = 1e-12;
    return std::make_shared<solver>(
        ops, nullptr, std::make_shared<iteration>( ops, std::make_shared<linear>( ops, nullptr, lp ) ),
        std::make_shared<convergence>( ops, nullptr, cp )
    );
}

void test()
{
    auto ops     = std::make_shared<operations>( 128 );
    auto initial = buffer( ops ), out = buffer( ops ), center = buffer( ops ), normal = buffer( ops );
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        for ( double direction : { -1., 1. } )
        {
            auto             problem = std::make_shared<drift_problem>( ops );
            auto             newton  = make_solver( ops );
            adaptive::params ap;
            ap.initial_step = ap.maximum_step = .02;
            ap.relative_tolerance = ap.absolute_tolerance = 1e-6;
            single_step::params sp{ method };
            sp.adaptive = true;
            auto single =
                std::make_shared<single_step>( ops, problem, std::make_shared<adaptive>( ops, ap ), newton, sp );
            periodic_orbit::section_map<operations, single_step, plane> flight(
                ops, single, std::make_shared<policy>( ops )
            );
            const double start = .35, crossing = start + direction * .733;
            ops->assign_scalar( start, *initial );
            ops->assign_scalar( crossing, *center );
            ops->assign_scalar( 1, *normal );
            auto                                          target = std::make_shared<plane>( ops, *center, *normal );
            periodic_orbit::flight_request<double, plane> request;
            request.start_time = start;
            request.end_time   = start + direction;
            request.target     = target;
            for ( int repeat = 0; repeat < 2; ++repeat )
            {
                auto report = flight.evaluate( *initial, request, *out );
                require(
                    report.succeeded() && std::abs( report.time - crossing ) < 1e-10, "BDF dense section crossing time"
                );
                ops->assign_lin_comb( 1, *out, -1, *center, *normal );
                require( ops->norm( *normal ) < 1e-9, "BDF dense section endpoint" );
                require(
                    single->get_order() == ( std::string( method ) == "BDF"    ? 5u
                                             : std::string( method ) == "BDF2" ? 2u
                                                                               : 1u ),
                    "Section flight did not exercise BDF interpolation order"
                );
                require_throws<std::logic_error>(
                    [&] { single->get_continuous_integration(); }, "Committed event retained BDF dense view"
                );
            }
        }
    // x'=-y, y'=x, (x,y)(0)=(1,0). The first x=0 crossing is
    // t=+/-pi/2 with state (0,+/-1). Refinement includes cold startup,
    // a genuinely curved trajectory and physical backward-time orientation.
    auto cycle_ops     = std::make_shared<operations>( 2 );
    auto cycle_initial = buffer( cycle_ops ), cycle_out = buffer( cycle_ops ), reference = buffer( cycle_ops ),
         error        = buffer( cycle_ops );
    auto cycle_target = make_plane( cycle_ops, 0, 0, 1, 0 );
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        for ( double direction : { -1., 1. } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                auto problem        = std::make_shared<drift_problem>( cycle_ops );
                problem->oscillator = true;
                adaptive::params ap;
                ap.initial_step = ap.maximum_step = .02 / ( level + 1 );
                ap.relative_tolerance = ap.absolute_tolerance = 1;
                single_step::params sp{ method };
                sp.adaptive = true;
                auto single = std::make_shared<single_step>(
                    cycle_ops, problem, std::make_shared<adaptive>( cycle_ops, ap ), make_solver( cycle_ops ), sp
                );
                periodic_orbit::section_map<operations, single_step, plane> flight(
                    cycle_ops, single, std::make_shared<policy>( cycle_ops )
                );
                periodic_orbit::flight_request<double, plane> request;
                request.end_time       = direction * 2;
                request.target         = cycle_target;
                request.rule.direction = direction > 0 ? periodic_orbit::crossing_direction::decreasing
                                                       : periodic_orbit::crossing_direction::increasing;
                fill( *cycle_ops, *cycle_initial, 1, 0 );
                fill( *cycle_ops, *reference, 0, direction );
                for ( int repeat = 0; repeat < 2; ++repeat )
                {
                    auto report = flight.evaluate( *cycle_initial, request, *cycle_out );
                    if ( !report.succeeded() )
                        std::cerr << method << " direction=" << direction << " status=" << int( report.status )
                                  << " time=" << report.time << " step=" << int( single->get_status() ) << '\n';
                    require( report.succeeded(), "BDF oscillator section flight failed" );
                    cycle_ops->assign_lin_comb( 1, *cycle_out, -1, *reference, *error );
                    errors[level] = cycle_ops->norm( *error ) + std::abs( report.time - direction * pi / 2 );
                }
            }
            require( errors[1] < .65 * errors[0] && errors[1] < .02, "BDF oscillator crossing did not refine" );
        }
    require( guarded_memory::host_bytes == 0, "BDF flight transferred vector data to/from host" );
}

int main()
{
    return run( test );
}
