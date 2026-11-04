#include "section_test_support.h"
#include <scfd/utils/log_std.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/time_steppers/runge_kutta/implicit_time_step.h>
using namespace section_tests;

// Non-autonomous u'=1+2*t, u(t)=t+t^2, in independent components.
// The exact quadratic is reproduced by second-order implicit dense formulas.
// Use real NMFD Newton/GMRES stages and guarded backend vectors on all backends.
struct quadratic_stage
{
    struct jacobian
    {
        std::shared_ptr<operations> ops;
        double                      factor = 1;
        void                        apply( const X &x, X &out ) const
        {
            ops->assign_lin_comb( factor, x, out );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<operations> ops;
    wrap                        previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    double                      time = 0, a = 1, b = 1;
    explicit quadratic_stage( std::shared_ptr<operations> o )
        : ops( std::move( o ) ), previous( buffer( ops ) ), rhs( buffer( ops ) ),
          jacobi( std::make_shared<jacobian>( jacobian{ ops } ) )
    {
    }
    void set_time( double t )
    {
        time = t;
    }
    void set_affine_shift( double alpha )
    {
        const double scale = std::max( 1., std::abs( alpha ) );
        a                  = alpha / scale;
        b                  = 1 / scale;
    }
    X &previous_state()
    {
        return *previous;
    }
    X &known_rhs()
    {
        return *rhs;
    }
    const X &previous_state() const
    {
        return *previous;
    }
    const X &known_rhs() const
    {
        return *rhs;
    }
    void apply( const X &x, X &out ) const
    {
        ops->assign_lin_comb( 1, x, -1, *previous, out );
        ops->scale( a, out );
        ops->add_lin_comb( -b, *rhs, out );
        ops->add_mul_scalar( -b * ( 1 + 2 * time ), 1, out );
    }
    void set_linearization_point( const X & )
    {
        jacobi->factor = a;
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
    bool evaluate_rate( const X &, X &rate ) const
    {
        ops->assign_scalar( 1 + 2 * time, rate );
        return true;
    }
};

void test()
{
    using log         = scfd::utils::log_std;
    using monitor     = nmfd::solvers::monitor_krylov<operations, log>;
    using linear      = nmfd::solvers::gmres<operations, monitor, log, quadratic_stage::jacobian>;
    using iteration   = nmfd::solvers::newton_iteration<operations, quadratic_stage, linear>;
    using solver      = nmfd::solvers::nonlinear_solver<operations, log, quadratic_stage, iteration>;
    using convergence = nmfd::solvers::default_convergence_strategy<operations, log, quadratic_stage>;
    using single_step =
        nmfd::time_steppers::runge_kutta::implicit_time_step<operations, quadratic_stage, fixed, solver, true>;
    auto       ops     = std::make_shared<operations>( 2 );
    auto       initial = buffer( ops ), out = buffer( ops ), center = buffer( ops ), normal = buffer( ops );
    const auto exact = []( double t ) { return t + t * t; };
    ops->assign_scalar( exact( .35 ), *initial );
    ops->assign_scalar( exact( .73 ), *center );
    ops->assign_scalar( 1, *normal );
    auto target = std::make_shared<plane>( ops, *center, *normal );
    for ( const auto *method : { "CN", "IM", "SDIRK2(1)2", "ESDIRK2(1)3" } )
    {
        auto           problem = std::make_shared<quadratic_stage>( ops );
        linear::params lp;
        lp.basis_size            = 4;
        lp.batch_size            = 1;
        lp.monitor.abs_tol       = 1e-13;
        lp.monitor.rel_tol       = 1e-11;
        lp.monitor.max_iters_num = 40;
        auto                lin  = std::make_shared<linear>( ops, nullptr, lp );
        auto                iter = std::make_shared<iteration>( ops, lin );
        convergence::params cp;
        cp.abs_tol       = 1e-12;
        cp.rel_tol       = 1e-12;
        cp.max_iters_num = 30;
        auto newton = std::make_shared<solver>( ops, nullptr, iter, std::make_shared<convergence>( ops, nullptr, cp ) );
        auto single = std::make_shared<single_step>(
            ops, problem, std::make_shared<fixed>( fixed::params{ .2 } ), newton, single_step::params{ method }
        );
        periodic_orbit::section_map<operations, single_step, plane> flight(
            ops, single, std::make_shared<policy>( ops )
        );
        periodic_orbit::flight_request<double, plane> request;
        request.start_time = .35;
        request.end_time   = 1;
        request.target     = target;
        const auto report  = flight.evaluate( *initial, request, *out );
        require( report.succeeded() && std::abs( report.time - .73 ) < 1e-10, "Implicit dense event time" );
        ops->assign_lin_comb( 1, *out, -1, *center, *normal );
        require( ops->norm( *normal ) < 1e-10, "Implicit event endpoint" );
        require( flight.evaluate( *initial, request, *out ).succeeded(), "Implicit reset after shortened commit" );
    }
    require( guarded_memory::host_bytes == 0, "Implicit flight transferred vector data to/from host" );
}
int main()
{
    return run( test );
}
