#ifndef PERIODIC_ORBIT_IMPLICIT_FLIGHT_TEST_SUPPORT_H
#define PERIODIC_ORBIT_IMPLICIT_FLIGHT_TEST_SUPPORT_H

#include "prepared_cycle_support.h"
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/time_steppers/runge_kutta/implicit_time_step.h>
#include <periodic_orbit/flows/prepared_dirk_section.h>

namespace implicit_flights
{
using namespace section_tests;
namespace ts = nmfd::time_steppers;
enum class system_kind
{
    cycle,
    coupled_mass,
    state_mass,
    time_rotation,
    linear_dae
};

// cycle: (x',y')=((1-r^2)x-y,(1-r^2)y+x), stable r=1 cycle.
// coupled_mass: M*u'=M*g(u), M=[[2,1],[1,3]], same isolated cycle.
// state_mass: (1+r^2)*u'=(1+r^2)*g(u), same cycle with a mass derivative.
// time_rotation: (x',y')=t*(-y,x), non-autonomous quarter-turn timing.
// linear_dae: (x',y')=(-y,x), 0=z-x-y, singular diag(1,1,0) mass.
// The DAE constraint is linear, so its dense polynomial needs no reconstruction.
struct model
{
    system_kind        kind = system_kind::cycle;
    __DEVICE_TAG__ int components() const
    {
        return kind == system_kind::linear_dae ? 3 : 2;
    }
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u, double t ) const
    {
        const auto   base = i - i % components();
        const double x = u( base ), y = u( base + 1 );
        const double radial =
            kind == system_kind::time_rotation || kind == system_kind::linear_dae ? 0 : 1 - x * x - y * y;
        const double speed = kind == system_kind::time_rotation ? t : 1;
        return i % components() == 0 ? radial * x - speed * y : i % components() == 1 ? radial * y + speed * x : x - y;
    }
    __DEVICE_TAG__ double rate_jacobian( std::ptrdiff_t i, const X &u, const X &v, double t ) const
    {
        const auto   base = i - i % components();
        const double x = u( base ), y = u( base + 1 ), dx = v( base ), dy = v( base + 1 );
        const bool   rotating = kind == system_kind::time_rotation || kind == system_kind::linear_dae;
        const double radial   = rotating ? 0 : 1 - x * x - y * y;
        const double dr       = rotating ? 0 : -2 * ( x * dx + y * dy );
        const double speed    = kind == system_kind::time_rotation ? t : 1;
        return i % components() == 0   ? radial * dx + dr * x - speed * dy
               : i % components() == 1 ? radial * dy + dr * y + speed * dx
                                       : dx - dy;
    }
    __DEVICE_TAG__ double mass( std::ptrdiff_t i, const X &u, const X &v ) const
    {
        const auto base = i - i % components();
        if ( kind == system_kind::linear_dae && i % 3 == 2 )
            return 0;
        if ( kind == system_kind::coupled_mass )
            return i % 2 ? v( base ) + 3 * v( base + 1 ) : 2 * v( base ) + v( base + 1 );
        const double m =
            kind == system_kind::state_mass ? 1 + u( base ) * u( base ) + u( base + 1 ) * u( base + 1 ) : 1;
        return m * v( i );
    }
    __DEVICE_TAG__ double
    residual( std::ptrdiff_t i, double t, double a, double b, const X &u, const X &previous, const X &rhs ) const
    {
        if ( kind == system_kind::linear_dae && i % 3 == 2 )
            return u( i ) - u( i - 2 ) - u( i - 1 ) - rhs( i );
        const auto   base = i - i % components();
        const double e0   = a * ( u( base ) - previous( base ) ) - b * ( rate( base, u, t ) + rhs( base ) );
        const double e1 =
            a * ( u( base + 1 ) - previous( base + 1 ) ) - b * ( rate( base + 1, u, t ) + rhs( base + 1 ) );
        if ( kind == system_kind::coupled_mass )
            return i % 2 ? e0 + 3 * e1 : 2 * e0 + e1;
        const double m =
            kind == system_kind::state_mass ? 1 + u( base ) * u( base ) + u( base + 1 ) * u( base + 1 ) : 1;
        return m * ( i % components() ? e1 : e0 );
    }
    __DEVICE_TAG__ double
    jacobian( std::ptrdiff_t i, double t, double a, double b, const X &u, const X &previous, const X &v ) const
    {
        if ( kind == system_kind::linear_dae && i % 3 == 2 )
            return v( i ) - v( i - 2 ) - v( i - 1 );
        const auto   base = i - i % components();
        const double e0   = a * v( base ) - b * rate_jacobian( base, u, v, t ),
                     e1   = a * v( base + 1 ) - b * rate_jacobian( base + 1, u, v, t );
        if ( kind == system_kind::coupled_mass )
            return i % 2 ? e0 + 3 * e1 : 2 * e0 + e1;
        if ( kind == system_kind::state_mass )
        {
            const double m      = 1 + u( base ) * u( base ) + u( base + 1 ) * u( base + 1 );
            const double dm     = 2 * ( u( base ) * v( base ) + u( base + 1 ) * v( base + 1 ) );
            const double defect = a * ( u( i ) - previous( i ) ) - b * rate( i, u, t );
            return m * ( i % 2 ? e1 : e0 ) + dm * defect;
        }
        return i % components() ? e1 : e0;
    }
};

struct model_kernel
{
    model               equation;
    double              time = 0, a = 0, b = 1;
    X                   state, previous, rhs, direction, out;
    int                 action = 0;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = action == 0   ? equation.residual( i, time, a, b, state, previous, rhs )
                   : action == 1 ? equation.jacobian( i, time, a, b, state, previous, direction )
                   : action == 2 ? equation.rate( i, state, time )
                   : action == 3 ? equation.rate_jacobian( i, state, direction, time )
                                 : a * equation.mass( i, state, direction );
    }
};
inline void execute( const operations &ops, model_kernel kernel )
{
    backend::for_each_type{}( kernel, ops.get_vector_size() );
}
struct stage_jacobian
{
    std::shared_ptr<operations> ops;
    wrap                        point, previous;
    model                       equation;
    double                      time = 0, a = 0, b = 1;
    stage_jacobian( std::shared_ptr<operations> o, model e )
        : ops( std::move( o ) ), point( buffer( ops ) ), previous( buffer( ops ) ), equation( e )
    {
    }
    void apply( const X &in, X &out ) const
    {
        execute( *ops, { equation, time, a, b, *point, *previous, {}, in, out, 1 } );
    }
};
struct residual_problem
{
    using jacobi_operator_type = stage_jacobian;
    std::shared_ptr<operations>     ops;
    wrap                            previous, rhs;
    std::shared_ptr<stage_jacobian> jacobi;
    model                           equation;
    double                          time = 0, a = 0, b = 1;
    residual_problem( std::shared_ptr<operations> o, model e )
        : ops( std::move( o ) ), previous( buffer( ops ) ), rhs( buffer( ops ) ),
          jacobi( std::make_shared<stage_jacobian>( ops, e ) ), equation( e )
    {
    }
    void set_time( double t )
    {
        time = t;
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
        execute( *ops, { equation, time, a, b, in, *previous, *rhs, {}, out, 0 } );
    }
    void set_linearization_point( const X &in )
    {
        ops->assign( in, *jacobi->point );
        ops->assign( *previous, *jacobi->previous );
        jacobi->time = time;
        jacobi->a    = a;
        jacobi->b    = b;
    }
    std::shared_ptr<const stage_jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
};
struct rate_problem : residual_problem
{
    using residual_problem::residual_problem;
    bool evaluate_rate( const X &in, X &out )
    {
        execute( *ops, { equation, time, 0, 1, in, {}, {}, {}, out, 2 } );
        return true;
    }
};
struct control_flags
{
    bool refuse_rhs = false, refuse_rate = false, fail_creation = false;
};
struct sensitivity
{
    std::shared_ptr<operations>    ops;
    model                          equation;
    std::shared_ptr<control_flags> flags;
    bool stage_tangent_rhs( const periodic_orbit::dirk_stage_context<operations> &c, const X &dw, X &out ) const
    {
        if ( flags->refuse_rhs )
            return false;
        execute( *ops, { equation, c.time, c.a, c.b, c.state, c.previous, {}, dw, out, 4 } );
        return true;
    }
    bool rate_jacobian_apply( double time, const X &state, const X &v, X &out ) const
    {
        if ( flags->refuse_rate )
            return false;
        execute( *ops, { equation, time, 0, 1, state, {}, {}, v, out, 3 } );
        return true;
    }
};
using log_type = scfd::utils::log_std;
using linear =
    nmfd::solvers::gmres<operations, nmfd::solvers::monitor_krylov<operations, log_type>, log_type, stage_jacobian>;
template <class Problem>
using iteration = nmfd::solvers::newton_iteration<operations, Problem, linear>;
template <class Problem>
using newton = nmfd::solvers::nonlinear_solver<operations, log_type, Problem, iteration<Problem>>;
template <class Problem>
std::shared_ptr<newton<Problem>> make_newton( std::shared_ptr<operations> ops, std::shared_ptr<linear> lin )
{
    using strategy = nmfd::solvers::default_convergence_strategy<operations, log_type, Problem>;
    typename strategy::params p;
    p.abs_tol       = 1e-12;
    p.rel_tol       = 1e-13;
    p.max_iters_num = 20;
    return std::make_shared<newton<Problem>>(
        ops, nullptr, std::make_shared<iteration<Problem>>( ops, lin ), std::make_shared<strategy>( ops, nullptr, p )
    );
}
inline std::shared_ptr<linear> make_linear( std::shared_ptr<operations> ops )
{
    linear::params p;
    p.basis_size            = 8;
    p.batch_size            = 1;
    p.monitor.abs_tol       = 1e-13;
    p.monitor.rel_tol       = 1e-12;
    p.monitor.max_iters_num = 80;
    return std::make_shared<linear>( ops, nullptr, p );
}
struct provider_factory
{
    using provider_type =
        periodic_orbit::dirk_stage_provider<operations, rate_problem, newton<rate_problem>, linear, sensitivity>;
    std::shared_ptr<operations>    ops;
    model                          equation;
    std::shared_ptr<control_flags> flags     = std::make_shared<control_flags>();
    mutable std::size_t            creations = 0;
    std::shared_ptr<provider_type> make_provider() const
    {
        if ( flags->fail_creation )
            throw std::runtime_error( "Injected provider construction failure" );
        ++creations;
        auto lin = make_linear( ops );
        return std::make_shared<provider_type>(
            ops, std::make_shared<rate_problem>( ops, equation ), make_newton<rate_problem>( ops, lin ), lin,
            std::make_shared<sensitivity>( sensitivity{ ops, equation, flags } )
        );
    }
};
using factory_type = periodic_orbit::dirk_flight_factory<operations, provider_factory, plane>;
using native_step  = ts::runge_kutta::implicit_time_step<operations, rate_problem, fixed, newton<rate_problem>, true>;
struct physical_rate
{
    std::shared_ptr<operations> ops;
    model                       equation;
    bool                        evaluate( double time, const X &state, X &out ) const
    {
        execute( *ops, { equation, time, 0, 1, state, {}, {}, {}, out, 2 } );
        return true;
    }
};
using recorded_map = periodic_orbit::recorded_section_map<operations, native_step, plane, physical_rate>;
inline std::shared_ptr<recorded_map>
make_map( std::shared_ptr<operations> ops, model equation, const std::string &method, double h )
{
    auto step = std::make_shared<native_step>(
        ops, std::make_shared<rate_problem>( ops, equation ), std::make_shared<fixed>( fixed::params{ h } ),
        make_newton<rate_problem>( ops, make_linear( ops ) ), native_step::params{ method }
    );
    return std::make_shared<recorded_map>(
        ops, step, std::make_shared<physical_rate>( physical_rate{ ops, equation } )
    );
}
inline auto request( std::shared_ptr<operations> ops, double start = .4 )
{
    periodic_orbit::flight_request<double, plane> r;
    r.start_time                  = start;
    r.end_time                    = start + 2;
    r.target                      = make_plane( ops, 0, 0, 1, 0 );
    r.rule.direction              = periodic_orbit::crossing_direction::decreasing;
    r.rule.minimum_crossing_speed = .1;
    return r;
}
}
#endif
