#ifndef PERIODIC_ORBIT_ORBIT_METHOD_ASSEMBLY_H
#define PERIODIC_ORBIT_ORBIT_METHOD_ASSEMBLY_H

#include "variational_test_support.h"
#include "newton_test_support.h"
#include <nmfd/time_steppers/runge_kutta/implicit_time_step.h>
#include <nmfd/time_steppers/bdf/implicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <periodic_orbit/flows/prepared_bdf_section.h>
#include <periodic_orbit/flows/prepared_dirk_section.h>
#include <periodic_orbit/flows/prepared_explicit_section.h>

namespace orbit_benchmarks
{
using namespace section_tests;

// Identity-mass ODE adapter: R=a*(Y-w)-b*(f(t,Y)+r). Model kernels
// supply the same physical RHS and Jv used by the explicit benchmarks.
template <class Model>
struct stage_kernel
{
    double              time, a, b;
    const double       *state, *previous, *rhs, *direction;
    double             *out;
    bool                tangent;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        const Model model{};
        out[i] = tangent ? a * direction[i] - b * model.jacobian( i, state, direction, time )
                         : a * ( state[i] - previous[i] ) - b * ( model.rhs( i, state, time ) + rhs[i] );
    }
};
template <class Model>
struct stage_jacobian
{
    std::shared_ptr<operations> ops;
    wrap                        point;
    double                      time = 0, a = 0, b = 1;
    explicit stage_jacobian( std::shared_ptr<operations> space ) : ops( std::move( space ) ), point( buffer( ops ) )
    {
    }
    void apply( const X &v, X &out ) const
    {
        backend::for_each_type{}(
            stage_kernel<Model>{ time, a, b, ( *point ).raw_ptr(), nullptr, nullptr, v.raw_ptr(), out.raw_ptr(), true },
            out.size()
        );
    }
};
template <class Model>
struct stage_problem : problem<Model>
{
    using jacobi_operator_type = stage_jacobian<Model>;
    wrap                                  previous, rhs;
    std::shared_ptr<jacobi_operator_type> jacobi;
    double                                a = 0, b = 1;
    explicit stage_problem( const std::shared_ptr<operations> &ops )
        : problem<Model>{ ops, {} }, previous( buffer( ops ) ), rhs( buffer( ops ) ),
          jacobi( std::make_shared<jacobi_operator_type>( ops ) )
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
        backend::for_each_type{}(
            stage_kernel<Model>{
                this->time, a, b, in.raw_ptr(), ( *previous ).raw_ptr(), ( *rhs ).raw_ptr(), nullptr, out.raw_ptr(),
                false
            },
            out.size()
        );
    }
    void set_linearization_point( const X &in )
    {
        this->ops->assign( in, *jacobi->point );
        jacobi->time = this->time;
        jacobi->a    = a;
        jacobi->b    = b;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi;
    }
};
template <class Model>
struct stage_sensitivity
{
    std::shared_ptr<operations> ops;
    bool stage_tangent_rhs( const periodic_orbit::dirk_stage_context<operations> &context, const X &dw, X &rhs ) const
    {
        ops->assign_lin_comb( context.a, dw, rhs );
        return true;
    }
    bool rate_jacobian_apply( double time, const X &state, const X &v, X &out ) const
    {
        backend::for_each_type{}(
            jacobian_kernel<Model>{ {}, time, state.raw_ptr(), v.raw_ptr(), out.raw_ptr() }, out.size()
        );
        return true;
    }
};
template <class Model>
struct stage_solvers
{
    using types = newton_types<operations, stage_problem<Model>>;
    static auto make_linear( const std::shared_ptr<operations> &ops )
    {
        typename types::linear::params p;
        p.basis_size = static_cast<unsigned>( std::min<std::size_t>( 8, ops->size() ) );
        p.batch_size = 1;
        // Tangent directions can be arbitrarily small inside outer Krylov solves.
        p.monitor.abs_tol       = 0;
        p.monitor.rel_tol       = 1e-12;
        p.monitor.max_iters_num = 40;
        return std::make_shared<typename types::linear>( ops, nullptr, p );
    }
    static auto
    make_nonlinear( const std::shared_ptr<operations> &ops, const std::shared_ptr<typename types::linear> &linear )
    {
        typename types::convergence::params p;
        p.abs_tol       = 2e-14;
        p.rel_tol       = 0;
        p.max_iters_num = 20;
        p.verbose       = false;
        return std::make_shared<typename types::solver>(
            ops, nullptr, std::make_shared<typename types::iteration>( ops, linear ),
            std::make_shared<typename types::convergence>( ops, nullptr, p )
        );
    }
};
template <class Model>
struct stage_provider_factory
{
    using solvers       = stage_solvers<Model>;
    using provider_type = periodic_orbit::dirk_stage_provider<
        operations, stage_problem<Model>, typename solvers::types::solver, typename solvers::types::linear,
        stage_sensitivity<Model>>;
    std::shared_ptr<operations> ops;
    auto                        make_provider() const
    {
        auto linear = solvers::make_linear( ops );
        return std::make_shared<provider_type>(
            ops, std::make_shared<stage_problem<Model>>( ops ), solvers::make_nonlinear( ops, linear ), linear,
            std::make_shared<stage_sensitivity<Model>>( stage_sensitivity<Model>{ ops } )
        );
    }
};
template <class Model>
struct explicit_assembly : explicit_step_assembly<Model>
{
    using factory_type = periodic_orbit::explicit_flight_factory<operations, tangent_rate<Model>, plane>;
    static auto make_factory(
        const std::shared_ptr<operations> &ops, std::size_t, std::shared_ptr<const tangent_rate<Model>> rate = nullptr
    )
    {
        return std::make_shared<factory_type>(
            ops, rate ? std::move( rate ) : std::make_shared<tangent_rate<Model>>()
        );
    }
};
template <class Model>
struct dirk_assembly
{
    using solvers   = stage_solvers<Model>;
    using step_type = nmfd::time_steppers::runge_kutta::implicit_time_step<
        operations, stage_problem<Model>, fixed, typename solvers::types::solver, true>;
    using factory_type = periodic_orbit::dirk_flight_factory<operations, stage_provider_factory<Model>, plane>;
    using record_type  = periodic_orbit::flight_step_record<typename step_type::scalar_type>;
    static auto make_stepper( const std::shared_ptr<operations> &ops, double h, const std::string &method )
    {
        return std::make_shared<step_type>(
            ops, std::make_shared<stage_problem<Model>>( ops ), std::make_shared<fixed>( fixed::params{ h } ),
            solvers::make_nonlinear( ops, solvers::make_linear( ops ) ), typename step_type::params{ method }
        );
    }
    static auto make_factory(
        const std::shared_ptr<operations> &ops, std::size_t budget,
        const std::shared_ptr<const tangent_rate<Model>> & = nullptr
    )
    {
        return std::make_shared<factory_type>(
            ops, std::make_shared<stage_provider_factory<Model>>( stage_provider_factory<Model>{ ops } ),
            typename factory_type::params{ budget }
        );
    }
};
template <class Model, unsigned int Order>
struct bdf_assembly
{
    using solvers   = stage_solvers<Model>;
    using step_type = nmfd::time_steppers::bdf::implicit_time_step<
        operations, stage_problem<Model>, fixed, typename solvers::types::solver, true>;
    using record_type  = periodic_orbit::bdf_step_record<typename step_type::scalar_type>;
    using factory_type = periodic_orbit::bdf_flight_factory<operations, stage_provider_factory<Model>, plane>;
    static auto make_stepper( const std::shared_ptr<operations> &ops, double h, const std::string &method )
    {
        if ( method != ( Order == 1 ? "BDF1" : "BDF2" ) )
            throw std::invalid_argument( "BDF assembly and method disagree" );
        return std::make_shared<step_type>(
            ops, std::make_shared<stage_problem<Model>>( ops ), std::make_shared<fixed>( fixed::params{ h } ),
            solvers::make_nonlinear( ops, solvers::make_linear( ops ) ), typename step_type::params{ method }
        );
    }
    static auto make_factory(
        const std::shared_ptr<operations> &ops, std::size_t budget,
        const std::shared_ptr<const tangent_rate<Model>> & = nullptr
    )
    {
        return std::make_shared<factory_type>(
            ops, std::make_shared<stage_provider_factory<Model>>( stage_provider_factory<Model>{ ops } ),
            typename factory_type::params{ Order, budget }
        );
    }
};
struct adaptive_bdf_options
{
    double       relative_tolerance = 1e-12, absolute_tolerance = 1e-13;
    double       initial_step = .001, maximum_step = .1;
    unsigned int maximum_order = 5;
    template <class Configuration>
    static adaptive_bdf_options read( const Configuration &c )
    {
        return {
            c.value( "relative_tolerance", 1e-12 ), c.value( "absolute_tolerance", 1e-13 ), c.value( "step", .001 ),
            c.value( "maximum_step", .1 ), c.value( "maximum_order", 5u )
        };
    }
};
template <class Configuration, class Segments>
void add_bdf_statistics( Configuration &report, const Segments &segments )
{
    using record = typename std::decay_t<decltype( segments.front().flight->records() )>::value_type;
    if constexpr ( std::is_same_v<record, periodic_orbit::bdf_step_record<double>> )
    {
        std::array<std::size_t, 5> orders{};
        std::size_t                rejected = 0, accepted = 0;
        double                     minimum = std::numeric_limits<double>::max(), maximum = 0;
        for ( const auto &segment : segments )
            for ( const auto &r : segment.flight->records() )
            {
                ++orders.at( r.order - 1 );
                ++accepted;
                rejected += r.attempts - 1;
                minimum = std::min( minimum, std::abs( r.dt ) );
                maximum = std::max( maximum, std::abs( r.dt ) );
            }
        report["integration"] = {
            { "orders", orders },
            { "accepted_steps", accepted },
            { "rejected_attempts", rejected },
            { "minimum_step", minimum },
            { "maximum_step", maximum }
        };
    }
}
template <class Model>
struct adaptive_bdf_assembly
{
    using solvers   = stage_solvers<Model>;
    using control   = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations>;
    using step_type = nmfd::time_steppers::bdf::implicit_time_step<
        operations, stage_problem<Model>, control, typename solvers::types::solver, true>;
    using record_type  = periodic_orbit::bdf_step_record<double>;
    using factory_type = periodic_orbit::bdf_flight_factory<operations, stage_provider_factory<Model>, plane>;
    adaptive_bdf_options options;
    auto make_stepper( const std::shared_ptr<operations> &ops, double h, const std::string &method ) const
    {
        if ( method != "BDF" || !std::isfinite( options.initial_step ) || options.initial_step <= 0 )
            throw std::invalid_argument( "Invalid adaptive BDF assembly" );
        typename control::params c;
        // Verification tightens LTE tolerances as well as the initial step.
        const double factor  = std::min( 1., h / options.initial_step );
        c.relative_tolerance = options.relative_tolerance * factor;
        c.absolute_tolerance = options.absolute_tolerance * factor;
        c.initial_step       = h;
        c.maximum_step       = options.maximum_step * factor;
        typename step_type::params p;
        p.method        = method;
        p.adaptive      = true;
        p.maximum_order = options.maximum_order;
        return std::make_shared<step_type>(
            ops, std::make_shared<stage_problem<Model>>( ops ), std::make_shared<control>( ops, c ),
            solvers::make_nonlinear( ops, solvers::make_linear( ops ) ), p
        );
    }
    auto make_factory(
        const std::shared_ptr<operations> &ops, std::size_t budget,
        const std::shared_ptr<const tangent_rate<Model>> & = nullptr
    ) const
    {
        return std::make_shared<factory_type>(
            ops, std::make_shared<stage_provider_factory<Model>>( stage_provider_factory<Model>{ ops } ),
            typename factory_type::params{ options.maximum_order, budget, true }
        );
    }
};
template <class Model, class Run>
int dispatch( const std::string &method, Run run, const adaptive_bdf_options &options = {} )
{
    if ( method == "BDF" )
        return run( adaptive_bdf_assembly<Model>{ options } );
    if ( method == "BDF1" )
        return run( bdf_assembly<Model, 1>{} );
    if ( method == "BDF2" )
        return run( bdf_assembly<Model, 2>{} );
    const auto table = nmfd::time_steppers::runge_kutta::make_butcher_table( method );
    if ( !table.has_dense_output() )
        throw std::invalid_argument( "Orbit benchmarks require native dense output" );
    if ( table.type() == decltype( table )::scheme_type::explicit_rk )
        return run( explicit_assembly<Model>{} );
    if ( !periodic_orbit::dirk_flight_replay<
             operations, typename stage_provider_factory<Model>::provider_type>::supports( table ) )
        throw std::invalid_argument( "Unsupported DIRK orbit method" );
    return run( dirk_assembly<Model>{} );
}
}
#endif
