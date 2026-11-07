#ifndef PERIODIC_ORBIT_BDF_FLIGHT_TEST_SUPPORT_H
#define PERIODIC_ORBIT_BDF_FLIGHT_TEST_SUPPORT_H

#include "implicit_flight_test_support.h"
#include <nmfd/time_steppers/bdf/implicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <periodic_orbit/flows/prepared_bdf_section.h>

namespace bdf_flights
{
using namespace section_tests;
using implicit_flights::control_flags;
using implicit_flights::linear;
using implicit_flights::log_type;
using implicit_flights::model;
using implicit_flights::physical_rate;
using implicit_flights::residual_problem;
using implicit_flights::system_kind;
template <class Problem>
using newton = implicit_flights::newton<Problem>;
namespace ts = nmfd::time_steppers;

// These systems are described in implicit_flight_test_support.h. BDF needs
// shifted residuals and their history derivative, never an explicit mass inverse.
struct sensitivity
{
    implicit_flights::sensitivity base;
    bool
    stage_tangent_rhs( const periodic_orbit::implicit_stage_context<operations> &context, const X &dw, X &out ) const
    {
        return base.stage_tangent_rhs( context, dw, out );
    }
};
inline auto make_linear( const std::shared_ptr<operations> &ops )
{
    linear::params p;
    p.basis_size            = 8;
    p.batch_size            = 1;
    p.monitor.abs_tol       = 0;
    p.monitor.rel_tol       = 1e-12;
    p.monitor.max_iters_num = 80;
    return std::make_shared<linear>( ops, nullptr, p );
}
struct provider_factory
{
    using provider_type = periodic_orbit::implicit_stage_provider<
        operations, residual_problem, newton<residual_problem>, linear, sensitivity>;
    std::shared_ptr<operations>    ops;
    model                          equation;
    std::shared_ptr<control_flags> flags     = std::make_shared<control_flags>();
    mutable std::size_t            creations = 0;
    auto                           make_provider() const
    {
        if ( flags->fail_creation )
            throw std::runtime_error( "Injected BDF provider construction failure" );
        ++creations;
        auto lin = make_linear( ops );
        return std::make_shared<provider_type>(
            ops, std::make_shared<residual_problem>( ops, equation ),
            implicit_flights::make_newton<residual_problem>( ops, lin ), lin,
            std::make_shared<sensitivity>( sensitivity{ { ops, equation, flags } } )
        );
    }
};
using factory_type = periodic_orbit::bdf_flight_factory<operations, provider_factory, plane>;
using native_step  = ts::bdf::implicit_time_step<operations, residual_problem, fixed, newton<residual_problem>, true>;
using record_type  = periodic_orbit::bdf_step_record<double>;
using recorded_map = periodic_orbit::recorded_section_map<operations, native_step, plane, physical_rate, record_type>;
inline auto make_step( const std::shared_ptr<operations> &ops, model equation, unsigned int order, double h )
{
    return std::make_shared<native_step>(
        ops, std::make_shared<residual_problem>( ops, equation ), std::make_shared<fixed>( fixed::params{ h } ),
        implicit_flights::make_newton<residual_problem>( ops, make_linear( ops ) ),
        native_step::params{ order == 1 ? "BDF1" : "BDF2" }
    );
}
inline auto make_map( const std::shared_ptr<operations> &ops, model equation, unsigned int order, double h )
{
    return std::make_shared<recorded_map>(
        ops, make_step( ops, equation, order, h ), std::make_shared<physical_rate>( physical_rate{ ops, equation } )
    );
}
using implicit_flights::request;
using adaptive_control = ts::integration::time_step_adaptation_scipy<operations>;
using adaptive_step =
    ts::bdf::implicit_time_step<operations, residual_problem, adaptive_control, newton<residual_problem>, true>;
using adaptive_map = periodic_orbit::recorded_section_map<operations, adaptive_step, plane, physical_rate, record_type>;
inline auto make_adaptive_step(
    const std::shared_ptr<operations> &ops, model equation, unsigned int order, double tolerance = 1e-6, double h = .02
)
{
    adaptive_control::params control;
    control.relative_tolerance = tolerance;
    control.absolute_tolerance = tolerance * .01;
    control.initial_step       = h;
    control.maximum_step       = .05;
    adaptive_step::params p;
    p.method        = "BDF";
    p.adaptive      = true;
    p.maximum_order = order;
    return std::make_shared<adaptive_step>(
        ops, std::make_shared<residual_problem>( ops, equation ), std::make_shared<adaptive_control>( ops, control ),
        implicit_flights::make_newton<residual_problem>( ops, make_linear( ops ) ), p
    );
}
}
#endif
