#ifndef PERIODIC_ORBIT_BENCHMARK_BRANCH_ARCHIVE_TRACE_H
#define PERIODIC_ORBIT_BENCHMARK_BRANCH_ARCHIVE_TRACE_H
#include "branch_solver.h"
#include "initial_tangent_solver.h"
#include "../ode_orbit_support.h"
#include <containers/knots.hpp>
#include <continuation/continuation.hpp>
#include <periodic_orbit/orbit_observable.h>
#include <periodic_orbit/periodic_curve_recorder.h>
#include <optional>

namespace orbit_benchmarks
{
struct archive_trace_settings
{
    double lower, upper, ds;
    std::size_t budget;
    double closure_tol;
    unsigned max_steps = 40;
    double matching_tol = 1e-8;
};

template <class Fixture>
auto orbit_measurement( Fixture &check, const typename Fixture::state &point, double lambda,
                        typename Fixture::state &residual, section_tests::X &initial,
                        const archive_trace_settings &settings ) -> std::optional<container::periodic_measurement>
{
    const auto &report = check.op->inspect_crossings( point, lambda, residual );
    if ( !report.succeeded() || !(check.coords->norm( residual ) < settings.matching_tol) )
        return {};
    check.first->lift( point.comp( 0 ), initial );
    auto evaluate = [&]( const section_tests::X &x ) { return check.observable( x ); };
    auto observer = std::make_shared<periodic_orbit::orbit_observable<double, decltype( evaluate )>>( evaluate );
    const double error = ode_orbit_tests::closure_error( check.physical, check.verification_step( lambda ), initial,
                                                         0, report.period, settings.budget, observer );
    if ( !(error < settings.closure_tol) )
        return {};
    const auto q = observer->summary();
    return container::periodic_measurement{ report.period, q.rms, q.minimum, q.maximum, error };
}

template <class Fixture, class Recorder>
auto drive_archive( Fixture &f, branch_corrections<Fixture> &solvers, Recorder &recorder,
                    const typename Fixture::state &z, double p, const archive_trace_settings &settings )
{
    using corrections = branch_corrections<Fixture>;
    using linear_type = initial_tangent_solver<typename Fixture::space, typename Fixture::orbit>;
    using driver_type = continuation::continuation<
        typename Fixture::space, void, typename corrections::alg::log, typename Fixture::orbit, typename Fixture::orbit,
        container::knots<double>, linear_type, typename corrections::fixed_correction_type, Recorder,
        continuation::system_operator_continuation, corrections>;
    linear_type linear( f.coords );
    container::knots<double> knots;
    driver_type driver( f.coords.get(), nullptr, &solvers.log, f.op.get(), f.op.get(), &knots, &linear,
                        &solvers.ordinary, solvers.coupled, solvers.tan );
    driver.set_parameter_bounds( settings.lower, settings.upper );
    driver.set_corrector_controls( 1e-10, false );
    driver.set_steps( settings.max_steps, settings.ds, settings.ds, -1 );
    auto *sink = &recorder;
    return driver.continuate_curve_result( sink, z, p );
}
}
#endif
