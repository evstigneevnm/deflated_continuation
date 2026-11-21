#ifndef PERIODIC_ORBIT_BENCHMARK_ODE_CONTINUATION_BENCHMARK_H
#define PERIODIC_ORBIT_BENCHMARK_ODE_CONTINUATION_BENCHMARK_H

#include "ode_continuation_fixture.h"
#include <periodic_orbit/sections/layout_health.h>
#include <periodic_orbit/matching/refresh_section_branch.h>
#include <periodic_orbit/orbit_observable.h>
#include <containers/bifurcation_diagram/observable_column.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <optional>
#include <type_traits>

namespace ode_orbit_tests
{
using json = nlohmann::json;
struct branch_reference
{
    double parameter, period;
};

template <std::size_t N>
std::array<double, N> origin_reference( double ) { return {}; }

// Effective settings come from the system's preset, with the runner's CLI overrides applied.
struct branch_settings
{
    json        config;
    std::string dir;
    std::size_t sections, budget;
    double      step, start, arclength_step, newton_tolerance, matching_tolerance;
    unsigned    iterations, steps;

    static branch_settings read( int argc, char **argv, const char *system )
    {
        require( argc == 3, "Expected config.json and output directory" );
        json c;
        std::ifstream( argv[1] ) >> c;
        require( c.at( "system" ) == system, "Continuation config does not match the executable's system" );
        require(
            c.at( "mode" ) == "continuation" && c.at( "method" ) == "RK45" && c.at( "branch_steps" ).get<int>() >= 2,
            "Unsupported continuation configuration"
        );
        return {
            c,
            argv[2],
            c.at( "sections" ),
            c.at( "maximum_steps" ),
            c.at( "step" ),
            c.at( "start_time" ),
            c.at( "arclength_step" ),
            c.at( "newton_tolerance" ),
            c.at( "matching_tolerance" ),
            c.at( "newton_max_iterations" ),
            c.at( "branch_steps" )
        };
    }
};

template <class MakeFlights, class MakeSolver>
struct branch_case
{
    using fixture_type = std::invoke_result_t<MakeFlights &>;
    using state        = typename fixture_type::state;
    using condition    = periodic_orbit::layout_condition;
    struct epoch
    {
        fixture_type                                       flights;
        std::invoke_result_t<MakeSolver &, fixture_type &> solver;
        epoch( fixture_type f, MakeSolver &make ) : flights( std::move( f ) ), solver( make( flights ) )
        {
        }
        epoch( const epoch & ) = delete;
    };
    MakeFlights            make_flights;
    MakeSolver             make_solver;
    epoch                  initial{ make_flights(), make_solver };
    std::unique_ptr<epoch> replacement;
    epoch                 &active()
    {
        return replacement ? *replacement : initial;
    }
    const branch_settings                       &options;
    continuation::corrector_retry_policy<double> policy;
    json                                         probes          = json::array();
    json                                         refreshes       = json::array();
    json                                         reference_point = nullptr;
    json                                         replay_check = nullptr;
    double                                       restart_error = 0, parameter_column_error = 0;
    std::size_t                                  export_bytes = 0;
    using steady_reference = std::function<std::optional<std::array<double, fixture_type::dimension>>( double )>;
    wrap                                         steady_state = buffer( initial.flights.physical );
    steady_reference                             equilibrium;
    std::ofstream                                csv, steady;
    branch_case( MakeFlights flights, MakeSolver newton, const branch_settings &settings, steady_reference ref )
        : make_flights( flights ), make_solver( newton ), options( settings ), equilibrium( std::move( ref ) ),
          csv( options.dir + "/branch.csv" ), steady( options.dir + "/steady_reference.csv" )
    {
        csv.exceptions( std::ios::failbit | std::ios::badbit );
        steady.exceptions( std::ios::failbit | std::ios::badbit );
        steady << std::setprecision( 17 ) << "direction,index,parameter,value\n";
        csv << std::setprecision( 17 ) << "direction,index,parameter,period,residual,max_gap,closure,ds,retries,"
                                         "observable_rms,observable_min,observable_max\n";
        policy.success_growth_factor = 1;
        policy.minimum_step_size     = options.arclength_step * 1e-6;
    }
    branch_case( const branch_case & ) = delete; // The solver refers to this instance's flights.

    template <std::size_t N>
    std::array<double, N> checkpoint_state( const X &x )
    {
        require( x.size() == N, "Checkpoint vector dimension mismatch" );
        std::array<double, N> values;
        // Explicit persistence boundary. Keep the solve's transfer guard enabled.
        native_backend::memory_type::copy_to_host( N * sizeof(double), x.raw_ptr(), values.data() );
        export_bytes += N * sizeof(double);
        return values;
    }
    template <std::size_t N>
    void restore_state( const json &saved, X &x )
    {
        require( x.size() == N && saved.size() == N, "Checkpoint vector dimension mismatch" );
        const auto values = saved.get<std::array<double, N>>();
        native_backend::memory_type::copy_from_host( N * sizeof(double), values.data(), x.raw_ptr() );
        export_bytes += N * sizeof(double);
    }
    json encode( const state &z )
    {
        json  saved = json::array();
        for ( std::size_t i = 0; i < options.sections; ++i )
            saved.push_back( checkpoint_state<fixture_type::dimension - 1>( z.comp( i ) ) );
        return saved;
    }
    void decode( const json &saved, state &z )
    {
        require( saved.size() == z.size(), "Checkpoint section count mismatch" );
        for ( std::size_t i = 0; i < z.size(); ++i )
            restore_state<fixture_type::dimension - 1>( saved.at( i ), z.comp( i ) );
    }
    void check_restart(
        const state &z, double p, const state &v, double vp, double next_ds, const state &next, double q,
        const state &nv, double vq, int direction, bool refreshed
    )
    {
        auto      &s          = active().flights;
        const auto path       = options.dir + ( refreshed ? "/refresh_checkpoint.json" : "/checkpoint.json" );
        json       checkpoint = {
            { "configuration", options.config },
            { "pilot_indices", s.pilot_indices },
            { "direction", direction },
            { "pilot_parameter", s.pilot_parameter },
            { "pilot_period", s.period },
            { "pilot_seed", checkpoint_state<fixture_type::dimension>( s.layout->at( 0 ).section->center() ) },
            { "state", encode( z ) },
            { "tangent", encode( v ) },
            { "parameter", p },
            { "parameter_tangent", vp },
            { "next_ds", next_ds }
        };
        {
            std::ofstream stream( path );
            stream.exceptions( std::ios::badbit | std::ios::failbit );
            stream << checkpoint.dump( 2 );
        }
        std::ifstream( path ) >> checkpoint;
        auto seed = buffer( s.physical );
        restore_state<fixture_type::dimension>( checkpoint.at( "pilot_seed" ), *seed );
        fixture_type fresh(
            s, *seed, checkpoint.at( "pilot_parameter" ), checkpoint.at( "pilot_period" ), options.step, options.start,
            options.budget
        );
        require(
            checkpoint.at( "configuration" ) == options.config &&
                checkpoint.at( "pilot_indices" ) == fresh.pilot_indices && checkpoint.at( "direction" ) == direction,
            "Restart does not reproduce pilot layout/direction"
        );
        auto resumed = make_solver( fresh );
        resumed.pred.set_steps( checkpoint.at( "next_ds" ).get<double>(), options.arclength_step, policy );
        auto rz = fresh.workspace(), rv = fresh.workspace(), out = fresh.workspace(), ov = fresh.workspace();
        decode( checkpoint.at( "state" ), *rz );
        decode( checkpoint.at( "tangent" ), *rv );
        double rp = checkpoint.at( "parameter" ), rvp = checkpoint.at( "parameter_tangent" ), a = 0, b = 0;
        require(
            resumed.driver.solve( fresh.op.get(), *rz, rp, *rv, rvp, *out, a, *ov, b ), "Restart corrector failed"
        );
        fresh.coords->add_lin_comb( -1, next, *out );
        fresh.coords->add_lin_comb( -1, nv, *ov );
        restart_error = std::max(
            restart_error,
            fresh.coords->norm( *out ) + fresh.coords->norm( *ov ) + std::abs( a - q ) + std::abs( b - vq )
        );
        require( restart_error < options.newton_tolerance, "Restart differs from uninterrupted branch" );
    }
    void check_parameter_column( const state &anchor, double p, bool verify_replay )
    {
        auto &s = active().flights;
        // Hold section coordinates fixed. Each perturbed flight must locate its
        // own crossing; differentiating at a fixed arrival time is insufficient.
        constexpr double eps = 2e-5;
        // Pilot sections may be centered exactly on RK grid nodes at the
        // anchor. Probe off the node to test one smooth dense-output interval.
        if ( verify_replay )
            p += 8 * eps;
        auto             col = s.workspace(), plus = s.workspace(), minus = s.workspace();
        s.op->set_linearization_point( anchor, p );
        s.op->jacobian_alpha( *col );
        const auto base = s.op->report();
        const auto partial = s.op->period_parameter_partial();
        auto prepared = s.op->get_jacobi_operator();
        auto direction = s.workspace(), before = s.workspace(), after = s.workspace();
        if ( verify_replay )
        {
            s.coords->assign_scalar( .1, *direction );
            prepared->apply( *direction, *before );
        }
        s.op->F( anchor, p + eps, *plus );
        const auto positive = s.op->report();
        s.op->F( anchor, p - eps, *minus );
        const auto &negative = s.op->report();
        if ( verify_replay )
        {
            // Fixed h and identical pilot start times: only the location within
            // the last RK step may differ between the two perturbed crossings.
            for ( std::size_t i = 0; i < base.segments.size(); ++i )
                require( base.segments[i].flight.steps == positive.segments.at(i).flight.steps &&
                         base.segments[i].flight.steps == negative.segments.at(i).flight.steps,
                         "Parameter perturbation changed the frozen RK mesh" );
            const double fd_time = ( positive.period - negative.period ) / ( 2 * eps );
            const double time_error = std::abs( partial - fd_time );
            require( time_error < 5e-8 + 2e-5 * std::abs( partial ), "Incorrect return-time parameter partial" );
            const auto allocations = guarded_memory::allocation_calls;
            prepared->apply( *direction, *after );
            require( guarded_memory::allocation_calls == allocations, "Repeated matching Jv allocated a vector" );
            s.coords->add_lin_comb( -1, *before, *after );
            const double difference = s.coords->norm( *after );
            require( difference < 1e-12, "Trial parameter invalidated prepared matching action" );
            replay_check = { { "parameter", p }, { "period_partial", partial }, { "finite_difference", fd_time },
                             { "prepared_action_difference", difference } };
        }
        s.coords->assign_lin_comb( .5 / eps, *plus, -.5 / eps, *minus, *plus );
        const double size = s.coords->norm( *plus );
        s.coords->add_lin_comb( -1, *col, *plus );
        parameter_column_error = s.coords->norm( *plus ) / std::max( 1., size );
        require( size > 1e-5 && parameter_column_error < 2e-5, "Incorrect orbit parameter derivative" );
    }
    struct measurement
    {
        double    period, norm, gap, closure;
        condition health;
        periodic_orbit::orbit_observable_summary<double> observable;
    };
    measurement measure( fixture_type &s, const state &z, double p, double previous_period )
    {
        auto        residual = s.workspace();
        const auto &report   = s.op->inspect_crossings( z, p, *residual );
        if ( !report.succeeded() )
            throw periodic_orbit::matching_evaluation_error<double>( report );
        double norm = s.coords->norm( *residual ), gap = 0;
        for ( const auto &segment : report.segments )
            gap = std::max( gap, segment.gap_norm );
        auto initial = buffer( s.physical );
        s.first->lift( z.comp( 0 ), *initial );
        auto closure_at = [&]( double period ) {
            return closure_error(
                s.physical, s.verification_step( p ), *initial,
                options.start, period, options.budget
            );
        };
        auto evaluate = [&]( const X &x ) { return s.observable( x ); };
        auto observer = std::make_shared<periodic_orbit::orbit_observable<double, decltype( evaluate )>>( evaluate );
        const double closure = closure_error(
            s.physical, s.verification_step( p ), *initial,
            options.start, report.period, options.budget, observer
        );
        require(
            report.succeeded() && report.period > 0 && norm <= options.newton_tolerance &&
                gap < options.matching_tolerance && closure < options.matching_tolerance,
            "Branch point failed matching or independent closure"
        );
        return {
            report.period, norm, gap, closure,
            periodic_orbit::layout_health<double>{}.inspect(
                *s.layout, s.period, previous_period, report,
                [&]( double t ) { return closure_at( t ) < options.matching_tolerance; }
            ), observer->summary()
        };
    }
    void record( const measurement &m, int sign, unsigned index, double p, double used, unsigned retries )
    {
        csv << sign << ',' << index << ',' << p << ',' << m.period << ',' << m.norm << ',' << m.gap << ',' << m.closure
            << ',' << used << ',' << retries << ',' << m.observable.rms << ',' << m.observable.minimum << ','
            << m.observable.maximum << std::endl;
        if ( const auto ref = equilibrium ? equilibrium( p ) : std::nullopt )
        {
            fill_profile( *steady_state, *ref );
            steady << sign << ',' << index << ',' << p << ',' << active().flights.observable( *steady_state ) << '\n';
        }
    }
    auto corrected_sample( const state &z, double p, const state &next, double q, double target )
    {
        auto &s      = active().flights;
        auto &solver = active().solver;
        // Compare section counts at identical physical parameters, not at equal arclength indices.
        auto   at = s.workspace(), residual = s.workspace();
        double a = ( target - p ) / ( q - p );
        s.coords->assign_lin_comb( 1 - a, z, a, next, *at );
        require( solver.ordinary.solve( s.op.get(), *at, target ), "Common-parameter correction failed" );
        s.op->F( *at, target, *residual );
        require( s.coords->norm( *residual ) <= options.newton_tolerance, "Common-parameter residual" );
        return at;
    }
    void probe( const state &z, double p, const state &next, double q, double target )
    {
        auto &s       = active().flights;
        auto  at      = corrected_sample( z, p, next, q, target );
        auto  initial = buffer( s.physical );
        s.first->lift( ( *at ).comp( 0 ), *initial );
        const auto m = measure( s, *at, target, s.op->report().period );
        probes.push_back(
            { { "parameter", target },
              { "period", m.period }, { "observable_rms", m.observable.rms },
              { "state", checkpoint_state<fixture_type::dimension>( *initial ) } }
        );
    }
    auto correct_anchor( double p0, bool verify_replay = false )
    {
        auto &s      = initial.flights;
        auto &solver = initial.solver;
        auto  anchor = s.workspace();
        perturb_sections<fixture_type::dimension>( options.config.at( "perturbation" ), *anchor );
        require( solver.ordinary.solve( s.op.get(), *anchor, p0 ), "Anchor correction failed" );
        auto m = measure( s, *anchor, p0, s.period );
        require(
            std::abs( m.period - s.period ) < options.matching_tolerance && m.health == condition::healthy,
            "Incorrect anchor layout"
        );
        record( m, 0, 0, p0, 0, 0 );
        check_parameter_column( *anchor, p0, verify_replay );
        return anchor;
    }
    void refresh(
        state &z, double p, double period, state &previous, double prev_p, double prev_period, state &v, double &vp,
        double ds
    )
    {
        auto &old   = active().flights;
        replacement = periodic_orbit::refresh_section_branch(
            old, z, p, previous, prev_p, prev_period, v, vp, options.matching_tolerance,
            [&]( const X &seed ) {
                auto e = std::make_unique<epoch>(
                    fixture_type( old, seed, p, period, options.step, options.start, options.budget ), make_solver
                );
                e->solver.pred.set_steps( ds, options.arclength_step, policy );
                return e;
            },
            [&]( auto &f, const state &root ) {
                auto m = measure( f, root, p, period );
                return m.health == condition::healthy && std::abs( m.period - period ) < options.matching_tolerance;
            },
            [&]( auto &f, X &prior, state &before ) {
                return f.rechart_previous( prior, prev_p, prev_period, options.step, options.budget, before );
            }
        );
    }
    void trace( const state &anchor, double p0, int sign, std::optional<branch_reference> reference = {} )
    {
        replacement.reset();
        auto  &s = initial.flights;
        auto   a = s.workspace(), b = s.workspace(), c = s.workspace(), d = s.workspace(), e = s.workspace();
        state *z = &*a, *previous = &*b, *next = &*c, *v = &*d, *nv = &*e;
        s.coords->assign( anchor, *z );
        double p = p0, vp = 0, q = 0, vq = 0;
        initial.solver.pred.set_steps( options.arclength_step, options.arclength_step, policy );
        require( initial.solver.tan->initial( *z, p, sign, *v, vp ), "Initial tangent failed" );
        const double target  = p0 + sign * options.config.at( "probe_offset" ).get<double>();
        bool         sampled = false;
        double       period = s.period, prev_period = 0, prev_p = p;
        bool         pending = false, rebuilt = false, checked_refresh_restart = false;
        const bool   observe_only = options.config.value( "observe_layout", false );
        std::string  reason;
        double       retry_ds = options.arclength_step;
        for ( unsigned i = 1; i <= options.steps; )
        {
            if ( pending )
            {
                require( !rebuilt && i > 1, "Cannot refresh the same accepted point twice or without a secant" );
                refresh( *z, p, period, *previous, prev_p, prev_period, *v, vp, retry_ds );
                refreshes.push_back(
                    { { "direction", sign }, { "index", i - 1 }, { "parameter", p }, { "reason", reason } }
                );
                pending = false;
                rebuilt = true;
            }
            auto        &solver         = active().solver;
            auto        &f              = active().flights;
            const double next_ds        = solver.pred.get_ds();
            solver.coupled->last_failure = {};
            try
            {
                require(
                    solver.driver.solve( f.op.get(), *z, p, *v, vp, *next, q, *nv, vq ), "Arclength correction failed"
                );
            }
            catch ( const std::runtime_error & )
            {
                const auto failure = solver.coupled->last_failure;
                if ( !failure.layout_related() || rebuilt || i == 1 || observe_only )
                    throw;
                reason = "failed flight on section " + std::to_string( failure.section ) + " (status " +
                         std::to_string( static_cast<int>( failure.status ) ) + ")";
                retry_ds = next_ds;
                pending  = true;
                continue;
            }
            const auto attempt = solver.driver.last_attempt_state();
            require( i != 1 || sign * ( q - p ) > 0, "Incorrect initial branch direction" );
            auto m = measure( f, *next, q, period );
            reason = periodic_orbit::name( m.health );
            if ( m.health != condition::healthy )
                std::cout << "Layout: " << reason << ", parameter=" << std::setprecision( 17 ) << q
                          << ", period=" << m.period << '\n';
            if ( m.health == condition::repeated_circuit )
            {
                require( !observe_only && !rebuilt && i > 1, "Candidate repeated a circuit; not committed" );
                retry_ds = next_ds;
                pending  = true;
                continue;
            }
            require( m.health != condition::slow_flow, "No regular transverse flow at the candidate" );
            if ( reference && q != p && ( p - reference->parameter ) * ( q - reference->parameter ) <= 0 )
            {
                auto       at     = corrected_sample( *z, p, *next, q, reference->parameter );
                const auto sample = measure( f, *at, reference->parameter, period );
                require(
                    sample.health != condition::repeated_circuit &&
                        std::abs( sample.period - reference->period ) < options.matching_tolerance,
                    "Reference orbit did not close in the expected single period"
                );
                reference_point = {
                    { "parameter", reference->parameter }, { "period", sample.period }, { "closure", sample.closure }
                };
                reference.reset();
            }
            if ( !sampled && ( target - p ) * ( target - q ) <= 0 && q != p )
            {
                probe( *z, p, *next, q, target );
                sampled = true;
            }
            if ( ( sign == 1 && i == options.steps / 2 + 1 ) || ( rebuilt && !checked_refresh_restart ) )
            {
                check_restart( *z, p, *v, vp, next_ds, *next, q, *nv, vq, sign, rebuilt );
                checked_refresh_restart |= rebuilt;
            }
            record( m, sign, i, q, attempt.attempted_step, attempt.retry_count );
            prev_p      = p;
            prev_period = period;
            period      = m.period;
            std::swap( previous, z );
            std::swap( z, next );
            std::swap( v, nv );
            p        = q;
            vp       = vq;
            pending  = !observe_only && m.health != condition::healthy;
            retry_ds = solver.pred.get_ds();
            rebuilt  = false;
            ++i;
        }
        require( sampled, "Branch did not bracket common parameter; increase --branch-steps or reduce --probe-offset" );
        if ( sign == -1 && options.config.value( "require_reference", false ) )
            require( !reference_point.is_null(), "Branch did not reach the reference parameter" );
    }
    void write_report() const
    {
        require(
            guarded_memory::host_bytes == 0, "Continuation transferred vectors outside checkpoint I/O"
        );
        json result = {
            { "format", "periodic_orbit.continuation.v2" },
            { "observable", fixture_type::observable_label },
            { "configuration", options.config },
            { "probes", probes },
            { "layout_refreshes", refreshes },
            { "reference_point", reference_point },
            { "restart_error", restart_error },
            { "parameter_column_error", parameter_column_error },
            { "replay_check", replay_check },
            { "diagnostic_host_bytes", export_bytes },
            { "core_host_bytes", guarded_memory::host_bytes },
            { "passed", true }
        };
        std::ofstream stream( options.dir + "/report.json" );
        stream.exceptions( std::ios::failbit | std::ios::badbit );
        stream << result.dump( 2 ) << '\n';
    }
};

}
#endif
