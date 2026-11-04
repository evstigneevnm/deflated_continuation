#ifndef PERIODIC_ORBIT_CYCLE_BENCHMARK_IO_H
#define PERIODIC_ORBIT_CYCLE_BENCHMARK_IO_H

#include "../analytic_cycle_support.h"
#include <array>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace cycle_benchmarks
{
using namespace cycle_tests;
using json               = nlohmann::json;
using integration_status = nmfd::time_steppers::integration_status;
#if defined( TEST_VECTOR_BACKEND_CUDA )
inline constexpr const char *backend_name = "cuda";
#elif defined( TEST_VECTOR_BACKEND_HIP )
inline constexpr const char *backend_name = "hip";
#elif defined( TEST_VECTOR_BACKEND_OMP )
inline constexpr const char *backend_name = "cpu_omp";
#else
inline constexpr const char *backend_name = "cpu";
#endif

inline json describe( const periodic_orbit::matching_report<double> &report )
{
    json segments = json::array();
    for ( const auto &s : report.segments )
        segments.push_back(
            { { "flight_time", s.flight.flight_time },
              { "steps", s.flight.steps },
              { "crossing_speed", s.flight.crossing_speed },
              { "gap_norm", s.gap_norm },
              { "section_value", s.arrival_section_value } }
        );
    return {
        { "succeeded", report.succeeded() },
        { "status", int( report.status ) },
        { "failed_section", report.failed_section },
        { "period", report.period },
        { "segments", segments }
    };
}

// Export scalar observables outside matching and Krylov, without retaining a
// trajectory of backend state vectors or changing pending integration endpoints.
struct trajectory_writer
{
    std::shared_ptr<operations> space;
    std::ostream               &stream;
    double                      start, time = 0, maximum_error = 0;
    std::size_t                 rows = 0;
    void                        write( double t, const X &state )
    {
        const auto x = space->get_value_at_point( 0, state ), y = space->get_value_at_point( 1, state );
        maximum_error = std::max( maximum_error, std::hypot( x - std::cos( t - start ), y - std::sin( t - start ) ) );
        stream << t << ',' << x << ',' << y << '\n';
        ++rows;
    }
    void set_time_interval( const double &, double &end )
    {
        time = end;
    }
    bool apply( const X &, X &out )
    {
        write( time, out );
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

inline void write_sections( fixture &f, const blocks::vector_type &z, const std::string &path )
{
    std::ofstream out( path );
    out.exceptions( std::ios::badbit | std::ios::failbit );
    out << std::setprecision( 17 )
        << "section,time,cx,cy,nx,ny,x,y,arrival_x,arrival_y,target_x,target_y,flight_time,steps,gap\n";
    auto initial = buffer( f.physical ), arrival = buffer( f.physical ), target = buffer( f.physical );
    for ( std::size_t i = 0; i < f.layout->size(); ++i )
    {
        const auto &node = f.layout->at( i );
        const auto  next = ( i + 1 ) % f.layout->size();
        f.segments[i].chart->lift( z.comp( i ), *initial );
        f.segments[next].chart->lift( z.comp( next ), *target );
        periodic_orbit::flight_request<double, plane> request;
        request.start_time                  = node.pilot_time;
        request.end_time                    = node.pilot_time + 2 * node.expected_flight;
        request.target                      = f.layout->at( next ).section;
        request.rule.minimum_crossing_speed = 1e-10;
        const auto flight                   = f.segments[i].flight->evaluate( *initial, request, *arrival );
        if ( !flight.succeeded() )
            throw std::runtime_error( "Benchmark export replay failed" );
        out << i << ',' << node.pilot_time;
        for ( const auto *state : std::array<const X *, 5>{
                  &node.section->center(), &node.section->normal(), &*initial, &*arrival, &*target
              } )
            out << ',' << f.physical->get_value_at_point( 0, *state ) << ','
                << f.physical->get_value_at_point( 1, *state );
        f.physical->add_lin_comb( -1, *arrival, *target );
        out << ',' << flight.flight_time << ',' << flight.steps << ',' << f.physical->norm( *target ) << '\n';
    }
}
}
#endif
