#ifndef PERIODIC_ORBIT_ODE_ORBIT_BENCHMARK_H
#define PERIODIC_ORBIT_ODE_ORBIT_BENCHMARK_H

#include "../ode_orbit_support.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace ode_orbit_tests
{
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
        { "period", report.period },
        { "segments", segments }
    };
}

// Sample identical physical times through borrowed dense views. Only three
// scalar plotting observables leave the backend; no trajectory vectors are kept.
template <class Model>
class trajectory_writer
{
    std::shared_ptr<operations>  ops_;
    std::shared_ptr<step<Model>> step_;
    std::ostream                &stream_;
    wrap                         sample_, initial_;
    double                       start_, period_, begin_ = 0, end_ = 0;
    std::size_t                  next_   = 1;
    static constexpr std::size_t samples = 400;
    void                         write( double time, const X &x )
    {
        stream_ << time;
        for ( std::size_t i = 0; i < 3; ++i )
            stream_ << ',' << ops_->get_value_at_point( i, x );
        stream_ << '\n';
        ops_->assign_lin_comb( 1, x, -1, *initial_, *sample_ );
        excursion = std::max( excursion, ops_->norm( *sample_ ) );
    }

public:
    double excursion = 0;
    trajectory_writer(
        std::shared_ptr<operations> ops, std::shared_ptr<step<Model>> step, std::ostream &stream, double start,
        double period, const X &initial
    )
        : ops_( std::move( ops ) ), step_( std::move( step ) ), stream_( stream ), sample_( buffer( ops_ ) ),
          initial_( buffer( ops_ ) ), start_( start ), period_( period )
    {
        ops_->assign( initial, *initial_ );
        write( start, initial );
    }
    void set_time_interval( const double &begin, double &end )
    {
        begin_ = begin;
        end_   = end;
    }
    bool apply( const X &, X & )
    {
        const auto dense = step_->get_continuous_integration();
        while ( next_ <= samples )
        {
            const double time = next_ == samples ? start_ + period_ : start_ + period_ * next_ / samples;
            if ( time > end_ )
                break;
            dense.evaluate( std::max( 0., std::min( 1., ( time - begin_ ) / ( end_ - begin_ ) ) ), *sample_ );
            write( time, *sample_ );
            ++next_;
        }
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
    bool complete() const
    {
        return next_ == samples + 1;
    }
};

template <class Model>
double write_trajectory(
    const fixture<Model> &f, const X &initial, double period, double dt, const std::string &path, double &excursion
)
{
    std::ofstream out( path );
    out.exceptions( std::ios::badbit | std::ios::failbit );
    out << std::setprecision( 17 ) << "time,x,y,z\n";
    auto step        = f.make_stepper( dt );
    auto observer    = std::make_shared<trajectory_writer<Model>>( f.physical, step, out, f.start, period, initial );
    using integrator = nmfd::time_steppers::integration::time_integrator<
        operations, section_tests::step<Model>, trajectory_writer<Model>>;
    integrator run( f.physical, step, typename integrator::params{ f.maximum_steps }, observer );
    auto       end = buffer( f.physical );
    run.set_time_interval( f.start, f.start + period );
    run.apply( initial, *end );
    if ( run.get_status() != integration_status::completed || !observer->complete() )
        throw std::runtime_error( "Incomplete independent trajectory export" );
    excursion = observer->excursion;
    f.physical->add_lin_comb( -1, initial, *end );
    return f.physical->norm( *end );
}

template <class Model>
void write_sections(
    const fixture<Model> &f, const typename fixture<Model>::blocks::vector_type &z, const std::string &path
)
{
    std::ofstream out( path );
    out.exceptions( std::ios::badbit | std::ios::failbit );
    out << std::setprecision( 17 );
    out << "section,time,cx,cy,cz,nx,ny,nz,x,y,z,arrival_x,arrival_y,arrival_z,target_x,target_y,target_z,flight_time,"
           "steps,gap\n";
    auto       initial = buffer( f.physical ), arrival = buffer( f.physical ), target = buffer( f.physical );
    const auto segments = f.make_segments( f.dt );
    for ( std::size_t i = 0; i < f.layout->size(); ++i )
    {
        const auto  next = ( i + 1 ) % f.layout->size();
        const auto &node = f.layout->at( i );
        segments[i].chart->lift( z.comp( i ), *initial );
        segments[next].chart->lift( z.comp( next ), *target );
        periodic_orbit::flight_request<double, plane> request;
        request.start_time                  = node.pilot_time;
        request.end_time                    = node.pilot_time + 2 * node.expected_flight;
        request.target                      = f.layout->at( next ).section;
        request.rule.minimum_crossing_speed = 1e-10;
        const auto flight                   = segments[i].flight->evaluate( *initial, request, *arrival );
        if ( !flight.succeeded() )
            throw std::runtime_error( "ODE section export flight failed" );
        out << i << ',' << node.pilot_time;
        for ( const auto *x : std::array<const X *, 5>{
                  &node.section->center(), &node.section->normal(), &*initial, &*arrival, &*target
              } )
            for ( std::size_t j = 0; j < 3; ++j )
                out << ',' << f.physical->get_value_at_point( j, *x );
        f.physical->add_lin_comb( -1, *arrival, *target );
        out << ',' << flight.flight_time << ',' << flight.steps << ',' << f.physical->norm( *target ) << '\n';
    }
}

template <class Model>
int run_benchmark( int argc, char **argv )
{
    try
    {
        if ( argc != 3 )
            throw std::invalid_argument( "Usage: periodic ODE benchmark config.json output-directory" );
        std::ifstream input( argv[1] );
        const auto    config = json::parse( input );
        const auto    count  = config.at( "sections" ).get<std::size_t>(),
                   budget    = config.at( "maximum_steps" ).get<std::size_t>();
        const auto dt = config.at( "step" ).get<double>(), start = config.at( "start_time" ).get<double>();
        const auto perturbation     = config.at( "perturbation" ).get<double>(),
                   tolerance        = config.at( "matching_tolerance" ).get<double>();
        const auto newton_tolerance = config.at( "newton_tolerance" ).get<double>();
        const auto iterations       = config.at( "newton_max_iterations" ).get<unsigned int>();
        if ( config.at( "system" ) != Model::name || config.at( "backend" ) != backend_name || !count || count > 64 ||
             !budget || !iterations || !std::isfinite( dt ) || dt <= 0 || !std::isfinite( start ) ||
             !std::isfinite( perturbation ) || perturbation <= 0 || perturbation > .1 || !std::isfinite( tolerance ) ||
             tolerance <= 0 || !std::isfinite( newton_tolerance ) || newton_tolerance <= 0 )
            throw std::invalid_argument( "Invalid ODE orbit benchmark configuration" );
        fixture<Model> f( count, dt, config.at( "method" ), start, budget );
        auto           initial = f.block_buffer(), solution = f.block_buffer(), residual = f.block_buffer();
        f.perturb( perturbation, *initial );
        auto       solver = make_newton( f.space, f.problem, newton_tolerance, iterations );
        const auto before = guarded_memory::host_bytes;
        f.problem->apply( *initial, *residual );
        const auto          initial_report = f.problem->report();
        const auto          initial_norm   = f.space->norm( *residual );
        const bool          solved         = solver->solve( *initial, *solution );
        const std::string   directory( argv[2] );
        std::vector<double> history{ initial_norm };
        const auto         &norms = *solver->convergence_strategy()->get_norms_history_handle();
        history.insert( history.end(), norms.begin(), norms.end() );
        std::ofstream convergence( directory + "/newton_history.csv" );
        convergence.exceptions( std::ios::badbit | std::ios::failbit );
        convergence << std::setprecision( 17 ) << "iteration,residual\n";
        for ( std::size_t i = 0; i < history.size(); ++i )
            convergence << i << ',' << history[i] << '\n';
        if ( !solved )
            throw std::runtime_error(
                "NMFD ODE orbit correction failed, status " +
                std::to_string( solver->convergence_strategy()->get_result_status() )
            );
        f.problem->apply( *solution, *residual );
        const auto                             corrected      = f.problem->report();
        const auto                             corrected_norm = f.space->norm( *residual );
        typename fixture<Model>::residual_type refined( f.physical, f.space, f.layout, f.make_segments( dt / 2 ) );
        refined.apply( *solution, *residual );
        const auto refined_report = refined.report();
        const auto refined_norm   = f.space->norm( *residual );
        auto       physical = buffer( f.physical ), rate = buffer( f.physical );
        chart      section( f.physical, f.coordinates, f.layout->at( 0 ).section );
        section.lift( ( *solution ).comp( 0 ), *physical );
        f.rate->evaluate( start, *physical, *rate );
        const auto speed      = f.physical->norm( *rate );
        const auto host_bytes = guarded_memory::host_bytes - before;
        write_sections( f, *initial, directory + "/initial_sections.csv" );
        write_sections( f, *solution, directory + "/corrected_sections.csv" );
        double     excursion = 0, refined_excursion = 0;
        const auto closure =
            write_trajectory( f, *physical, corrected.period, dt / 2, directory + "/trajectory.csv", excursion );
        const auto refined_closure = write_trajectory(
            f, *physical, corrected.period, dt / 4, directory + "/refined_trajectory.csv", refined_excursion
        );
        section.lift( ( *initial ).comp( 0 ), *physical );
        double initial_excursion = 0;
        write_trajectory(
            f, *physical, corrected.period, dt / 2, directory + "/initial_trajectory.csv", initial_excursion
        );
        json parameters = json::object();
        for ( std::size_t i = 0; i < Model::parameters.size(); ++i )
            parameters[Model::parameter_names[i]] = Model::parameters[i];
        const bool passed = corrected_norm <= newton_tolerance && refined_norm < tolerance &&
                            std::abs( corrected.period - Model::period ) < tolerance &&
                            std::abs( refined_report.period - Model::period ) < tolerance && closure < tolerance &&
                            refined_closure < tolerance && speed > 1 && excursion > 1 && host_bytes == 0;
        json report = {
            { "format", "periodic_orbit.ode_newton.v1" },
            { "stage", "periodic_newton" },
            { "orbit_corrected", solved },
            { "configuration", config },
            { "parameters", parameters },
            { "reference_seed", Model::seed },
            { "reference_period", Model::period },
            { "passed", passed },
            { "initial", describe( initial_report ) },
            { "corrected", describe( corrected ) },
            { "refined", describe( refined_report ) },
            { "initial_norm", initial_norm },
            { "corrected_norm", corrected_norm },
            { "refined_norm", refined_norm },
            { "closure_errors", { closure, refined_closure } },
            { "excursion", excursion },
            { "rate_norm", speed },
            { "history", history },
            { "matching_host_bytes", host_bytes },
            { "result_status", solver->convergence_strategy()->get_result_status() }
        };
        std::ofstream output( directory + "/report.json" );
        output.exceptions( std::ios::badbit | std::ios::failbit );
        output << report.dump( 2 ) << '\n';
        std::cout << Model::name << " M=" << count << ": " << ( passed ? "PASS" : "FAIL" )
                  << ", residual=" << corrected_norm << '\n';
        return passed ? 0 : 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 2;
    }
}
}
#endif
