#include "cycle_benchmark_io.h"
#include "../prepared_cycle_support.h"
#include "../orbit_method_assembly.h"
using namespace cycle_benchmarks;

template <class Assembly>
int run_benchmark( const json &config, const std::string &directory )
{
    try
    {
        const auto count        = config.at( "sections" ).get<std::size_t>();
        const auto dt           = config.at( "step" ).get<double>();
        const auto start        = config.at( "start_time" ).get<double>();
        const auto perturbation = config.at( "perturbation" ).get<double>();
        const auto tolerance    = config.at( "matching_tolerance" ).get<double>();
        const auto budget       = config.at( "maximum_steps" ).get<std::size_t>();
        const auto method       = config.at( "method" ).get<std::string>();
        if ( config.at( "system" ) != "analytic_cycle" || config.at( "backend" ) != backend_name || !count ||
             count > 64 || !budget || !std::isfinite( dt ) || dt <= 0 || !std::isfinite( start ) ||
             !std::isfinite( perturbation ) || perturbation <= 0 || perturbation > .1 || !std::isfinite( tolerance ) ||
             tolerance <= 0 )
            throw std::invalid_argument( "Invalid analytical-cycle benchmark configuration" );
        using fixture  = basic_fixture<Assembly>;
        using matching = typename fixture::matching;
        fixture  f( count, dt, method, start, budget );
        matching residual( f.physical, f.space, f.layout, f.segments );
        auto     z = block_buffer( f.space ), r = block_buffer( f.space );
        f.space->assign_scalar( 0, *z );
        const auto before = guarded_memory::host_bytes;
        const auto base   = residual.evaluate( *z, *r );
        if ( !base.succeeded() )
            throw std::runtime_error( "Cycle matching failed" );
        const auto base_norm = f.space->norm( *r );
        matching   reference( f.physical, f.space, f.layout, f.make_segments( dt / 2 ) );
        const auto refined = reference.evaluate( *z, *r );
        if ( !refined.succeeded() )
            throw std::runtime_error( "Refined cycle matching failed" );
        const auto refined_norm         = f.space->norm( *r );
        const auto numerical_host_bytes = guarded_memory::host_bytes - before;
        write_sections( f, *z, directory + "/sections.csv" );
        for ( std::size_t i = 0; i < count; ++i )
            f.coordinates->assign_scalar( perturbation * ( i + 1 ) / count, ( *z ).comp( i ) );
        const auto perturbed_before = guarded_memory::host_bytes;
        const auto perturbed        = residual.evaluate( *z, *r );
        if ( !perturbed.succeeded() )
            throw std::runtime_error( "Perturbed cycle flight failed" );
        const auto perturbed_norm       = f.space->norm( *r );
        const auto perturbed_host_bytes = guarded_memory::host_bytes - perturbed_before;
        write_sections( f, *z, directory + "/perturbed_sections.csv" );

        std::ofstream trajectory( directory + "/trajectory.csv" );
        trajectory.exceptions( std::ios::badbit | std::ios::failbit );
        trajectory << std::setprecision( 17 ) << "time,x,y\n";
        auto observer = std::make_shared<trajectory_writer>( trajectory_writer{ f.physical, trajectory, start } );
        auto endpoint = buffer( f.physical );
        observer->write( start, f.layout->at( 0 ).section->center() );
        using integrator = nmfd::time_steppers::integration::time_integrator<
            operations, typename fixture::step_type, trajectory_writer>;
        integrator verification(
            f.physical, f.make_stepper( dt / 2 ), typename integrator::params{ budget }, observer
        );
        verification.set_time_interval( start, start + 2 * pi );
        verification.apply( f.layout->at( 0 ).section->center(), *endpoint );
        f.physical->add_lin_comb( -1, f.layout->at( 0 ).section->center(), *endpoint );
        const auto closure = f.physical->norm( *endpoint );
        const bool passed  = verification.get_status() == integration_status::completed && base_norm < tolerance &&
                            refined_norm < tolerance && std::abs( base.period - 2 * pi ) < tolerance &&
                            std::abs( refined.period - 2 * pi ) < tolerance && perturbed_norm > perturbation * .01 &&
                            closure < tolerance && observer->maximum_error < tolerance && numerical_host_bytes == 0 &&
                            perturbed_host_bytes == 0;
        json report = {
            { "format", "periodic_orbit.matching.v1" },
            { "stage", "matching_only" },
            { "orbit_corrected", false },
            { "configuration", config },
            { "passed", passed },
            { "baseline", describe( base ) },
            { "refined", describe( refined ) },
            { "perturbed", describe( perturbed ) },
            { "baseline_norm", base_norm },
            { "refined_norm", refined_norm },
            { "perturbed_norm", perturbed_norm },
            { "closure_error", closure },
            { "trajectory_error", observer->maximum_error },
            { "trajectory_rows", observer->rows },
            { "matching_host_bytes", numerical_host_bytes + perturbed_host_bytes }
        };
        std::ofstream output( directory + "/report.json" );
        output.exceptions( std::ios::badbit | std::ios::failbit );
        output << report.dump( 2 ) << '\n';
        std::cout << "Analytical cycle, M=" << count << ": " << ( passed ? "PASS" : "FAIL" )
                  << ", matching=" << base_norm << ", perturbed=" << perturbed_norm << '\n';
        return passed ? 0 : 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 2;
    }
}

int main( int argc, char **argv )
{
    try
    {
        if ( argc != 3 )
            throw std::invalid_argument( "Usage: analytic_cycle config.json output-directory" );
        std::ifstream input( argv[1] );
        const auto    config = json::parse( input );
        return orbit_benchmarks::dispatch<differentiable_cycle>( config.at( "method" ), [&]( auto assembly ) {
            return run_benchmark<decltype( assembly )>( config, argv[2] );
        } );
    }
    catch ( const std::exception &e )
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 2;
    }
}
