#include "cycle_benchmark_io.h"
#include "../cycle_newton_support.h"
#include "../orbit_method_assembly.h"
using namespace cycle_benchmarks;

// Correct perturbed section radii to the isolated unit cycle with NMFD Newton.
// Independent half/quarter-step integrations validate the recovered physical orbit.
template <class Assembly>
int run_benchmark( const json &config, const std::string &directory )
{
    try
    {
        const auto count              = config.at( "sections" ).get<std::size_t>();
        const auto dt                 = config.at( "step" ).get<double>();
        const auto start              = config.at( "start_time" ).get<double>();
        const auto perturbation       = config.at( "perturbation" ).get<double>();
        const auto tolerance          = config.at( "matching_tolerance" ).get<double>();
        const auto newton_tolerance   = config.at( "newton_tolerance" ).get<double>();
        const auto maximum_iterations = config.at( "newton_max_iterations" ).get<unsigned int>();
        const auto budget             = config.at( "maximum_steps" ).get<std::size_t>();
        if ( config.at( "system" ) != "analytic_cycle_newton" || config.at( "backend" ) != backend_name || !count ||
             count > 64 || !budget || !std::isfinite( dt ) || dt <= 0 || !std::isfinite( start ) ||
             !std::isfinite( perturbation ) || perturbation <= 0 || perturbation > .1 || !std::isfinite( tolerance ) ||
             tolerance <= 0 || !std::isfinite( newton_tolerance ) || newton_tolerance <= 0 )
            throw std::invalid_argument( "Invalid cycle-correction benchmark configuration" );
        using fixture       = basic_fixture<Assembly>;
        using matching      = typename fixture::matching;
        using orbit_problem = periodic_orbit::multiple_section_problem<
            operations, operations, recorded_map_for<Assembly>, cycle_rate, chart, typename Assembly::factory_type>;
        fixture f( count, dt, config.at( "method" ), start, budget );
        auto    problem = std::make_shared<orbit_problem>(
            f.physical, f.space, f.layout, recorded_segments( f ), Assembly::make_factory( f.physical, budget )
        );
        auto solver  = make_newton( f.space, problem, newton_tolerance, maximum_iterations );
        auto initial = block_buffer( f.space ), solution = block_buffer( f.space ), residual = block_buffer( f.space );
        for ( std::size_t i = 0; i < count; ++i )
            f.coordinates->assign_scalar( perturbation * ( i + 1 ) / count, ( *initial ).comp( i ) );
        const auto before = guarded_memory::host_bytes;
        problem->apply( *initial, *residual );
        const auto          initial_report = problem->report();
        const auto          initial_norm   = f.space->norm( *residual );
        const bool          solved         = solver->solve( *initial, *solution );
        std::vector<double> history{ initial_norm };
        const auto         &iterations = *solver->convergence_strategy()->get_norms_history_handle();
        history.insert( history.end(), iterations.begin(), iterations.end() );
        std::ofstream convergence( directory + "/newton_history.csv" );
        convergence.exceptions( std::ios::badbit | std::ios::failbit );
        convergence << std::setprecision( 17 ) << "iteration,residual\n";
        for ( std::size_t i = 0; i < history.size(); ++i )
            convergence << i << ',' << history[i] << '\n';
        if ( !solved )
            throw std::runtime_error(
                "NMFD orbit correction failed with status " +
                std::to_string( solver->convergence_strategy()->get_result_status() )
            );
        problem->apply( *solution, *residual );
        const auto corrected      = problem->report();
        const auto corrected_norm = f.space->norm( *residual );
        matching   refined( f.physical, f.space, f.layout, f.make_segments( dt / 2 ) );
        refined.apply( *solution, *residual );
        const auto refined_report       = refined.report();
        const auto refined_norm         = f.space->norm( *residual );
        const auto numerical_host_bytes = guarded_memory::host_bytes - before;
        write_sections( f, *initial, directory + "/initial_sections.csv" );
        write_sections( f, *solution, directory + "/corrected_sections.csv" );
        auto start_state = buffer( f.physical ), endpoint = buffer( f.physical );
        f.segments[0].chart->lift( ( *solution ).comp( 0 ), *start_state );
        std::vector<double> closure_errors;
        for ( const double factor : { .5, .25 } )
        {
            std::ofstream trajectory( directory + ( factor == .5 ? "/trajectory.csv" : "/refined_trajectory.csv" ) );
            trajectory.exceptions( std::ios::badbit | std::ios::failbit );
            trajectory << std::setprecision( 17 ) << "time,x,y\n";
            auto observer = std::make_shared<trajectory_writer>( trajectory_writer{ f.physical, trajectory, start } );
            observer->write( start, *start_state );
            using integrator = nmfd::time_steppers::integration::time_integrator<
                operations, typename fixture::step_type, trajectory_writer>;
            integrator verify(
                f.physical, f.make_stepper( dt * factor ), typename integrator::params{ budget }, observer
            );
            verify.set_time_interval( start, start + corrected.period );
            verify.apply( *start_state, *endpoint );
            if ( verify.get_status() != integration_status::completed )
                throw std::runtime_error( "Independent orbit verification failed" );
            f.physical->add_lin_comb( -1, *start_state, *endpoint );
            closure_errors.push_back( f.physical->norm( *endpoint ) );
            if ( observer->maximum_error >= tolerance )
                throw std::runtime_error( "Corrected orbit disagrees with analytical cycle" );
        }
        const bool passed = corrected.succeeded() && refined_report.succeeded() && corrected_norm <= newton_tolerance &&
                            refined_norm < tolerance && std::abs( corrected.period - 2 * pi ) < tolerance &&
                            *std::max_element( closure_errors.begin(), closure_errors.end() ) < tolerance &&
                            numerical_host_bytes == 0;
        json report = {
            { "format", "periodic_orbit.newton.v1" },
            { "stage", "periodic_newton" },
            { "orbit_corrected", solved },
            { "configuration", config },
            { "passed", passed },
            { "initial", describe( initial_report ) },
            { "corrected", describe( corrected ) },
            { "refined", describe( refined_report ) },
            { "initial_norm", initial_norm },
            { "corrected_norm", corrected_norm },
            { "refined_norm", refined_norm },
            { "closure_errors", closure_errors },
            { "history", history },
            { "matching_host_bytes", numerical_host_bytes },
            { "result_status", solver->convergence_strategy()->get_result_status() }
        };
        std::ofstream output( directory + "/report.json" );
        output.exceptions( std::ios::badbit | std::ios::failbit );
        output << report.dump( 2 ) << '\n';
        std::cout << "Cycle correction M=" << count << ": " << ( passed ? "PASS" : "FAIL" )
                  << ", residual=" << corrected_norm << '\n';
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
            throw std::invalid_argument( "Usage: analytic_cycle_newton config.json output-directory" );
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
