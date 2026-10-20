#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>

#include <nonlinear_operators/star_shaped/convergence_strategy.h>
#include <nonlinear_operators/star_shaped/linear_operator_star_shaped.h>
#include <nonlinear_operators/star_shaped/preconditioner_star_shaped.h>
#include <nonlinear_operators/star_shaped/star_shaped.h>
#include <nonlinear_operators/star_shaped/system_operator.h>

#include <main/parameters.hpp>
#include <main/stability_continuation.hpp>


#include <stability/eigensolvers/direct_scalar_eigensolver.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

#include "star_shaped_backend_typedefs.h"
#include "star_shaped_linear_solvers.h"

#if defined( STAR_SHAPED_VECTOR_BACKEND_HIP )
#    include <scfd/utils/init_hip.h>
#elif !defined( STAR_SHAPED_VECTOR_BACKEND_OMP )
#    include <scfd/utils/init_cuda.h>
#endif

namespace
{

struct command_line_options
{
    std::string config_file = "json_project_files/star_shaped_test.json";
    bool        quiet       = false;
    bool        edit        = false;
};

command_line_options parse_command_line( int argc, char **argv )
{
    command_line_options result;
    bool                 config_was_set = false;
    for ( int index = 1; index < argc; ++index )
    {
        const std::string argument = argv[index];
        if ( argument == "--quiet" )
            result.quiet = true;
        else if ( argument == "--edit" )
            result.edit = true;
        else if ( !config_was_set )
        {
            result.config_file = argument;
            config_was_set     = true;
        }
        else
        {
            throw std::invalid_argument( "unexpected star-shaped stability argument: " + argument );
        }
    }
    return result;
}

void initialize_device( int pci_id )
{
#if defined( STAR_SHAPED_VECTOR_BACKEND_HIP )
    std::cout << "Using device " << scfd::utils::init_hip( pci_id ) << '\n';
#elif defined( STAR_SHAPED_VECTOR_BACKEND_OMP )
    (void)pci_id;
#else
    std::cout << "Using device " << scfd::utils::init_cuda( pci_id ) << '\n';
#endif
}

void validate_direct_configuration( const stability::analysis::matrix_free_stability_config<real> &config )
{
    if ( !config.enabled )
    {
        throw std::invalid_argument( "star-shaped stability eigensolver is disabled" );
    }
    if ( !std::isfinite( config.linearization_scale ) || config.linearization_scale == real{} )
    {
        throw std::invalid_argument(
            "star-shaped stability linearization scale must be "
            "finite and nonzero"
        );
    }
}

} // namespace

int main( int argc, char **argv )
{
    try
    {
        const auto command_line = parse_command_line( argc, argv );
        using parameters_type   = main_classes::parameters<real>;
        auto configuration =
            std::make_shared<parameters_type>( main_classes::read_parameters_json<real>( command_line.config_file ) );
        auto       &parameters = *configuration;
        const auto &config     = parameters.stability_continuation.matrix_free_eigensolver;
        validate_direct_configuration( config );
        if ( parameters.nonlinear_operator.N_size.size() != 1 || parameters.nonlinear_operator.N_size.front() != 1 )
        {
            throw std::invalid_argument( "star-shaped stability requires vector dimension one" );
        }
        initialize_device( parameters.nvidia_pci_id );

        real curvature = real( 0.2 );
        if ( !parameters.nonlinear_operator.problem_real_parameters_vector.empty() )
        {
            curvature = parameters.nonlinear_operator.problem_real_parameters_vector.front();
        }

        using file_operations_type    = gpu_file_operations<vec_ops_real>;
        using nonlinear_operator_type = nonlinear_operators::star_shaped<vec_ops_real, Blocks_x_>;
        using newton_operator_type =
            nonlinear_operators::linear_operator_star_shaped<vec_ops_real, nonlinear_operator_type>;
        using newton_preconditioner_type = nonlinear_operators::preconditioner_star_shaped<
            vec_ops_real, nonlinear_operator_type, newton_operator_type>;
        using stability_operator_type =
            stability::eigensolvers::transformations::scaled_real_operator<vec_ops_real, newton_operator_type>;
        using eigensolver_type =
            stability::eigensolvers::direct_scalar_eigensolver<vec_ops_real, stability_operator_type>;
        using monitor_type = star_shaped_model::monitor<vec_ops_real, log_t>;
        using driver_type  = main_classes::stability_continuation<
             vec_ops_real, file_operations_type, log_t, monitor_type, nonlinear_operator_type, newton_operator_type,
             newton_preconditioner_type, star_shaped_model::direct_solver, nonlinear_operators::system_operator,
             parameters_type, eigensolver_type>;

        auto  real_space   = std::make_shared<vec_ops_real>( 1 );
        auto &vector_space = *real_space;
        if ( parameters.use_high_precision_reduction )
            vector_space.use_high_precision();
        auto                    file_operations = std::make_shared<file_operations_type>( real_space.get() );
        auto                    nonlinear_owner = std::make_shared<nonlinear_operator_type>( 1, real_space, curvature );
        newton_operator_type    newton_operator( nonlinear_owner );
        stability_operator_type stability_operator( vector_space, newton_operator, config.linearization_scale );
        auto                    eigensolver = std::make_shared<eigensolver_type>( vector_space, stability_operator );
        log_t                   log;
        log_t                   linear_solver_log;
        log.set_verbosity( command_line.quiet ? 0 : 1 );
        linear_solver_log.set_verbosity( command_line.quiet ? 0 : 1 );

        if ( !command_line.quiet )
        {
            std::cout << "Using star-shaped stability backend: " << STAR_SHAPED_BACKEND_NAME << '\n';
            parameters.plot_all();
        }

        driver_type driver(
            real_space, file_operations, &log, &linear_solver_log, nonlinear_owner, configuration, eigensolver
        );
        driver.set_parameters();
        if ( command_line.edit )
            driver.edit();
        else
            driver.execute();
    }
    catch ( const std::exception &error )
    {
        std::cerr << "Star-shaped stability failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
