#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <main/parameters.hpp>
#include <visualization/bd_prepare_visualization.hpp>

#include "circle_backend_typedefs.h"

int main( int argc, char const *argv[] )
{
    using files_ops_t = gpu_file_operations<vec_ops_real>;

#if defined( CIRCLE_VECTOR_BACKEND_VAR_PREC )
    using parameters_real = double;
    using parameters_t    = main_classes::parameters<double>;
#else
    using parameters_real = real;
    using parameters_t    = main_classes::parameters<real>;
#endif

    std::string                       path_to_config_file = "json_project_files/circle_test.json";
    visualization::bd_prepare_options options;

    try
    {
        for ( int argi = 1; argi < argc; ++argi )
        {
            const std::string arg = argv[argi];
            if ( visualization::parse_common_prepare_argument( arg, argi, argc, argv, options ) )
            {
                continue;
            }
            if ( path_to_config_file == "json_project_files/circle_test.json" )
            {
                path_to_config_file = arg;
            }
            else
            {
                visualization::print_prepare_usage( argv[0] );
                return EXIT_FAILURE;
            }
        }

        parameters_t parameters = main_classes::read_parameters_json<parameters_real>( path_to_config_file );
        if ( parameters.nonlinear_operator.N_size.empty() || parameters.nonlinear_operator.N_size.at( 0 ) != 1 )
        {
            throw std::runtime_error(
                "circle visualization expects nonlinear_operator.discrete_problem_dimensions[0] == 1"
            );
        }
        if ( options.project_dir.empty() )
        {
            options.project_dir = parameters.path_to_project;
        }

        auto                                             vec_ops  = std::make_shared<vec_ops_real>( 1 );
        auto                                             file_ops = std::make_shared<files_ops_t>( vec_ops.get() );
        visualization::state_vector_writer<vec_ops_real> writer( vec_ops.get(), "state_scalar" );
        visualization::bd_visualization_preparer<vec_ops_real, files_ops_t, decltype( writer )> preparer(
            vec_ops.get(), file_ops.get()
        );
        preparer.prepare( options, writer );
    }
    catch ( const std::exception &e )
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
