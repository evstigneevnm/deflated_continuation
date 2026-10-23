#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <common/gpu_file_operations.h>
#include <main/deflation_continuation/parameter_application.h>
#include <main/nmfd_newton_types.h>
#include <main/parameters.hpp>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/default_monitor.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h>

#include "KS1D_backend_typedefs.h"

// Validate inversion-odd KS equilibria using the production residual and fixed
// NMFD Newton correction. Branch predictors are sampled at archived discovery
// parameters; neither curve IDs nor continuation point counts need to match.
int main( int argc, char **argv )
{
    if ( argc != 4 )
    {
        std::cerr << "Usage: " << argv[0] << " CONFIG MANIFEST REPORT\n";
        return 2;
    }
    try
    {
        using space_type    = vec_ops_real;
        using problem_type  = nonlinear_operators::kuramoto_sivashinskiy_1d<space_type, fft_backend_t>;
        using jacobian_type = nonlinear_operators::linear_operator_KS_1D<space_type, problem_type>;
        using prec_type     = nonlinear_operators::preconditioner_KS_1D<space_type, problem_type, jacobian_type>;
        using monitor_type  = nmfd::solvers::default_monitor<space_type, log_t>;
        using linear_type   = nmfd::solvers::bicgstabl<space_type, monitor_type, log_t, jacobian_type, prec_type>;
        using types =
            main_classes::nmfd_fixed_newton_types<space_type, log_t, problem_type, jacobian_type, linear_type>;

        const auto  parameters = main_classes::read_parameters_json<real>( argv[1] );
        const auto &model      = parameters.nonlinear_operator;
        auto        space      = std::make_shared<space_type>( model.N_size.at( 0 ) / 2 - 1 );
        if ( parameters.use_high_precision_reduction )
            space->use_high_precision();
        auto problem = std::make_shared<problem_type>(
            model.problem_real_parameters_vector.at( 0 ), model.problem_real_parameters_vector.at( 1 ),
            model.N_size.at( 0 ), space
        );
        auto jacobian = std::make_shared<jacobian_type>( problem );
        auto linear   = std::make_shared<linear_type>(
            space, nullptr, linear_type::params(), std::make_shared<prec_type>( problem )
        );
        linear->set_operator( jacobian );
        main_classes::deflation_continuation_detail::configure_linear_solver<real>( linear.get(), model.linear_solver );
        auto op     = std::make_shared<types::fixed_operator_type>( problem, jacobian, real( 0 ) );
        auto solver = types::fixed_method::make(
            space, problem, op, space, linear, nullptr, std::make_shared<types::projector_type>( problem )
        );
        types::configure_fixed( solver->get_convergence_strategy_handle(), model.newton );
        nmfd::detail::vector_wrap<space_type, false, false> state( *space ), target( *space ), residual( *space );
        for ( auto *buffer : { &state, &target, &residual } )
        {
            buffer->init();
            buffer->start_use();
        }
        gpu_file_operations<space_type> files( space.get() );
        auto                            report = nlohmann::json::array();
        for ( const auto &entry : main_classes::read_json( argv[2] ) )
        {
            const auto path      = entry.at( "path" ).get<std::string>();
            real       parameter = entry.at( "parameter" ).get<real>();
            files.read_vector( path, *state );
            if ( !std::isfinite( parameter ) || !space->is_valid_number( *state ) )
                throw std::runtime_error( "Nonfinite reference input: " + path );
            problem->F( *state, parameter, *residual );
            nlohmann::json result = { { "path", path }, { "physical_residual", space->norm_l2( *residual ) } };
            if ( entry.contains( "reference" ) )
            {
                result["converged"] = solver->solve( problem.get(), *state, parameter );
                problem->F( *state, parameter, *residual );
                result["corrected_residual"] = space->norm_l2( *residual );
                files.read_vector( entry.at( "reference" ).get<std::string>(), *target );
                space->assign_mul( real( 1 ), *state, real( -1 ), *target, *residual );
                result["relative_difference"] =
                    space->norm( *residual ) / std::max( real( 1 ), space->norm( *target ) );
            }
            report.push_back( result );
        }
        std::ofstream output( argv[3] );
        output << report.dump( 2 ) << '\n';
        if ( !output )
            throw std::runtime_error( "Cannot write reference report" );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
