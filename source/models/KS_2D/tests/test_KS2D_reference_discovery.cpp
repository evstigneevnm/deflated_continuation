#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <common/gpu_file_operations.h>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>
#include <containers/bifurcation_diagram/diagram_archive.h>
#include <deflation/symmetry_solution_storage.h>
#include <main/deflation_continuation/parameter_application.h>
#include <main/nmfd_newton_types.h>
#include <main/parameters.hpp>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <models/KS_2D/KS2D_backend_typedefs.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

#if defined( KS2D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
#endif

namespace
{

using space_type          = vec_ops_real;
using files_type          = gpu_file_operations<space_type>;
using buffer_type         = nmfd::detail::vector_wrap<space_type, false, false>;
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_2d<space_type, fft_backend_t>;
using jacobian_type       = nonlinear_operators::linear_operator_KS_2D<space_type, problem_type>;
using preconditioner_type = nonlinear_operators::preconditioner_KS_2D<space_type, problem_type, jacobian_type>;
using system_type         = nmfd::solvers::sherman_morrison<
            space_type, nmfd::solvers::default_monitor<space_type, log_t>, log_t, jacobian_type, preconditioner_type,
            nmfd::solvers::bicgstabl>;
using actions_type  = symmetry::finite_action_registry<space_type>;
using identity_type = deflation::identity_symmetry_adapter<space_type>;
using aligner_type  = symmetry::fourier::residual_translation_orbit_aligner_2d<space_type>;
using quotient_type = symmetry::finite_quotient_adapter<space_type, identity_type, aligner_type>;
using storage_type  = deflation::symmetry_solution_storage<
     space_type, quotient_type, log_t, deflation::symmetry_deflation_metric::nearest_orbit>;
using types = main_classes::nmfd_augmented_newton_types<
    space_type, log_t, problem_type, jacobian_type, system_type, storage_type, preconditioner_type,
    nmfd::solvers::bicgstabl>;
using helper_type = container::curve_helper_container<space_type>;
using curve_type  = container::bifurcation_diagram_curve<
     space_type, files_type, log_t, problem_type, types::fixed_correction_type, storage_type, helper_type>;
using diagram_type = container::bifurcation_diagram<
    space_type, files_type, log_t, problem_type, types::fixed_correction_type, storage_type, curve_type, helper_type>;
using json = nlohmann::json;

json read_json( const char *path )
{
    std::ifstream input( path );
    if ( !input )
        throw std::runtime_error( std::string( "Cannot open " ) + path );
    json result;
    input >> result;
    return result;
}

void require( bool condition, const char *message )
{
    if ( !condition )
        throw std::runtime_error( message );
}

// Steady KS2D: lambda*Delta u + 4*Delta^2 u + 2*lambda*u*(u_x+u_y)=0,
// in the zero-mean, inversion-odd real Fourier subspace. Distinct symmetry
// representatives are one root. Reconstruct the complete archived root set at
// each knot before testing perturbed reference states; exact-root acceptance
// alone does not establish recovery of a missing branch's local Newton basin.
int run( const char *config_path, const char *requests_path, const char *report_path )
{
    const auto parameters = main_classes::read_parameters_json<real>( config_path );
    const auto requests   = read_json( requests_path );
    require( parameters.nonlinear_operator.N_size.size() == 2, "Expected a two-dimensional grid" );
    const auto nx = parameters.nonlinear_operator.N_size.at( 0 );
    const auto ny = parameters.nonlinear_operator.N_size.at( 1 );
    require( nx >= 4 && ny >= 4 && nx % 2 == 0 && ny % 2 == 0, "Invalid KS2D grid" );
    auto space = std::make_shared<space_type>( nx * ny / 2 - 2 );
    if ( parameters.use_high_precision_reduction )
        space->use_high_precision();
    const auto &physical       = parameters.nonlinear_operator.problem_real_parameters_vector;
    auto        problem        = std::make_shared<problem_type>( physical.at( 0 ), physical.at( 1 ), nx, ny, space );
    auto        jacobian       = std::make_shared<jacobian_type>( problem );
    auto        preconditioner = std::make_shared<preconditioner_type>( problem );
    log_t       log;
    log.set_verbosity( 0 );
    auto system = std::make_shared<system_type>( space, jacobian, preconditioner, &log );
    auto original =
        std::shared_ptr<system_type::linear_solver_original_type>( system, system->get_linsolver_handle_original() );
    main_classes::deflation_continuation_detail::configure_linear_solver<real>(
        original.get(), parameters.nonlinear_operator.linear_solver
    );
    main_classes::deflation_continuation_detail::configure_linear_solver<real>(
        system->get_linsolver_handle(), parameters.deflation_continuation.linear_solver_extended
    );
    system->is_small_alpha( parameters.deflation_continuation.linear_solver_extended.is_small_alpha );
    auto fixed_op = std::make_shared<types::fixed_operator_type>( problem, jacobian, 0. );
    auto fixed    = types::fixed_method::make(
        space, problem, fixed_op, space, original, &log, std::make_shared<types::projector_type>( problem )
    );
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters.nonlinear_operator.newton );
    auto       actions = std::make_shared<actions_type>( space );
    const auto action_workspace =
        problem_type::configure_finite_symmetry_actions( *actions, problem->finite_symmetry_group(), problem );
    auto quotient = std::make_shared<quotient_type>(
        space, std::make_shared<identity_type>( space ), actions,
        std::make_shared<aligner_type>( space, problem->state_modes() )
    );
    const auto &policy    = parameters.deflation_continuation.restart_policy;
    const auto &newton    = parameters.deflation_continuation.newton_extended_deflation;
    const real  exclusion = 100 * newton.tolerance;
    auto        storage   = std::make_shared<storage_type>(
        space, 50, real( space->get_vector_size() ), 2., quotient, policy.duplicate_after_deflation_tolerance, &log
    );
    auto op = std::make_shared<types::deflated_operator_type>( space, problem, jacobian, storage, 0., exclusion );
    auto correction = types::deflation_method::make(
        space, problem, op, space, std::make_shared<types::rank_one_type>( system ), &log,
        std::make_shared<types::deflated_operator_type::trial_projector>( problem )
    );
    auto *strategy = correction->get_convergence_strategy_handle();
    types::configure_deflation( strategy, newton );
    files_type   files( space.get() );
    diagram_type diagram(
        space.get(), &files, &log, problem.get(), fixed.get(), parameters.path_to_project,
        parameters.deflation_continuation.skip_files
    );
    const auto archive = container::load_diagram_archive(
        parameters.path_to_project + "/" + parameters.bifurcaiton_diagram_file_name, diagram
    );
    require( archive.succeeded(), "Cannot load known-branch archive" );
    buffer_type reference( *space ), state( *space ), saved( *space ), residual( *space ), gradient( *space );
    for ( auto *buffer : { &reference, &state, &saved, &residual, &gradient } )
    {
        buffer->init();
        buffer->start_use();
    }
    json report = {
        { "backend", KS2D_BACKEND_NAME },
        { "high_precision", parameters.use_high_precision_reduction },
        { "tolerance", newton.tolerance },
        { "exclusion", exclusion },
        { "cases", json::array() }
    };
    bool passed      = true;
    bool initialized = false;
    real knot        = 0;
    for ( const auto &request : requests.at( "cases" ) )
    {
        const real parameter = request.at( "parameter" );
        if ( !initialized || parameter != knot )
        {
            storage->clear();
            auto      *known  = storage.get();
            const auto status = diagram.find_intersection( parameter, known );
            require( status.ok(), "Known-root reconstruction is incomplete" );
            require( storage->get_size() > 0, "Known-root reconstruction is empty" );
            knot        = parameter;
            initialized = true;
        }
        files.read_vector( request.at( "state" ).get<std::string>(), *reference );
        problem->F( *reference, parameter, *residual );
        const real reference_residual = space->norm_l2( *residual );
        const real nearest            = storage->nearest_stabilized_distance( *reference );
        const bool expected_known     = request.value( "known", false );
        json       item               = { { "id", request.at( "id" ) },           { "parameter", parameter },
                                          { "known_roots", storage->get_size() }, { "reference_residual", reference_residual },
                                          { "reference_nearest", nearest },       { "expected_known", expected_known },
                                          { "attempts", json::array() } };
        bool       recovered          = false;
        bool       valid =
            std::isfinite( reference_residual ) && ( expected_known ? nearest <= exclusion : nearest > exclusion );
        for ( const auto &value : request.at( "perturbations" ) )
        {
            const real perturbation = value;
            require( std::isfinite( perturbation ) && std::abs( perturbation ) < 1, "Invalid radial perturbation" );
            space->assign_mul( 1 + perturbation, *reference, *state );
            space->assign( *state, *saved );
            real       result_parameter = parameter;
            const bool converged        = correction->solve( problem.get(), *state, result_parameter );
            problem->F( *state, parameter, *residual );
            const real physical_norm = space->norm_l2( *residual );
            real       multiplier;
            storage->calc_distance( *state, multiplier, *gradient );
            const real minimum  = storage->last_minimum_distance();
            const real distance = storage->canonical_distance( *reference, *state );
            space->assign_mul( 1., *state, -1., *saved, *gradient );
            const real change   = space->norm_l2( *gradient );
            const bool accepted = converged && physical_norm <= newton.tolerance && std::isfinite( multiplier ) &&
                                  multiplier * physical_norm <= newton.tolerance && minimum > exclusion;
            const bool matches = accepted && distance <= requests.value( "orbit_tolerance", 1e-7 );
            recovered          = recovered || ( perturbation != 0 && matches );
            valid              = valid && result_parameter == parameter && ( converged ? accepted : change == 0 );
            if ( expected_known && perturbation == 0 )
                valid = valid && !converged;
            item["attempts"].push_back(
                { { "perturbation", perturbation },
                  { "converged", converged },
                  { "status", strategy->get_result_status() },
                  { "iterations", strategy->get_number_of_iterations() },
                  { "relaxed", strategy->used_relaxed_tolerance() },
                  { "inexact", strategy->used_inexact_increment() },
                  { "physical_residual", physical_norm },
                  { "multiplier", multiplier },
                  { "deflated_residual", multiplier * physical_norm },
                  { "minimum_distance", minimum },
                  { "reference_distance", distance },
                  { "matches_reference", matches },
                  { "output_change", change } }
            );
            std::cerr << request.at( "id" ) << " perturbation=" << perturbation << " converged=" << converged
                      << " status=" << strategy->get_result_status() << " physical=" << physical_norm
                      << " reference_distance=" << distance << '\n';
        }
        item["recovered"] = recovered;
        item["passed"]    = valid && ( expected_known || recovered );
        passed            = passed && item["passed"].get<bool>();
        report["cases"].push_back( std::move( item ) );
        report["passed"] = passed;
        std::ofstream output( report_path );
        require( bool( output ), "Cannot open report" );
        output << report.dump( 2 ) << '\n';
        require( bool( output ), "Cannot write report" );
    }
    require( !report["cases"].empty(), "No discovery cases supplied" );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main( int argc, char **argv )
{
    try
    {
        if ( argc != 4 )
            throw std::invalid_argument( "Usage: test_KS2D_reference_discovery config.json requests.json report.json" );
#if defined( KS2D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        return run( argv[1], argv[2], argv[3] );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "KS2D discovery regression: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
