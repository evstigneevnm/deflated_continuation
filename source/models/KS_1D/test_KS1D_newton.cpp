#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <common/gpu_file_operations.h>
#include <deflation/solution_storage.h>
#include <main/nmfd_newton_types.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h>

#include "KS1D_backend_typedefs.h"
#include "KS1D_linear_solvers.h"

#if defined( KS1D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
#elif defined( KS1D_VECTOR_BACKEND_HIP )
#    include <common/hip_init_scfd.h>
#endif

namespace
{

int checks = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

using space_type          = vec_ops_real;
using buffer_type         = nmfd::detail::vector_wrap<space_type, false, false>;
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_1d<space_type, fft_backend_t>;
using jacobian_type       = nonlinear_operators::linear_operator_KS_1D<space_type, problem_type>;
using preconditioner_type = nonlinear_operators::preconditioner_KS_1D<space_type, problem_type, jacobian_type>;
using monitor_type        = nmfd::solvers::default_monitor<space_type, log_t>;
using system_type         = nmfd::solvers::sherman_morrison<
            space_type, monitor_type, log_t, jacobian_type, preconditioner_type, nmfd::solvers::bicgstabl>;
using storage_type = deflation::solution_storage<space_type, log_t>;
using types =
    main_classes::nmfd_newton_types<space_type, log_t, problem_type, jacobian_type, system_type, storage_type>;

struct settings
{
    real tolerance = 1e-9, newton_wight = 1, relax_tolerance_factor = 1, maximum_norm_increase = .1,
         newton_wight_threshold = 1e-12;
    unsigned int newton_max_it = 40, stagnation_max = 10;
    int          relax_tolerance_steps = 0;
    bool         store_norms_history = true, verbose = false;
};

template <class Solver>
void configure_linear( Solver &solver )
{
    monitor_type::params parameters;
    parameters.rel_tol       = 1e-11;
    parameters.abs_tol       = 1e-12;
    parameters.max_iters_num = 300;
    solver.monitor().configure( parameters );
    solver.set_basis_size( 4 );
    solver.set_use_precond_resid( false );
}

// The inversion-odd, zero-mean KS discretization has 63 real Fourier modes on
// 128 grid points. The archived nonzero equilibrium at lambda=6 is stable in
// this subspace. Exercise physical residuals, coupled arclength correction,
// and deflation using the same NMFD assembly as the production workflows.
void test_corrections()
{
    auto space    = std::make_shared<space_type>( 63 );
    auto problem  = std::make_shared<problem_type>( 2.0, 4.0, 128, space );
    auto jacobian = std::make_shared<jacobian_type>( problem );
    auto system   = std::make_shared<system_type>( space, jacobian, std::make_shared<preconditioner_type>( problem ) );
    configure_linear( *system->get_linsolver_handle() );
    configure_linear( *system->get_linsolver_handle_original() );
    buffer_type reference( *space ), state( *space ), tangent( *space ), direction( *space ), work( *space ),
        plus( *space ), minus( *space ), action( *space ), saved( *space );
    for ( auto *buffer : { &reference, &state, &tangent, &direction, &work, &plus, &minus, &action, &saved } )
    {
        buffer->init();
        buffer->start_use();
    }
    gpu_file_operations<space_type> files( space.get() );
    files.read_vector( "source/models/KS_1D/tests/data/symmetric_lambda6.dat", *reference );
    problem->F( *reference, 6.0, *work );
    require( space->norm_l2( *work ) < 1e-8 && space->norm_l2( *reference ) > 10, "Invalid nonzero fixture" );

    auto ordinary =
        std::shared_ptr<system_type::linear_solver_original_type>( system, system->get_linsolver_handle_original() );
    auto fixed_op = std::make_shared<types::fixed_operator_type>( problem, jacobian, 6.0 );
    auto fixed    = types::fixed_method::make(
        space, problem, fixed_op, space, ordinary, nullptr, std::make_shared<types::projector_type>( problem )
    );
    settings parameters;
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
    real parameter = 6;
    space->assign_mul( 1.001, *reference, *state );
    require( fixed->solve( problem.get(), *state, parameter ) && parameter == 6, "Fixed correction failed" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Fixed physical residual" );
    space->assign_mul( 1.0, *state, -1.0, *reference, *work );
    require( space->norm_l2( *work ) < 1e-7, "Fixed correction changed branch" );
    require( !fixed->get_convergence_strategy_handle()->get_norms_history_handle()->empty(), "Missing history" );

    space->assign_mul( 1.01, *reference, *state );
    space->assign( *state, *saved );
    ordinary->monitor().set_temp_max_iterations( 0 );
    require( !fixed->solve( problem.get(), *state, parameter ), "Linear failure reported convergence" );
    space->assign_mul( 1.0, *state, -1.0, *saved, *work );
    require( space->norm( *work ) == 0 && parameter == 6, "Fixed failure changed committed state" );
    require( !system->preconditioner_bypass_enabled(), "Failed solve retained preconditioner bypass" );
    ordinary->monitor().restore_max_iterations();
    require( fixed->solve( problem.get(), *state, parameter ), "Fixed solve did not recover" );

    problem->set_linearization_point( *reference, 6.0 );
    problem->jacobian_alpha( *work );
    space->scale( -1.0, *work );
    space->assign_scalar( 0.0, *tangent );
    require( system->solve( *jacobian, *work, *tangent ), "Tangent solve failed" );
    const real tangent_parameter = 1 / std::sqrt( 1 + space->scalar_prod( *tangent, *tangent ) );
    space->scale( tangent_parameter, *tangent );
    auto coupled_op = std::make_shared<types::corrector_operator_type>( space, problem, jacobian );
    auto coupled    = types::continuation_method::make(
        space, problem, coupled_op, coupled_op->space(), std::make_shared<types::bordered_type>( system ), nullptr,
        std::make_shared<types::corrector_operator_type::trial_projector>( coupled_op )
    );
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    coupled->set_tangent_space( *reference, 6.0, *tangent, tangent_parameter, .02, 'S' );
    space->assign_mul( 1.0, *reference, .02, *tangent, *state );
    parameter = 6 + .02 * tangent_parameter;
    require( coupled->solve( problem.get(), *state, parameter ), "Arclength correction failed" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Coupled physical residual" );
    const real constraint = continuation::evaluate_arclength_constraint(
        *space, *reference, 6.0, *tangent, tangent_parameter, .02, *state, parameter, *work
    );
    require( std::abs( constraint ) <= parameters.tolerance && parameter > 6, "Arclength constraint" );
    parameters.newton_max_it = 1;
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    space->assign_mul( 1.01, *reference, *state );
    space->assign( *state, *saved );
    parameter = 6;
    require( !coupled->solve( problem.get(), *state, parameter ), "Exhausted corrector reported convergence" );
    space->assign_mul( 1.0, *state, -1.0, *saved, *work );
    require( space->norm( *work ) == 0 && parameter == 6, "Coupled failure changed committed unknowns" );
    parameters.newton_max_it = 40;

    // RMS distance with a Euclidean dot product requires gradient weight N.
    auto solutions = std::make_shared<storage_type>( space.get(), 2, real( space->get_vector_size() ) );
    solutions->push_back( *reference );
    auto deflated_op = std::make_shared<types::deflated_operator_type>(
        space, problem, jacobian, solutions, 6.0, 100 * parameters.tolerance
    );
    space->assign_mul( .8, *reference, *state );
    space->assign_mul( .03, *reference, *direction );
    real multiplier, multiplier_plus, multiplier_minus;
    solutions->calc_distance( *state, multiplier, *work );
    const real derivative = -space->scalar_prod( *work, *direction );
    const real h          = 1e-4;
    space->assign_mul( 1.0, *state, h, *direction, *plus );
    space->assign_mul( 1.0, *state, -h, *direction, *minus );
    solutions->calc_distance( *plus, multiplier_plus, *work );
    solutions->calc_distance( *minus, multiplier_minus, *work );
    require(
        multiplier > 1 && std::abs( ( multiplier_plus - multiplier_minus ) / ( 2 * h ) - derivative ) < 1e-8,
        "Deflation gradient is inconsistent with RMS distance"
    );
    deflated_op->set_linearization_point( *state );
    require( deflated_op->get_jacobi_operator()->apply( *direction, *action ), "Deflated Jacobian failed" );
    deflated_op->apply( *plus, *work );
    deflated_op->apply( *minus, *plus );
    space->assign_mul( 1 / ( 2 * h ), *work, -1 / ( 2 * h ), *plus, *work );
    space->add_mul( -1.0, *action, 1.0, *work );
    require( space->norm( *work ) < 2e-7 * space->norm( *action ), "Deflated Jacobian finite difference" );

    auto deflated = types::deflation_method::make(
        space, problem, deflated_op, space, std::make_shared<types::rank_one_type>( system ), nullptr,
        std::make_shared<types::deflated_operator_type::trial_projector>( problem )
    );
    types::configure_deflation( deflated->get_convergence_strategy_handle(), parameters );
    space->assign( *reference, *state );
    require( !deflated->solve( problem.get(), *state, parameter ), "Stored root was not excluded" );
    space->assign_mul( 1.0, *state, -1.0, *reference, *work );
    require( space->norm( *work ) == 0 && parameter == 6, "Deflation failure changed committed state" );
    solutions->clear();
    space->assign_scalar( 0.0, *state );
    require( !deflated->solve( problem.get(), *state, parameter ), "Zero root was not excluded" );
    space->assign_mul( .99, *reference, *state );
    require( deflated->solve( problem.get(), *state, parameter ), "Nonzero deflated correction failed" );
    problem->F( *state, parameter, *work );
    require(
        space->norm_l2( *work ) <= parameters.tolerance && space->norm_l2( *state ) > 10,
        "Deflation accepted a nonphysical root or the zero branch"
    );
    const auto *history = deflated->get_convergence_strategy_handle()->get_norms_history_handle();
    require(
        !history->empty() && std::abs( history->back() - space->norm_l2( *work ) ) < 1e-12,
        "Deflation history does not report the physical residual"
    );
}

}

int main()
{
    try
    {
#if defined( KS1D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#elif defined( KS1D_VECTOR_BACKEND_HIP )
        common::init_hip_from_scfd_selector( "auto" );
#endif
        test_corrections();
        std::cout << "Symmetric KS1D NMFD Newton: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "Symmetric KS1D NMFD Newton failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
