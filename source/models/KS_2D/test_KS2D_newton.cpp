#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <common/gpu_file_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <main/nmfd_newton_types.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

#include "KS2D_backend_typedefs.h"
#include "KS2D_linear_solvers.h"

#if defined( KS2D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
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
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_2d<space_type, fft_backend_t>;
using jacobian_type       = nonlinear_operators::linear_operator_KS_2D<space_type, problem_type>;
using preconditioner_type = nonlinear_operators::preconditioner_KS_2D<space_type, problem_type, jacobian_type>;
using monitor_type        = nmfd::solvers::default_monitor<space_type, log_t>;
using system_type         = nmfd::solvers::sherman_morrison<
            space_type, monitor_type, log_t, jacobian_type, preconditioner_type, nmfd::solvers::bicgstabl>;
using actions_type  = symmetry::finite_action_registry<space_type>;
using identity_type = deflation::identity_symmetry_adapter<space_type>;
using aligner_type  = symmetry::fourier::residual_translation_orbit_aligner_2d<space_type>;
using quotient_type = symmetry::finite_quotient_adapter<space_type, identity_type, aligner_type>;
using storage_type  = deflation::symmetry_solution_storage<
     space_type, quotient_type, log_t, deflation::symmetry_deflation_metric::nearest_orbit>;
static_assert( storage_type::deflation_metric == deflation::symmetry_deflation_metric::nearest_orbit );
using types = main_classes::nmfd_augmented_newton_types<
    space_type, log_t, problem_type, jacobian_type, system_type, storage_type, preconditioner_type,
    nmfd::solvers::bicgstabl>;
static_assert( types::uses_augmented_linear_solver, "KS2D tests require full bordered solves" );

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
    typename std::remove_reference_t<decltype( solver.monitor() )>::params parameters;
    parameters.rel_tol       = 1e-11;
    parameters.abs_tol       = 1e-12;
    parameters.max_iters_num = 500;
    solver.monitor().configure( parameters );
    solver.set_basis_size( 4 );
    solver.set_use_precond_resid( false );
}

// u_t + lambda*Delta u + 4*Delta^2 u + 2*lambda*u*(u_x+u_y) = 0, with
// zero spatial mean, is restricted to the existing real, inversion-odd Fourier
// subspace. A 12x12 grid has 70 unknowns. The archived nonzero equilibrium at
// lambda=4.333 has finite spatial symmetries; equivalent copies are one root.
void test_corrections()
{
    auto space          = std::make_shared<space_type>( 70 );
    auto problem        = std::make_shared<problem_type>( 2., 4., 12, 12, space );
    auto jacobian       = std::make_shared<jacobian_type>( problem );
    auto preconditioner = std::make_shared<preconditioner_type>( problem );
    auto system         = std::make_shared<system_type>( space, jacobian, preconditioner );
    configure_linear( *system->get_linsolver_handle() );
    configure_linear( *system->get_linsolver_handle_original() );
    buffer_type reference( *space ), state( *space ), tangent( *space ), work( *space ), saved( *space ),
        direction( *space ), plus( *space ), minus( *space ), action( *space );
    for ( auto *buffer : { &reference, &state, &tangent, &work, &saved, &direction, &plus, &minus, &action } )
    {
        buffer->init();
        buffer->start_use();
    }
    gpu_file_operations<space_type>( space.get() )
        .read_vector( "source/models/KS_2D/tests/data/symmetric_lambda4_333.dat", *reference );
    problem->F( *reference, 4.333, *work );
    require( space->norm( *work ) < 1e-7 && space->norm( *reference ) > 1, "Invalid nonzero KS2D fixture" );

    auto ordinary =
        std::shared_ptr<system_type::linear_solver_original_type>( system, system->get_linsolver_handle_original() );
    auto fixed_op = std::make_shared<types::fixed_operator_type>( problem, jacobian, 4.333 );
    auto fixed    = types::fixed_method::make(
        space, problem, fixed_op, space, ordinary, nullptr, std::make_shared<types::projector_type>( problem )
    );
    settings parameters;
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
    space->assign_mul( 1.001, *reference, *state );
    for ( real parameter : { real( 4.333 ), real( 4.334 ), real( 4.333 ) } )
    {
        const real expected = parameter;
        require( fixed->solve( problem.get(), *state, parameter ), "Fixed correction or parameter rebinding failed" );
        problem->F( *state, parameter, *work );
        require( parameter == expected && space->norm_l2( *work ) <= parameters.tolerance, "Fixed physical residual" );
        require( !fixed->get_convergence_strategy_handle()->get_norms_history_handle()->empty(), "Missing history" );
    }
    space->assign_mul( 1., *state, -1., *reference, *work );
    require( space->norm( *work ) < 1e-7, "Repeated correction changed branch" );

    real parameter = 4.333;
    space->assign_mul( 1.01, *reference, *state );
    space->assign( *state, *saved );
    ordinary->monitor().set_temp_max_iterations( 0 );
    require( !fixed->solve( problem.get(), *state, parameter ), "Linear failure reported convergence" );
    space->assign_mul( 1., *state, -1., *saved, *work );
    require( space->norm( *work ) == 0 && parameter == 4.333, "Fixed failure changed committed unknowns" );
    require( !system->preconditioner_bypass_enabled(), "Failed solve retained preconditioner bypass" );
    ordinary->monitor().restore_max_iterations();
    require( fixed->solve( problem.get(), *state, parameter ), "Fixed solve did not recover" );

    auto coupled_op = std::make_shared<types::corrector_operator_type>( space, problem, jacobian );
    auto linear     = types::make_bordered( coupled_op->space(), system, preconditioner, nullptr );
    configure_linear( linear->linear_solver() );
    auto tangent_solver    = types::make_tangent( coupled_op, linear );
    real tangent_parameter = 0;
    require(
        tangent_solver->initial( *reference, parameter, 1., *tangent, tangent_parameter ),
        "Initial bordered tangent failed"
    );
    require(
        tangent_parameter > 0 && std::abs( space->norm_rank1( *tangent, tangent_parameter ) - 1 ) < 1e-10,
        "Initial tangent orientation or normalization"
    );
    auto coupled = types::continuation_method::make(
        space, problem, coupled_op, coupled_op->space(), linear, nullptr,
        std::make_shared<types::corrector_operator_type::trial_projector>( coupled_op )
    );
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    coupled->set_tangent_space( *reference, 4.333, *tangent, tangent_parameter, .02, 'S' );
    space->assign_mul( 1., *reference, .02, *tangent, *state );
    parameter = 4.333 + .02 * tangent_parameter;
    require( coupled->solve( problem.get(), *state, parameter ), "Arclength correction failed" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Coupled physical residual" );
    const real constraint = continuation::evaluate_arclength_constraint(
        *space, *reference, 4.333, *tangent, tangent_parameter, .02, *state, parameter, *work
    );
    require( std::abs( constraint ) <= parameters.tolerance && parameter > 4.333, "Arclength constraint" );

    real next_tangent_parameter = 0;
    require(
        tangent_solver->apply(
            *state, parameter, *tangent, tangent_parameter, *direction, next_tangent_parameter, 1e-10, 500
        ),
        "Bordered tangent update failed"
    );
    jacobian->apply( *direction, *work );
    problem->jacobian_alpha( *action );
    space->add_mul( next_tangent_parameter, *action, *work );
    require( space->norm_l2( *work ) < 1e-8, "Tangent equation residual" );
    require(
        space->scalar_prod( *direction, *tangent ) + next_tangent_parameter * tangent_parameter > 0,
        "Tangent update changed orientation"
    );

    // A state-only normalization row has a zero scalar border, not a singular full system.
    require(
        tangent_solver->solve( *state, parameter, *direction, 0., -2., *action, next_tangent_parameter, 1e-10, 500 ),
        "Zero-border tangent solve failed"
    );
    require(
        std::abs( space->scalar_prod( *direction, *action ) + 2 ) < 1e-8,
        "Raw tangent solve changed the supplied normalization RHS"
    );
    jacobian->apply( *action, *work );
    problem->jacobian_alpha( *plus );
    space->add_mul( next_tangent_parameter, *plus, *work );
    require( space->norm_l2( *work ) < 1e-8, "Zero-border tangent equation residual" );
    space->assign( *action, *saved );
    const real saved_tangent_parameter = next_tangent_parameter;
    require(
        !tangent_solver->solve( *state, parameter, *direction, 0., 1., *action, next_tangent_parameter, 1e-10, 0 ),
        "Exhausted tangent solve reported convergence"
    );
    space->assign_mul( 1., *action, -1., *saved, *work );
    require(
        space->norm( *work ) == 0 && next_tangent_parameter == saved_tangent_parameter,
        "Failed tangent solve changed committed outputs"
    );
    require(
        !linear->preconditioner_bypass_enabled() && linear->monitor().rel_tol() == 1e-11 &&
            linear->monitor().abs_tol() == 1e-12,
        "Tangent failure changed recovery or monitor settings"
    );
    parameters.newton_max_it = 1;
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    space->assign_mul( 1.01, *reference, *state );
    space->assign( *state, *saved );
    parameter = 4.333;
    require( !coupled->solve( problem.get(), *state, parameter ), "Exhausted corrector reported convergence" );
    space->assign_mul( 1., *state, -1., *saved, *work );
    require( space->norm( *work ) == 0 && parameter == 4.333, "Coupled failure changed committed unknowns" );
    parameters.newton_max_it = 40;
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    coupled->set_tangent_space( *reference, 4.333, *tangent, tangent_parameter, .01, 'S' );
    space->assign_mul( 1., *reference, .01, *tangent, *state );
    parameter = 4.333 + .01 * tangent_parameter;
    require( coupled->solve( problem.get(), *state, parameter ), "Rebound corrector did not recover" );
    parameter = 4.333;

    auto actions = std::make_shared<actions_type>( space );
    auto workspace =
        problem_type::configure_finite_symmetry_actions( *actions, problem->finite_symmetry_group(), problem );
    auto quotient = std::make_shared<quotient_type>(
        space, std::make_shared<identity_type>( space ), actions,
        std::make_shared<aligner_type>( space, problem->state_modes() )
    );
    // RMS distance paired with a Euclidean dot product requires weight N.
    auto solutions = std::make_shared<storage_type>( space, 2, real( space->get_vector_size() ), 2., quotient, 1e-8 );
    solutions->push_back( *reference );
    auto deflated_op = std::make_shared<types::deflated_operator_type>(
        space, problem, jacobian, solutions, parameter, 100 * parameters.tolerance
    );
    auto deflated = types::deflation_method::make(
        space, problem, deflated_op, space, std::make_shared<types::rank_one_type>( system ), nullptr,
        std::make_shared<types::deflated_operator_type::trial_projector>( problem )
    );
    types::configure_deflation( deflated->get_convergence_strategy_handle(), parameters );
    for ( std::size_t i = 0; i < actions->size(); ++i )
    {
        actions->apply( i, *reference, *state );
        space->assign( *state, *saved );
        require( !deflated->solve( problem.get(), *state, parameter ), "A stored symmetry orbit was not excluded" );
        space->assign_mul( 1., *state, -1., *saved, *work );
        require( space->norm( *work ) == 0 && parameter == 4.333, "Deflation failure changed committed unknowns" );
        solutions->push_back( *state );
    }
    require( solutions->get_size() == 1, "Symmetry copies were stored as distinct solutions" );

    // Check the composed rank-one derivative away from an orbit-distance tie.
    space->assign_mul( .8, *reference, *state );
    space->assign_mul( .03, *reference, *direction );
    const real h = 1e-4;
    space->assign_mul( 1., *state, h, *direction, *plus );
    space->assign_mul( 1., *state, -h, *direction, *minus );
    deflated_op->set_linearization_point( *state );
    require( deflated_op->get_jacobi_operator()->apply( *direction, *action ), "Deflated Jacobian failed" );
    deflated_op->apply( *plus, *work );
    deflated_op->apply( *minus, *plus );
    space->assign_mul( 1 / ( 2 * h ), *work, -1 / ( 2 * h ), *plus, *work );
    space->add_mul( -1., *action, 1., *work );
    require( space->norm( *work ) < 2e-7 * space->norm( *action ), "Deflated Jacobian finite difference" );

    // A trial choosing another symmetry representative must not change the prepared rank-one term.
    actions->apply( 1, *state, *saved );
    deflated_op->apply( *saved, *work );
    deflated_op->deflated_residual_norm( *saved, *work );
    require( deflated_op->get_jacobi_operator()->apply( *direction, *plus ), "Prepared deflated Jacobian failed" );
    space->add_mul( -1., *action, *plus );
    require( space->norm( *plus ) == 0., "Orbit trials changed the prepared deflated Jacobian" );

    solutions->clear();
    space->assign_scalar( 0., *state );
    require( !deflated->solve( problem.get(), *state, parameter ), "Zero root was not excluded" );
    space->assign_mul( .99, *reference, *state );
    require( deflated->solve( problem.get(), *state, parameter ), "Nonzero deflated correction failed" );
    problem->F( *state, parameter, *work );
    require(
        space->norm_l2( *work ) <= parameters.tolerance && space->norm( *state ) > 1, "Deflated physical residual"
    );
    const auto *history = deflated->get_convergence_strategy_handle()->get_norms_history_handle();
    require(
        !history->empty() && std::abs( history->back() - deflated_op->deflated_residual_norm( *state, *work ) ) < 1e-12,
        "Deflation history does not report the acceptance residual"
    );
    (void)workspace;
}

}

int main()
{
    try
    {
#if defined( KS2D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        test_corrections();
        std::cout << "KS2D NMFD Newton: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "KS2D NMFD Newton failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
