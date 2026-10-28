#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

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
using types        = main_classes::nmfd_augmented_newton_types<
           space_type, log_t, problem_type, jacobian_type, system_type, storage_type, preconditioner_type,
           nmfd::solvers::bicgstabl>;
static_assert( types::uses_augmented_linear_solver, "Symmetric KS1D tests require augmented solves" );

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
    using solver_monitor_type = std::remove_reference_t<decltype( solver.monitor() )>;
    typename solver_monitor_type::params parameters;
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
// and deflation using the production assembly. Augmented BiCGStab(L) acts on
// the complete pair system; the Fourier inverse only preconditions its state block.
void test_corrections()
{
    auto space    = std::make_shared<space_type>( 63 );
    auto problem  = std::make_shared<problem_type>( 2.0, 4.0, 128, space );
    auto jacobian = std::make_shared<jacobian_type>( problem );
    auto inverse  = std::make_shared<preconditioner_type>( problem );
    auto system   = std::make_shared<system_type>( space, jacobian, inverse );
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
    auto augmented  = types::make_bordered( coupled_op->space(), system, inverse, nullptr );
    configure_linear( augmented->linear_solver() );
    augmented->linear_solver().set_resid_recalc_freq( 1 );
    using pair_buffer_type      = nmfd::detail::vector_wrap<types::pair_space_type, false, false>;
    auto            &pair_space = *coupled_op->space();
    pair_buffer_type rhs( pair_space ), correction( pair_space ), residual( pair_space );
    for ( auto *buffer : { &rhs, &correction, &residual } )
    {
        buffer->init();
        buffer->start_use();
    }

    // Rebinding the same handle must refresh Fourier and Schur setup, including E = 0.
    coupled_op->prepare_tangent( *reference, 6., *tangent, 0. );
    const auto prepared = coupled_op->get_jacobi_operator();
    for ( const real value : { 6., 7., 6. } )
    {
        coupled_op->prepare_tangent( *reference, value, *tangent, 0. );
        require( coupled_op->get_jacobi_operator() == prepared, "Prepared Jacobian handle changed" );
        augmented->set_operator( prepared );
        space->assign_mul( .001, *reference, ( *residual ).first );
        ( *residual ).second[0] = .3;
        require( prepared->apply( *residual, *rhs ), "Augmented right-hand side" );
        require( augmented->solve( *rhs, *correction ), "Zero-border spectral solve failed" );
        pair_space.add_lin_comb( -1., *residual, 1., *correction );
        require( pair_space.norm( *correction ) < 1e-7, "Augmented manufactured solution" );
        require( augmented->solve( *rhs, *correction ), "Repeated augmented solve failed" );
        require( prepared->apply( *correction, *residual ), "True augmented residual action" );
        pair_space.add_lin_comb( -1., *rhs, 1., *residual );
        require(
            pair_space.norm( *residual ) <=
                augmented->monitor().abs_tol() + augmented->monitor().rel_tol() * pair_space.norm( *rhs ),
            "True augmented residual exceeds tolerance"
        );
    }
    pair_space.assign( *correction, *residual );
    augmented->monitor().set_temp_max_iterations( 0 );
    require( !augmented->solve( *rhs, *correction ), "Zero-iteration augmented solve reported convergence" );
    pair_space.add_lin_comb( -1., *correction, 1., *residual );
    require( pair_space.norm( *residual ) == 0., "Failed linear solve published a partial correction" );
    augmented->monitor().restore_max_iterations();
    require( augmented->solve( *rhs, *correction ), "Augmented iteration-limit recovery" );

    // Make only the approximate Schur complement singular, then rebuild it.
    space->assign( *prepared->parameter_column, *work );
    inverse->apply( *work );
    const real schur_product = space->scalar_prod( *tangent, *work );
    coupled_op->prepare_tangent( *reference, 6., *tangent, schur_product );
    augmented->set_operator( prepared );
    require( !augmented->linear_solver().preconditioner()->ready(), "Singular Schur setup was accepted" );
    pair_space.assign( *correction, *residual );
    require( !augmented->solve( *rhs, *correction ), "Unusable Schur factors were applied" );
    pair_space.add_lin_comb( -1., *correction, 1., *residual );
    require( pair_space.norm( *residual ) == 0., "Schur failure changed the correction" );
    coupled_op->prepare_tangent( *reference, 6., *tangent, schur_product + 1. );
    augmented->set_operator( prepared );
    require( augmented->linear_solver().preconditioner()->ready(), "Schur rebinding recovery" );

    auto tangent_solver      = types::make_tangent( coupled_op, augmented );
    real direction_parameter = 0.;
    require(
        tangent_solver->initial( *reference, 6., 1., *direction, direction_parameter ), "Augmented initial tangent"
    );
    jacobian->apply( *direction, *work );
    problem->jacobian_alpha( *action );
    space->add_mul( direction_parameter, *action, 1., *work );
    require( space->norm_l2( *work ) < 1e-6 && direction_parameter > 0., "Initial tangent equation and orientation" );
    require(
        std::abs( space->norm_rank1( *direction, direction_parameter ) - 1. ) < 1e-12, "Initial tangent normalization"
    );
    require(
        augmented->monitor().rel_tol() == 1e-11 && augmented->monitor().abs_tol() == 1e-12 &&
            augmented->monitor().max_iters_num() == 300,
        "Tangent solve changed correction controls"
    );
    auto coupled = types::continuation_method::make(
        space, problem, coupled_op, coupled_op->space(), augmented, nullptr,
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
    require(
        tangent_solver->apply(
            *state, parameter, *tangent, tangent_parameter, *direction, direction_parameter, 1e-8, 300
        ),
        "Augmented tangent update failed"
    );
    jacobian->apply( *direction, *work );
    problem->jacobian_alpha( *action );
    space->add_mul( direction_parameter, *action, 1., *work );
    require(
        space->norm_l2( *work ) < 1e-6 &&
            space->scalar_prod( *tangent, *direction ) + tangent_parameter * direction_parameter > 0.,
        "Updated tangent equation and orientation"
    );
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
        !history->empty() && std::abs( history->back() - deflated_op->deflated_residual_norm( *state, *work ) ) < 1e-12,
        "Deflation history does not report the acceptance residual"
    );
}

// At lambda=4*k^2 the k-th zero-state eigenvalue vanishes. Small nonzero
// predictors can satisfy physical tolerance while m*F remains unresolved.
void test_singular_knots()
{
    auto                          space     = std::make_shared<space_type>( 31 );
    auto                          problem   = std::make_shared<problem_type>( 2., 4., 64, space );
    auto                          jacobian  = std::make_shared<jacobian_type>( problem );
    auto                          solutions = std::make_shared<storage_type>( space.get(), 4, 31. );
    preconditioner_type           inverse( problem );
    types::deflated_operator_type op( space, problem, jacobian, solutions, 4., 1e-6 );
    buffer_type                   state( *space ), work( *space ), direction( *space );
    for ( auto *buffer : { &state, &work, &direction } )
    {
        buffer->init();
        buffer->start_use();
        space->assign_scalar( 0., **buffer );
    }
    settings parameters;
    parameters.tolerance = 1e-8;
    types::deflation_convergence_type strategy( space, nullptr );
    types::configure_deflation( &strategy, parameters );
    for ( std::size_t mode : { 1u, 2u } )
    {
        std::vector<real> values( 31, 0. );
        values[mode - 1] = 1e-4;
        space->set( values.data(), *state, values.size() );
        op.set_parameter( 4. * mode * mode );
        require( op.original_residual_norm( *state, *work ) < parameters.tolerance, "Missing singular-knot predictor" );
        require(
            op.deflated_residual_norm( *state, *work ) > .1, "Singular-knot predictor resolved deflated residual"
        );
        strategy.reset_iterations();
        require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *direction ), "Approximate zero branch accepted"
        );

        problem->set_linearization_point( *state, 4. * mode * mode );
        inverse.set_operator( jacobian );
        space->assign_scalar( 1., *work );
        inverse.apply( *work );
        space->assign_scalar( 2., *direction );
        inverse.apply( *direction );
        space->add_mul( -2., *work, 1., *direction );
        require(
            space->is_valid_number( *work ) && space->norm( *direction ) == 0.,
            "Fourier inverse is not finite and linear at a pole"
        );
        space->get( *work, values.data(), values.size() );
        require( values[mode - 1] == 1., "Pole guard changed the null Fourier component" );
    }
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
        test_singular_knots();
        std::cout << "Symmetric KS1D NMFD Newton: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "Symmetric KS1D NMFD Newton failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
