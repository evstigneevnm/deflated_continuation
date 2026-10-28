#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <deflation/solution_storage.h>
#include <nonlinear_operators/bratu/bratu.h>
#include <nonlinear_operators/bratu/linear_operator_bratu.h>
#include <nonlinear_operators/bratu/preconditioner_bratu.h>
#include "bratu_backend_typedefs.h"
#include "bratu_linear_solvers.h"

namespace
{
using space_type    = vec_ops_real;
using problem_type  = nonlinear_operators::bratu<space_type>;
using jacobian_type = nonlinear_operators::linear_operator_bratu<space_type, problem_type>;
using inverse_type  = nonlinear_operators::preconditioner_bratu<space_type, problem_type, jacobian_type>;
using system_type =
    bratu_model::linear_system<space_type, bratu_model::monitor<space_type, log_t>, log_t, jacobian_type, inverse_type>;
using storage_type = deflation::solution_storage<space_type, log_t>;
using types = bratu_model::newton_types<space_type, log_t, problem_type, jacobian_type, system_type, storage_type>;
using buffer_type      = nmfd::detail::vector_wrap<space_type, false, false>;
using pair_buffer_type = nmfd::detail::vector_wrap<types::pair_space_type, false, false>;
int checks             = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

struct settings
{
    real tolerance = 1e-9, newton_wight = 1, relax_tolerance_factor = 1, maximum_norm_increase = .1,
         newton_wight_threshold = 1e-12;
    unsigned int newton_max_it = 40, stagnation_max = 10;
    int          relax_tolerance_steps = 0;
    bool         store_norms_history = true, verbose = false;
};

// u'' + lambda*exp(u) = 0, u(0)=u(1)=0. The two positive branches meet at
// a saddle-node. Check Chebyshev and FD3 discretizations with ordinary NMFD
// Newton, full bordered direct correction, and the unchanged rank-one deflation.
void test_corrections( problem_type::spatial_discretization discretization )
{
    auto space = std::make_shared<space_type>( 15 );
    space->use_high_precision();
    auto problem  = std::make_shared<problem_type>( 15, space, discretization );
    auto jacobian = std::make_shared<jacobian_type>( problem );
    auto inverse  = std::make_shared<inverse_type>( problem );
    auto system   = std::make_shared<system_type>( space, jacobian, inverse );
    system->get_linsolver_handle()->monitor().set_tolerances( 1e-11, 1e-11 );
    system->get_linsolver_handle_original()->monitor().set_tolerances( 1e-11, 1e-11 );
    buffer_type reference( *space ), state( *space ), tangent( *space ), work( *space ), saved( *space );
    for ( auto *buffer : { &reference, &state, &tangent, &work, &saved } )
    {
        buffer->init();
        buffer->start_use();
    }
    const real lambda = problem->lambda_from_theta( 1. );
    problem->exact_solution_from_theta( 1., *state );
    auto ordinary =
        std::shared_ptr<system_type::linear_solver_original_type>( system, system->get_linsolver_handle_original() );
    auto fixed_op = std::make_shared<types::fixed_operator_type>( problem, jacobian, lambda );
    auto fixed    = types::fixed_method::make(
        space, problem, fixed_op, space, ordinary, nullptr, std::make_shared<types::projector_type>( problem )
    );
    settings parameters;
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
    real parameter = lambda;
    for ( real value : { lambda, lambda + .1, lambda } )
    {
        parameter = value;
        space->scale( 1.001, *state );
        require( fixed->solve( problem.get(), *state, parameter ) && parameter == value, "Fixed solve/rebinding" );
        problem->F( *state, parameter, *work );
        require( space->norm_l2( *work ) <= parameters.tolerance, "Fixed physical residual" );
    }
    space->assign( *state, *reference );
    auto op     = std::make_shared<types::corrector_operator_type>( space, problem, jacobian );
    auto linear = types::make_bordered( op->space(), system, inverse, nullptr );
    linear->monitor().set_tolerances( 1e-11, 1e-11 );
    require( !linear->supports_unpreconditioned_retry(), "Direct inverse cannot be bypassed for recovery" );
    auto tangent_solver    = types::make_tangent( op, linear );
    real tangent_parameter = 0;
    require( tangent_solver->initial( *reference, lambda, 1., *tangent, tangent_parameter ), "Direct initial tangent" );
    require( tangent_parameter > 0, "Initial tangent orientation" );
    require(
        linear->monitor().rel_tol() == 1e-11 && linear->monitor().abs_tol() == 1e-11,
        "Tangent restores monitor tolerances"
    );
    auto coupled = types::continuation_method::make(
        space, problem, op, op->space(), linear, nullptr,
        std::make_shared<types::corrector_operator_type::trial_projector>( op )
    );
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    coupled->set_tangent_space( *reference, lambda, *tangent, tangent_parameter, .02, 'S' );
    space->assign_lin_comb( 1., *reference, .02, *tangent, *state );
    parameter = lambda + .02 * tangent_parameter;
    require( coupled->solve( problem.get(), *state, parameter ), "Direct arclength correction" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Corrected physical residual" );
    require(
        std::abs(
            continuation::evaluate_arclength_constraint(
                *space, *reference, lambda, *tangent, tangent_parameter, .02, *state, parameter, *work
            )
        ) <= parameters.tolerance,
        "Arclength residual"
    );

    pair_buffer_type exact( *op->space() ), rhs( *op->space() ), result( *op->space() );
    for ( auto *buffer : { &exact, &rhs, &result } )
    {
        buffer->init();
        buffer->start_use();
    }
    space->assign_scalar( .03, ( *exact ).first );
    ( *exact ).second[0] = .2;
    for ( real value : { lambda, lambda + .1, lambda } )
    {
        op->prepare_tangent( *reference, value, *tangent, 0. );
        auto prepared = op->get_jacobi_operator();
        linear->set_operator( prepared );
        prepared->apply( *exact, *rhs );
        require( linear->solve( *rhs, *result ), "Repeated zero-border direct solve" );
        op->space()->add_lin_comb( -1., *exact, 1., *result );
        require( op->space()->norm( *result ) < 1e-9, "Assembled bordered solution" );
    }
    parameters.newton_max_it = 1;
    types::configure_continuation( coupled->get_convergence_strategy_handle(), parameters );
    space->assign_mul( 1.1, *reference, *state );
    space->assign( *state, *saved );
    parameter = lambda;
    require( !coupled->solve( problem.get(), *state, parameter ), "Iteration exhaustion must fail" );
    space->add_lin_comb( -1., *saved, 1., *state );
    require( space->norm( *state ) == 0 && parameter == lambda, "Corrector failure rolls back both unknowns" );
    parameters.newton_max_it = 40;

    auto storage = std::make_shared<storage_type>( space.get(), 2, real( space->get_vector_size() ) );
    storage->push_back( *reference );
    auto deflated_op =
        std::make_shared<types::deflated_operator_type>( space, problem, jacobian, storage, lambda, 1e-7 );
    auto deflated = types::deflation_method::make(
        space, problem, deflated_op, space, std::make_shared<types::rank_one_type>( system ), nullptr,
        std::make_shared<types::deflated_operator_type::trial_projector>( problem )
    );
    types::configure_deflation( deflated->get_convergence_strategy_handle(), parameters );
    space->assign( *reference, *state );
    require( !deflated->solve( problem.get(), *state, parameter ), "Deflation excludes a stored root" );
    space->add_lin_comb( -1., *reference, 1., *state );
    require( space->norm( *state ) == 0 && parameter == lambda, "Deflation rejection preserves the state" );
    storage->clear();
    space->assign_mul( .99, *reference, *state );
    require( deflated->solve( problem.get(), *state, parameter ), "Rank-one deflated correction" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Deflated physical residual" );
}

// One FD3 unknown has J=-8+lambda*exp(u). At u=0, lambda=8,
// [J F_lambda; 1 0]=[0 1; 1 0] is invertible despite singular J and E=0.
void test_singular_state_block()
{
    auto space     = std::make_shared<space_type>( 1 );
    auto problem   = std::make_shared<problem_type>( 1, space, problem_type::spatial_discretization::fd3 );
    auto jacobian  = std::make_shared<jacobian_type>( problem );
    auto op        = std::make_shared<types::corrector_operator_type>( space, problem, jacobian );
    using assembly = bratu_model::augmented_solver_types<space_type, log_t, problem_type, jacobian_type>;
    auto inverse   = std::make_shared<assembly::inverse_type>( op->space() );
    auto native =
        std::make_shared<assembly::solver_type>( op->space(), nullptr, assembly::solver_type::params(), inverse );
    require(
        !numerical_algos::lin_solvers::recovery::supports_unpreconditioned_retry( native.get(), 0 ),
        "Project recovery excludes a bare direct solver without modifying its interface"
    );
    assembly::linear_type linear( op->space(), native );
    linear.monitor().set_tolerances( 0., 1e-12 );
    pair_buffer_type state( *op->space() ), row( *op->space() ), rhs( *op->space() ), result( *op->space() );
    for ( auto *buffer : { &state, &row, &rhs, &result } )
    {
        buffer->init();
        buffer->start_use();
    }
    space->assign_scalar( 0., ( *state ).first );
    space->assign_scalar( 1., ( *row ).first );
    space->assign_scalar( .3, ( *rhs ).first );
    ( *rhs ).second[0] = .2;
    op->prepare_tangent( ( *state ).first, 8., ( *row ).first, 0. );
    linear.set_operator( op->get_jacobi_operator() );
    require( linear.solve( *rhs, *result ), "Singular state block with invertible bordered system" );
    real first = 0;
    space->get( ( *result ).first, &first, 1 );
    require(
        std::abs( first - .2 ) < 1e-12 && std::abs( ( *result ).second[0] - .3 ) < 1e-12, "Pivoted full-system solution"
    );
    space->assign_scalar( 0., ( *row ).first );
    op->prepare_tangent( ( *state ).first, 8., ( *row ).first, 0. );
    linear.set_operator( op->get_jacobi_operator() );
    require( !linear.solve( *rhs, *result ), "Genuinely singular full matrix fails" );
    space->get( ( *result ).first, &first, 1 );
    require(
        std::abs( first - .2 ) < 1e-12 && std::abs( ( *result ).second[0] - .3 ) < 1e-12,
        "Direct failure preserves both outputs"
    );
}
}

int main()
{
    try
    {
        test_corrections( problem_type::spatial_discretization::chebyshev );
        test_corrections( problem_type::spatial_discretization::fd3 );
        test_singular_state_block();
        std::cout << "Bratu NMFD Newton: " << checks << " checks passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
