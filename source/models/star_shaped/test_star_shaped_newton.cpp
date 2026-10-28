#include <iostream>
#include <memory>
#include <stdexcept>
#include <common/scalar_math.h>
#include <deflation/solution_storage.h>
#include <nonlinear_operators/star_shaped/star_shaped.h>
#include <nonlinear_operators/star_shaped/linear_operator_star_shaped.h>
#include <nonlinear_operators/star_shaped/preconditioner_star_shaped.h>
#include "star_shaped_backend_typedefs.h"
#include "star_shaped_linear_solvers.h"

#if defined( STAR_SHAPED_VECTOR_BACKEND_HIP )
#    include <scfd/utils/init_hip.h>
#elif !defined( STAR_SHAPED_VECTOR_BACKEND_OMP ) && !defined( STAR_SHAPED_VECTOR_BACKEND_VAR_PREC )
#    include <scfd/utils/init_cuda.h>
#endif

namespace
{
using problem_type  = nonlinear_operators::star_shaped<vec_ops_real>;
using jacobian_type = nonlinear_operators::linear_operator_star_shaped<vec_ops_real, problem_type>;
using inverse_type  = nonlinear_operators::preconditioner_star_shaped<vec_ops_real, problem_type, jacobian_type>;
using system_type   = star_shaped_model::linear_system<
      vec_ops_real, star_shaped_model::monitor<vec_ops_real, log_t>, log_t, jacobian_type, inverse_type>;
using storage_type = deflation::solution_storage<vec_ops_real, log_t>;
using types =
    star_shaped_model::newton_types<vec_ops_real, log_t, problem_type, jacobian_type, system_type, storage_type>;
using buffer_type = nmfd::detail::vector_wrap<vec_ops_real, false, false>;
int  checks       = 0;
void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}
struct settings
{
#if defined( STAR_SHAPED_VECTOR_BACKEND_VAR_PREC )
    real tolerance = real( "1e-60" );
#else
    real tolerance = real( 1e-10 );
#endif
    real newton_wight = 1, relax_tolerance_factor = 1, maximum_norm_increase = 1,
         newton_wight_threshold = real( 1e-12 );
    unsigned int newton_max_it = 40, stagnation_max = 10;
    int          relax_tolerance_steps = 0;
    bool         store_norms_history = true, verbose = false;
};

// F(x,lambda)=r-1-C*sin(4*theta) defines a closed star with multiple folds.
// Its scalar-state bordered Jacobian can be invertible when J or alpha is zero.
void test_corrections()
{
    auto       space    = std::make_shared<vec_ops_real>( 1 );
    auto       problem  = std::make_shared<problem_type>( 1, space, real( .3 ) );
    auto       jacobian = std::make_shared<jacobian_type>( problem );
    auto       inverse  = std::make_shared<inverse_type>( problem );
    auto       system   = std::make_shared<system_type>( space, jacobian, inverse );
    settings   parameters;
    const real linear_tolerance = parameters.tolerance / 100;
    system->get_linsolver_handle()->monitor().set_tolerances( linear_tolerance, linear_tolerance );
    system->get_linsolver_handle_original()->monitor().set_tolerances( linear_tolerance, linear_tolerance );
    buffer_type state( *space ), reference( *space ), tangent( *space ), work( *space );
    for ( auto *buffer : { &state, &reference, &tangent, &work } )
    {
        buffer->init();
        buffer->start_use();
    }
    auto ordinary =
        std::shared_ptr<system_type::linear_solver_original_type>( system, system->get_linsolver_handle_original() );
    auto fixed_op = std::make_shared<types::fixed_operator_type>( problem, jacobian, real( .25 ) );
    auto fixed    = types::fixed_method::make(
        space, problem, fixed_op, space, ordinary, nullptr, std::make_shared<types::projector_type>( problem )
    );
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
    real parameter = real( .25 );
    for ( real value : { real( .25 ), real( .3 ), real( .25 ) } )
    {
        problem->exact_solution( value, *state );
        space->scale( real( 1.01 ), *state );
        parameter = value;
        require( fixed->solve( problem.get(), *state, parameter ) && parameter == value, "Fixed correction/rebinding" );
        problem->F( *state, parameter, *work );
        require( space->norm_l2( *work ) <= parameters.tolerance, "Physical residual at native precision" );
    }
    space->assign( *state, *reference );
    auto op     = std::make_shared<types::corrector_operator_type>( space, problem, jacobian );
    auto linear = types::make_bordered( op->space(), system, inverse, nullptr );
    linear->monitor().set_tolerances( linear_tolerance, linear_tolerance );
    require( !linear->supports_unpreconditioned_retry(), "Direct inverse must not be bypassed" );
    auto tangent_solver    = types::make_tangent( op, linear );
    real tangent_parameter = 0;
    require(
        tangent_solver->initial( *reference, parameter, real( 1 ), *tangent, tangent_parameter ), "Initial tangent"
    );
    require( tangent_parameter > 0, "Tangent orientation" );
    auto corrector = types::continuation_method::make(
        space, problem, op, op->space(), linear, nullptr,
        std::make_shared<types::corrector_operator_type::trial_projector>( op )
    );
    types::configure_continuation( corrector->get_convergence_strategy_handle(), parameters );
    corrector->set_tangent_space( *reference, parameter, *tangent, tangent_parameter, real( .02 ), 'S' );
    space->add_lin_comb( real( .02 ), *tangent, real( 1 ), *state );
    parameter += real( .02 ) * tangent_parameter;
    require( corrector->solve( problem.get(), *state, parameter ), "Full bordered arclength correction" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Continuation physical residual" );

    nmfd::detail::vector_wrap<types::pair_space_type, false, false> rhs( *op->space() ), result( *op->space() );
    rhs.init();
    rhs.start_use();
    result.init();
    result.start_use();
    // At x=0, lambda=1, C=0, the complete Jacobian [0 1; 1 0] is nonsingular.
    auto circle = std::make_shared<problem_type>( 1, space, real( 0 ) );
    auto circle_op =
        std::make_shared<types::corrector_operator_type>( space, circle, std::make_shared<jacobian_type>( circle ) );
    space->assign_scalar( real( 0 ), *state );
    space->assign_scalar( real( 1 ), *tangent );
    circle_op->prepare_tangent( *state, real( 1 ), *tangent, real( 0 ) );
    linear->set_operator( circle_op->get_jacobi_operator() );
    space->assign_scalar( real( .3 ), ( *rhs ).first );
    ( *rhs ).second[0] = real( .2 );
    require( linear->solve( *rhs, *result ), "Singular state block and zero scalar border" );
    real value;
    space->get( ( *result ).first, &value, 1 );
    require( value == real( .2 ) && ( *result ).second[0] == real( .3 ), "Full-system pivoting" );
    space->assign_scalar( real( 0 ), *tangent );
    circle_op->prepare_tangent( *state, real( 1 ), *tangent, real( 0 ) );
    linear->set_operator( circle_op->get_jacobi_operator() );
    require( !linear->solve( *rhs, *result ), "Singular complete Jacobian fails" );
    space->get( ( *result ).first, &value, 1 );
    require( value == real( .2 ) && ( *result ).second[0] == real( .3 ), "Failed linear solve preserves both outputs" );
    parameters.newton_max_it = 1;
    types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
    space->assign_scalar( real( 2 ), *state );
    parameter = real( .25 );
    require( !fixed->solve( problem.get(), *state, parameter ), "Exhausted correction fails" );
    space->get( *state, &value, 1 );
    require( value == real( 2 ) && parameter == real( .25 ), "Fixed failure rolls back state and parameter" );
    auto storage = std::make_shared<storage_type>( space.get(), 2, real( 1 ) );
    storage->push_back( *reference );
    auto deflated_op = std::make_shared<types::deflated_operator_type>(
        space, problem, jacobian, storage, real( .25 ), parameters.tolerance * 100
    );
    auto deflated = types::deflation_method::make(
        space, problem, deflated_op, space, std::make_shared<types::rank_one_type>( system ), nullptr,
        std::make_shared<types::deflated_operator_type::trial_projector>( problem )
    );
    parameters.newton_max_it = 40;
    types::configure_deflation( deflated->get_convergence_strategy_handle(), parameters );
    space->assign( *reference, *state );
    require( !deflated->solve( problem.get(), *state, parameter ), "Deflation excludes a known solution" );
    storage->clear();
    space->scale( real( 1.01 ), *state );
    require( deflated->solve( problem.get(), *state, parameter ), "Rank-one deflated correction" );
    problem->F( *state, parameter, *work );
    require( space->norm_l2( *work ) <= parameters.tolerance, "Deflated physical residual" );
}
}

int main()
{
    try
    {
#if defined( STAR_SHAPED_VECTOR_BACKEND_HIP )
        scfd::utils::init_hip( -1 );
#elif !defined( STAR_SHAPED_VECTOR_BACKEND_OMP ) && !defined( STAR_SHAPED_VECTOR_BACKEND_VAR_PREC )
        scfd::utils::init_cuda( -1 );
#endif
        test_corrections();
        std::cout << "Star-shaped NMFD Newton: " << checks << " checks passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
