#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <common/gpu_file_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <main/nmfd_newton_types.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_preconditioner.h>

#include "KS1D_backend_typedefs.h"
#include "KS1D_linear_solvers.h"

#if defined( KS1D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
#endif

namespace
{

int            checks    = 0;
constexpr real parameter = 16.14704430336377;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

using space_type          = vec_ops_real;
using buffer_type         = nmfd::detail::vector_wrap<space_type, false, false>;
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_1d_full<space_type, fft_backend_t>;
using jacobian_type       = symmetry::linearization::projected_linear_operator<space_type, problem_type>;
using preconditioner_type = symmetry::linearization::projected_preconditioner<space_type, problem_type, jacobian_type>;
using monitor_type        = nmfd::solvers::default_monitor<space_type, log_t>;
using system_type         = nmfd::solvers::sherman_morrison<
            space_type, monitor_type, log_t, jacobian_type, preconditioner_type, nmfd::solvers::bicgstabl>;
using adapter_type  = symmetry::fourier::real_packed_fourier_slice_1d_adapter<space_type>;
using actions_type  = symmetry::finite_action_registry<space_type>;
using quotient_type = symmetry::finite_quotient_adapter<space_type, adapter_type>;
using storage_type  = deflation::symmetry_solution_storage<space_type, quotient_type, log_t>;
using types         = main_classes::nmfd_newton_types<
            space_type, log_t, problem_type, jacobian_type, system_type, storage_type, main_classes::detail::newton_convergence,
            main_classes::detail::newton_convergence, main_classes::detail::fixed_damping_deflation_convergence>;

struct settings
{
    real tolerance = 1e-9, newton_wight = 1, relax_tolerance_factor = 1, maximum_norm_increase = .1,
         newton_wight_threshold = 1e-12;
    unsigned int newton_max_it = 40, stagnation_max = 10;
    int          relax_tolerance_steps = 0;
    bool         store_norms_history = true, verbose = false;
};

template <class Solver>
void configure_linear( Solver *linear )
{
    monitor_type::params p;
    p.rel_tol       = 1e-11;
    p.abs_tol       = 1e-12;
    p.max_iters_num = 500;
    linear->monitor().configure( p );
    linear->set_basis_size( 4 );
    linear->set_use_precond_resid( false );
}

// Full, zero-mean KS on 128 grid points has 126 real Fourier coefficients.
// Translation reduction removes the neutral group direction. The archived
// state has one unstable real eigenvalue in the quotient. Corrections must
// preserve its solution orbit and the prepared chart, not its absolute phase.
struct fixture
{
    std::shared_ptr<space_type>    space    = std::make_shared<space_type>( 126 );
    std::shared_ptr<problem_type>  problem  = std::make_shared<problem_type>( 2., 4., 128, space );
    std::shared_ptr<jacobian_type> jacobian = std::make_shared<jacobian_type>( problem );
    std::shared_ptr<system_type>   system =
        std::make_shared<system_type>( space, jacobian, std::make_shared<preconditioner_type>( problem ) );
    std::shared_ptr<adapter_type>               adapter = std::make_shared<adapter_type>( space, 63 );
    std::shared_ptr<actions_type>               actions = std::make_shared<actions_type>( space );
    std::shared_ptr<storage_type>               storage;
    std::shared_ptr<types::fixed_operator_type> fixed_op =
        std::make_shared<types::fixed_operator_type>( problem, jacobian, parameter );
    std::shared_ptr<types::fixed_correction_type> fixed;
    buffer_type reference{ *space }, state{ *space }, tangent{ *space }, direction{ *space }, action{ *space },
        plus{ *space }, minus{ *space }, work{ *space }, saved{ *space };
    settings parameters;

    fixture()
    {
        for ( auto *v : { &reference, &state, &tangent, &direction, &action, &plus, &minus, &work, &saved } )
        {
            v->init();
            v->start_use();
        }
        configure_linear( system->get_linsolver_handle() );
        configure_linear( system->get_linsolver_handle_original() );
        auto ordinary = std::shared_ptr<system_type::linear_solver_original_type>(
            system, system->get_linsolver_handle_original()
        );
        fixed = types::fixed_method::make(
            space, problem, fixed_op, space, ordinary, nullptr, std::make_shared<types::projector_type>( problem )
        );
        types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
        problem->configure_finite_symmetry_actions( *actions );
        storage = std::make_shared<storage_type>(
            space, 4, real( space->get_vector_size() ), 2., std::make_shared<quotient_type>( space, adapter, actions ),
            1e-8
        );
        gpu_file_operations<space_type>( space.get() )
            .read_vector( "source/models/KS_1D/tests/data/stability_replay/curve03_state058.dat", *reference );
        problem->project( *reference );
    }

    void difference(
        const space_type::vector_type &left, const space_type::vector_type &right, real tolerance, const char *message
    )
    {
        space->assign_mul( 1., left, -1., right, *work );
        if ( space->norm( *work ) > tolerance )
            std::cerr << message << ": difference=" << space->norm( *work ) << ", tolerance=" << tolerance << '\n';
        require( space->norm( *work ) <= tolerance, message );
    }
};

void test_fixed_and_frozen_jacobian()
{
    fixture           f;
    auto             &space = *f.space;
    std::vector<real> values( 126 );
    for ( std::size_t i = 0; i < values.size(); ++i )
        values[i] = std::sin( real( i + 1 ) ) / real( ( i + 1 ) * ( i + 1 ) );
    space.set( values.data(), *f.direction, values.size() );
    f.fixed_op->set_linearization_point( *f.reference );
    const auto mode = f.problem->last_slice_data().mode;
    space.assign_mul( 1., *f.reference, .1, *f.direction, *f.state );
    f.fixed_op->apply( *f.state, *f.work );
    require( f.problem->last_slice_data().mode == mode, "Residual trial reselected the frozen chart" );
    // Avoid a discrete phase tie at the subperiodic archived equilibrium.
    space.assign_mul( 1., *f.reference, .1, *f.direction, *f.state );
    f.problem->project( *f.state );
    f.fixed_op->set_linearization_point( *f.state );
    f.problem->project_current_tangent( *f.direction, *f.direction );
    f.jacobian->apply( *f.direction, *f.action );
    space.assign_mul( 1., *f.state, .01, *f.direction, *f.plus );
    f.fixed_op->apply( *f.plus, *f.work );
    f.jacobian->apply( *f.direction, *f.saved );
    f.difference( *f.action, *f.saved, 1e-10, "Residual trial changed prepared Jacobian" );
    const real h = 1e-6;
    space.assign_mul( 1., *f.state, h, *f.direction, *f.plus );
    space.assign_mul( 1., *f.state, -h, *f.direction, *f.minus );
    f.fixed_op->apply( *f.plus, *f.work );
    f.fixed_op->apply( *f.minus, *f.plus );
    space.assign_mul( 1 / ( 2 * h ), *f.work, -1 / ( 2 * h ), *f.plus, *f.saved );
    f.difference( *f.action, *f.saved, 2e-6 * space.norm( *f.action ), "Projected Jacobian finite difference" );

    real lambda = parameter;
    space.assign_mul( .999, *f.reference, *f.state );
    require( f.fixed->solve( f.problem.get(), *f.state, lambda ) && lambda == parameter, "Fixed correction failed" );
    f.problem->projected_F( *f.state, lambda, *f.work );
    require( space.norm_l2( *f.work ) <= f.parameters.tolerance, "Fixed quotient residual" );
    space.assign( *f.state, *f.reference );
    f.storage->push_back( *f.reference );
    f.adapter->apply_shift( *f.reference, *f.state, .37 );
    require( f.fixed->solve( f.problem.get(), *f.state, lambda ), "Translated fixed correction failed" );
    require( f.storage->nearest_stabilized_distance( *f.state ) < 1e-7, "Fixed correction changed solution orbit" );
    space.assign_mul( .99, *f.reference, *f.state );
    space.assign( *f.state, *f.saved );
    auto *linear = f.system->get_linsolver_handle_original();
    linear->monitor().set_temp_max_iterations( 0 );
    require( !f.fixed->solve( f.problem.get(), *f.state, lambda ), "Linear failure reported convergence" );
    f.difference( *f.state, *f.saved, 0., "Failed fixed correction committed a state" );
    require( lambda == parameter && !f.system->preconditioner_bypass_enabled(), "Fixed failure changed bindings" );
    linear->monitor().restore_max_iterations();
    require( f.fixed->solve( f.problem.get(), *f.state, lambda ), "Fixed correction did not recover" );
}

void test_continuation()
{
    fixture f;
    auto   &space  = *f.space;
    real    lambda = parameter;
    require( f.fixed->solve( f.problem.get(), *f.reference, lambda ), "Reference correction failed" );
    f.problem->set_projected_linearization_point( *f.reference, lambda );
    f.problem->projected_jacobian_alpha( *f.work );
    space.scale( -1., *f.work );
    space.assign_scalar( 0., *f.tangent );
    require( f.system->solve( *f.jacobian, *f.work, *f.tangent ), "Projected tangent solve failed" );
    const real tangent_parameter = 1 / std::sqrt( 1 + space.scalar_prod( *f.tangent, *f.tangent ) );
    space.scale( tangent_parameter, *f.tangent );
    f.problem->begin_continuation_chart( *f.reference, lambda, *f.tangent, tangent_parameter );
    const auto chart_mode = f.problem->last_slice_data().mode;
    auto       op         = std::make_shared<types::corrector_operator_type>( f.space, f.problem, f.jacobian );
    auto       correction = types::continuation_method::make(
        f.space, f.problem, op, op->space(), std::make_shared<types::bordered_type>( f.system ), nullptr,
        std::make_shared<types::corrector_operator_type::trial_projector>( op )
    );
    types::configure_continuation( correction->get_convergence_strategy_handle(), f.parameters );
    correction->set_tangent_space( *f.reference, lambda, *f.tangent, tangent_parameter, .02, 'S' );
    space.assign_mul( 1., *f.reference, .02, *f.tangent, *f.state );
    lambda += .02 * tangent_parameter;
    require( correction->solve( f.problem.get(), *f.state, lambda ), "Projected arclength correction failed" );
    f.problem->projected_F( *f.state, lambda, *f.work );
    require( space.norm_l2( *f.work ) <= f.parameters.tolerance, "Continuation quotient residual" );
    require(
        std::abs(
            continuation::evaluate_arclength_constraint(
                space, *f.reference, parameter, *f.tangent, tangent_parameter, .02, *f.state, lambda, *f.work
            )
        ) <= f.parameters.tolerance,
        "Continuation constraint residual"
    );
    f.parameters.newton_max_it = 1;
    types::configure_continuation( correction->get_convergence_strategy_handle(), f.parameters );
    space.assign_mul( .9, *f.reference, *f.state );
    space.assign( *f.state, *f.saved );
    lambda = parameter;
    require( !correction->solve( f.problem.get(), *f.state, lambda ), "Exhausted corrector reported convergence" );
    f.difference( *f.state, *f.saved, 0., "Failed corrector committed a state" );
    require( lambda == parameter, "Failed corrector committed a parameter" );
    f.problem->restore_continuation_chart( *f.reference, parameter, *f.tangent, tangent_parameter );
    require( f.problem->last_slice_data().mode == chart_mode, "Failed corrector lost continuation chart" );
}

void test_deflation()
{
    fixture f;
    auto   &space  = *f.space;
    real    lambda = parameter;
    require( f.fixed->solve( f.problem.get(), *f.reference, lambda ), "Deflation reference correction failed" );
    f.storage->push_back( *f.reference );
    f.adapter->apply_shift( *f.reference, *f.state, .37 );
    f.storage->push_back( *f.state );
    f.actions->apply( 1, *f.reference, *f.state );
    f.storage->push_back( *f.state );
    require( f.storage->get_size() == 1, "Symmetry-equivalent roots were stored separately" );
    auto op =
        std::make_shared<types::deflated_operator_type>( f.space, f.problem, f.jacobian, f.storage, parameter, 1e-7 );
    auto correction = types::deflation_method::make(
        f.space, f.problem, op, f.space, std::make_shared<types::rank_one_type>( f.system ), nullptr,
        std::make_shared<types::deflated_operator_type::trial_projector>( f.problem )
    );
    types::configure_deflation( correction->get_convergence_strategy_handle(), f.parameters );
    space.assign_mul( .97, *f.reference, *f.state );
    space.assign_mul( .03, *f.reference, *f.direction );
    op->set_linearization_point( *f.state );
    require( op->get_jacobi_operator()->apply( *f.direction, *f.action ), "Deflated Jacobian failed" );
    const real h = 1e-4;
    space.assign_mul( 1., *f.state, h, *f.direction, *f.plus );
    space.assign_mul( 1., *f.state, -h, *f.direction, *f.minus );
    op->apply( *f.plus, *f.work );
    op->apply( *f.minus, *f.plus );
    space.assign_mul( 1 / ( 2 * h ), *f.work, -1 / ( 2 * h ), *f.plus, *f.saved );
    f.difference( *f.action, *f.saved, 2e-6 * space.norm( *f.action ), "Deflated quotient Jacobian finite difference" );
    f.adapter->apply_shift( *f.reference, *f.state, .37 );
    space.assign( *f.state, *f.saved );
    require( !correction->solve( f.problem.get(), *f.state, lambda ), "Stored solution orbit was accepted" );
    f.difference( *f.state, *f.saved, 0., "Deflation failure committed a translated state" );
    f.actions->apply( 1, *f.reference, *f.state );
    require( !correction->solve( f.problem.get(), *f.state, lambda ), "Stored reflected orbit was accepted" );
    space.assign_mul( 1.00001, *f.reference, *f.state );
    f.storage->push_back( *f.state );
    require( f.storage->get_size() == 2, "Distinct nearby states were merged" );
    f.storage->clear();
    space.assign_mul( .999, *f.reference, *f.state );
    require( correction->solve( f.problem.get(), *f.state, lambda ), "Nonzero deflated correction failed" );
    require(
        op->deflated_residual_norm( *f.state, *f.work ) <= f.parameters.tolerance,
        "Deflation accepted an unresolved equation"
    );
    require( space.norm_l2( *f.state ) > 1, "Deflation returned the excluded zero root" );
}

}

int main()
{
    try
    {
#if defined( KS1D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        test_fixed_and_frozen_jacobian();
        test_continuation();
        test_deflation();
        std::cout << "Full KS1D NMFD Newton: " << checks << " checks, 0 failures\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "Full KS1D NMFD Newton failed: " << error.what() << '\n';
        return 1;
    }
}
