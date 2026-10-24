#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <common/gpu_file_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <main/stability_continuation/solver_bundle.h>
#include <main/nmfd_newton_types.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/kuramoto_sivashinskiy_2d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>
#include <stability/analysis/transition_state_alignment.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

#include "KS2D_backend_typedefs.h"
#include "KS2D_linear_solvers.h"

#if defined( KS2D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
#endif

namespace
{

int                   checks     = 0;
constexpr real        parameter  = 4.333;
constexpr const char *state_file = "source/models/KS_2D/tests/data/symmetric_lambda4_333.dat";

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

class vector_space : public vec_ops_real
{
public:
    mutable int        allocations = 0, frees = 0, activations = 0;
    int                fail_activation = -1;
    inline static bool leaked          = false;

    vector_space() : vec_ops_real( 70 )
    {
    }
    ~vector_space()
    {
        leaked = leaked || allocations != frees;
    }

    void init_vector( vector_type &v ) const
    {
        vec_ops_real::init_vector( v );
        ++allocations;
    }
    void free_vector( vector_type &v ) const
    {
        vec_ops_real::free_vector( v );
        ++frees;
    }
    void start_use_vector( vector_type &v ) const
    {
        vec_ops_real::start_use_vector( v );
        if ( ++activations == fail_activation )
            throw std::runtime_error( "injected activation failure" );
    }
};

using vector_type         = vector_space::vector_type;
using buffer_type         = nmfd::detail::vector_wrap<vector_space, false, false>;
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_2d<vector_space, fft_backend_t>;
using operator_type       = nonlinear_operators::linear_operator_KS_2D<vector_space, problem_type>;
using preconditioner_type = nonlinear_operators::preconditioner_KS_2D<vector_space, problem_type, operator_type>;
using monitor_type        = nmfd::solvers::default_monitor<vector_space, log_t>;
using solver_type         = nmfd::solvers::sherman_morrison<
            vector_space, monitor_type, log_t, operator_type, preconditioner_type, nmfd::solvers::bicgstabl>;
using identity_type = deflation::identity_symmetry_adapter<vector_space>;
using aligner_type  = symmetry::fourier::residual_translation_orbit_aligner_2d<vector_space>;
using actions_type  = symmetry::finite_action_registry<vector_space>;
using quotient_type = symmetry::finite_quotient_adapter<vector_space, identity_type, aligner_type>;
using storage_type  = deflation::symmetry_solution_storage<vector_space, quotient_type, log_t>;

struct fallback_types
{
    using vector_operations_type    = vector_space;
    using nonlinear_operations_type = problem_type;
    using log_type                  = log_t;
    using linear_operator_type      = operator_type;
    using preconditioner_type       = ::preconditioner_type;
    using linear_solver_type        = nmfd::solvers::gmres<
               vector_space, nmfd::solvers::monitor_krylov<vector_space, log_t>, log_t, operator_type, preconditioner_type>;
    using newton_types =
        main_classes::nmfd_fixed_newton_types<vector_space, log_t, problem_type, operator_type, linear_solver_type>;
    using convergence_type     = typename newton_types::fixed_convergence_type;
    using system_operator_type = typename newton_types::fixed_operator_type;
    using newton_type          = typename newton_types::fixed_correction_type;
};
using fallback_bundle =
    main_classes::stability_continuation_detail::solver_bundle<fallback_types, fallback_types::newton_types>;

template <class Factory>
void test_partial_construction( const std::shared_ptr<vector_space> &space, int count, Factory make )
{
    const int live = space->allocations - space->frees;
    for ( int i = 1; i <= count; ++i )
    {
        space->fail_activation = space->activations + i;
        bool failed            = false;
        try
        {
            make();
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        space->fail_activation = -1;
        require( failed, "workspace activation failure was not injected" );
        require( space->allocations - space->frees == live, "partially constructed workspace leaked" );
    }
}

template <class Solver>
void configure( Solver &solver )
{
    monitor_type::params p;
    p.rel_tol       = 1e-10;
    p.abs_tol       = 1e-12;
    p.max_iters_num = 500;
    solver.get_linsolver_handle()->monitor().configure( p );
    solver.get_linsolver_handle_original()->monitor().configure( p );
    solver.get_linsolver_handle()->set_basis_size( 4 );
    solver.get_linsolver_handle_original()->set_basis_size( 4 );
    solver.get_linsolver_handle()->set_use_precond_resid( false );
    solver.get_linsolver_handle_original()->set_use_precond_resid( false );
}

struct failing_preconditioner : preconditioner_type
{
    using preconditioner_type::preconditioner_type;
    mutable int applications = 0;
    bool        apply( vector_type & ) const
    {
        ++applications;
        return false;
    }
};

void test_solves()
{
    auto space = std::make_shared<vector_space>();
    test_partial_construction( space, 1, [&]() { return std::make_shared<problem_type>( 2.0, 4.0, 12, 12, space ); } );
    auto        problem = std::make_shared<problem_type>( 2.0, 4.0, 12, 12, space );
    auto        op      = std::make_shared<operator_type>( problem );
    auto        prec    = std::make_shared<preconditioner_type>( problem );
    solver_type solver( space, op, prec );
    configure( solver );
    buffer_type state( *space ), exact( *space ), rhs( *space ), solution( *space ), work( *space ), c( *space ),
        d( *space );
    for ( auto *v : { &state, &exact, &rhs, &solution, &work, &c, &d } )
    {
        v->init();
        v->start_use();
    }
    gpu_file_operations<vector_space>( space.get() ).read_vector( state_file, *state );
    problem->F( *state, parameter, *rhs );
    require( space->norm( *state ) > 1 && space->norm( *rhs ) < 1e-7, "saved nonzero KS2D equilibrium" );
    std::vector<real> values( 70 );
    for ( std::size_t i = 0; i < values.size(); ++i )
        values[i] = 0.01 * std::cos( real( i ) ) / real( ( i + 1 ) * ( i + 1 ) );
    space->set( values.data(), *exact, values.size() );
    space->assign_mul( 0.002, *state, *c );
    space->assign_mul( -0.001, *state, *d );
    problem->set_linearization_point( *state, parameter );
    for ( int kind = 0; kind < 3; ++kind )
    {
        op->apply( *exact, *rhs );
        if ( kind == 1 )
            space->assign_mul( 0.8, *rhs, -space->scalar_prod( *c, *exact ) / 2.0, *d, *rhs );
        if ( kind == 2 )
            space->add_mul( 0.3, *d, 1.0, *rhs );
        space->assign_scalar( 0.0, *solution );
        real       border = 0;
        const real beta   = space->scalar_prod( *c, *exact ) + 0.6;
        bool       solved;
        if ( kind == 0 )
            solved = solver.solve( *op, *rhs, *solution );
        else if ( kind == 1 )
            solved = solver.solve( 0.8, *op, 2.0, *c, *d, *rhs, *solution );
        else
            solved = solver.solve( *op, *c, *d, 2.0, *rhs, beta, *solution, border );
        require( solved, "ordinary/rank-one/bordered solve failed" );
        op->apply( *solution, *work );
        if ( kind == 1 )
            space->assign_mul( 0.8, *work, -space->scalar_prod( *c, *solution ) / 2.0, *d, *work );
        if ( kind == 2 )
            space->add_mul( border, *d, 1.0, *work );
        space->add_mul( -1.0, *rhs, 1.0, *work );
        require( space->norm( *work ) < 1e-8 * ( 1 + space->norm( *rhs ) ), "original-system true residual" );
        if ( kind == 2 )
            require(
                std::abs( space->scalar_prod( *c, *solution ) + 2 * border - beta ) < 1e-8, "border equation residual"
            );
    }
    op->apply( *exact, *rhs );
    space->assign_scalar( 0.0, *solution );
    solver.get_linsolver_handle_original()->monitor().set_temp_max_iterations( 0 );
    require( !solver.solve( *op, *rhs, *solution ), "exhausted solver reported success" );
    solver.get_linsolver_handle_original()->monitor().restore_max_iterations();
    require( solver.solve( *op, *rhs, *solution ), "solver did not recover after restoring its budget" );
    auto bad_prec = std::make_shared<failing_preconditioner>( problem );
    nmfd::solvers::sherman_morrison<
        vector_space, monitor_type, log_t, operator_type, failing_preconditioner, nmfd::solvers::bicgstabl>
        recovery( space, op, bad_prec );
    configure( recovery );
    int retries = 0;
    require(
        numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &recovery, [&]() { return recovery.solve( *op, *rhs, *solution ); },
            [&]() { space->assign_scalar( 0.0, *solution ); }, [&]() { ++retries; }
        ),
        "unpreconditioned recovery failed"
    );
    require(
        retries == 1 && bad_prec->applications == 1 && !recovery.preconditioner_bypass_enabled(),
        "preconditioner recovery was not restored transactionally"
    );
    op->apply( *solution, *work );
    space->add_mul( -1.0, *rhs, 1.0, *work );
    require( space->norm( *work ) < 1e-8 * ( 1 + space->norm( *rhs ) ), "recovery true residual" );
}

void test_symmetry_ownership()
{
    auto space   = std::make_shared<vector_space>();
    auto problem = std::make_shared<problem_type>( 2.0, 4.0, 12, 12, space );
    auto actions = std::make_shared<actions_type>( space );
    auto workspace =
        problem_type::configure_finite_symmetry_actions( *actions, problem->finite_symmetry_group(), problem );
    auto identity = std::make_shared<identity_type>( space );
    auto aligner  = std::make_shared<aligner_type>( space, problem->state_modes() );
    auto quotient = std::make_shared<quotient_type>( space, identity, actions, aligner );
    auto storage  = std::make_shared<storage_type>( space, 1, real( space->get_vector_size() ), 2.0, quotient, 1e-8 );
    auto op       = std::make_shared<operator_type>( problem );
    auto prec     = std::make_shared<preconditioner_type>( problem );
    prec->set_operator( op );
    std::weak_ptr<operator_type> weak_op = op;
    prec->set_operator( op.get() );
    op.reset();
    require( !weak_op.expired(), "same-operator raw binding lost its owner" );
    prec->set_operator( std::shared_ptr<const operator_type>{} );
    require( weak_op.expired(), "operator rebinding retained its previous owner" );
    prec.reset();
    std::weak_ptr<vector_space>  weak_space    = space;
    std::weak_ptr<problem_type>  weak_problem  = problem;
    std::weak_ptr<quotient_type> weak_quotient = quotient;
    {
        auto       &operations = *space;
        buffer_type state( *space ), image( *space );
        state.init();
        state.start_use();
        image.init();
        image.start_use();
        gpu_file_operations<vector_space>( space.get() ).read_vector( state_file, *state );
        storage->push_back( *state );
        stability::analysis::transition_state_alignment<vector_space> alignment( space.get() );
        alignment.set( quotient );
        space.reset();
        problem.reset();
        identity.reset();
        aligner.reset();
        quotient.reset();
        workspace.reset();
        for ( std::size_t i = 0; i < actions->size(); ++i )
        {
            actions->apply( i, *state, *image );
            storage->push_back( *image );
            alignment.align( *state, *image, *image );
            operations.add_mul( -1.0, *state, 1.0, *image );
            require( operations.norm( *image ) < 1e-7, "finite/residual-translation alignment" );
        }
        require(
            storage->get_size() == 1 && storage->duplicates_skipped_since_clear() == 8,
            "symmetry copies were accepted as distinct solutions"
        );
        require( !weak_problem.expired(), "finite action outlived its codec owner" );
    }
    storage->set_symmetry_adapter( std::shared_ptr<quotient_type>{} );
    require( weak_quotient.expired(), "symmetry rebinding retained the old quotient" );
    storage.reset();
    actions.reset();
    require( weak_space.expired() && weak_problem.expired(), "KS2D symmetry ownership cycle" );
}

void test_fallback_bundle()
{
    log_t log;
    log.set_verbosity( 0 );
    auto                                       space   = std::make_shared<vector_space>();
    auto                                       problem = std::make_shared<problem_type>( 2.0, 4.0, 12, 12, space );
    auto                                       op      = std::make_shared<operator_type>( problem );
    auto                                       prec    = std::make_shared<preconditioner_type>( problem );
    fallback_types::linear_solver_type::params p;
    p.monitor.rel_tol       = 1e-10;
    p.monitor.abs_tol       = 1e-12;
    p.monitor.max_iters_num = 300;
    auto solver             = std::make_shared<fallback_types::linear_solver_type>( space, &log, p, prec );
    solver->set_operator( op );
    const int activations = space->activations;
    auto      bundle      = std::make_shared<fallback_bundle>( space, &log, problem, op, prec, solver );
    test_partial_construction( space, space->activations - activations, [&]() {
        return std::make_shared<fallback_bundle>( space, &log, problem, op, prec, solver );
    } );
    bundle->convergence()->set_convergence_constants( 1e-10, 40, 1.0, 0, 1.0, false, false );
    std::shared_ptr<fallback_types::newton_type> newton( bundle, bundle->newton() );
    std::weak_ptr<fallback_bundle>               weak_bundle  = bundle;
    std::weak_ptr<problem_type>                  weak_problem = problem;
    std::weak_ptr<vector_space>                  weak_space   = space;
    auto                                         state        = std::make_unique<buffer_type>( *space );
    state->init();
    state->start_use();
    gpu_file_operations<vector_space>( space.get() ).read_vector( state_file, **state );
    auto *borrowed_problem = problem.get();
    bundle.reset();
    space.reset();
    problem.reset();
    op.reset();
    prec.reset();
    solver.reset();
    real parameter = 4.334;
    require(
        newton->solve( borrowed_problem, **state, parameter ),
        "owned fallback failed after releasing caller dependencies"
    );
    require( !weak_bundle.expired() && !weak_problem.expired(), "fallback alias lost its solver bundle" );
    state.reset();
    newton.reset();
    require(
        weak_bundle.expired() && weak_problem.expired() && weak_space.expired(), "fallback solver ownership cycle"
    );
}

} // namespace

int main()
{
    try
    {
#if defined( KS2D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        test_solves();
        test_symmetry_ownership();
        test_fallback_bundle();
        require( !vector_space::leaked, "KS2D workspace leak" );
        std::cout << "KS2D NMFD: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "KS2D NMFD failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
