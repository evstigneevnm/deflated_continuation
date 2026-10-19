#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <common/gpu_file_operations.h>
#include <deflation/symmetry_solution_storage.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d_full.h>
#include <nonlinear_operators/projected_system_operator.h>
#include <stability/analysis/dimension_guarded_eigensolver.h>
#include <stability/analysis/transition_state_alignment.h>
#include <stability/eigensolvers/host_dense_operator_eigensolver.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/linearization/projected_affine_inverse_provider.h>
#include <symmetry/linearization/projected_linear_operator.h>
#include <symmetry/linearization/projected_linearization_provider.h>
#include <symmetry/linearization/projected_preconditioner.h>
#include <symmetry/linearization/projected_stability_linear_operator.h>

#include "KS1D_backend_typedefs.h"
#include "KS1D_linear_solvers.h"

#if defined( KS1D_VECTOR_BACKEND_CUDA )
#    include <common/cuda_init_scfd.h>
#endif

namespace
{

int                   checks     = 0;
constexpr real        parameter  = 16.14704430336377;
constexpr const char *state_file = "source/models/KS_1D/tests/data/stability_replay/curve03_state058.dat";

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
    inline static bool invalid_cleanup = false;

    vector_space() : vec_ops_real( 126 )
    {
    }
    ~vector_space()
    {
        invalid_cleanup = invalid_cleanup || allocations != frees;
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
        start_use_vector( v, get_default_size() );
    }
    void start_use_vector( vector_type &v, std::size_t size ) const
    {
        vec_ops_real::start_use_vector( v, size );
        if ( ++activations == fail_activation )
            throw std::runtime_error( "injected activation failure" );
    }
};

using vector_type   = vector_space::vector_type;
using buffer_type   = nmfd::detail::vector_wrap<vector_space, false, false>;
using problem_type  = nonlinear_operators::kuramoto_sivashinskiy_1d_full<vector_space, fft_backend_t>;
using operator_type = symmetry::linearization::projected_linear_operator<vector_space, problem_type>;
using preconditioner_type =
    symmetry::linearization::projected_preconditioner<vector_space, problem_type, operator_type>;
using stability_operator_type =
    symmetry::linearization::projected_stability_linear_operator<vector_space, problem_type>;
using linearization_type = symmetry::linearization::projected_linearization_provider<vector_space, problem_type>;
using base_provider_type = stability::eigensolvers::transformations::nonlinear_operator_real_affine_inverse_provider<
    vector_space, problem_type>;
using provider_type =
    symmetry::linearization::projected_affine_inverse_provider<vector_space, problem_type, base_provider_type>;
using adapter_type  = symmetry::fourier::real_packed_fourier_slice_1d_adapter<vector_space>;
using actions_type  = symmetry::finite_action_registry<vector_space>;
using quotient_type = symmetry::finite_quotient_adapter<vector_space, adapter_type>;
using storage_type  = deflation::symmetry_solution_storage<vector_space, quotient_type, log_t>;
using monitor_type  = nmfd::solvers::default_monitor<vector_space, log_t>;
using solver_type   = nmfd::solvers::sherman_morrison<
      vector_space, monitor_type, log_t, operator_type, preconditioner_type, nmfd::solvers::bicgstabl>;

struct fixture
{
    std::shared_ptr<vector_space>  space   = std::make_shared<vector_space>();
    std::shared_ptr<problem_type>  problem = std::make_shared<problem_type>( 2.0, 4.0, 128, space );
    std::shared_ptr<adapter_type>  adapter = std::make_shared<adapter_type>( space, 63 );
    std::shared_ptr<actions_type>  actions = std::make_shared<actions_type>( space );
    std::shared_ptr<quotient_type> quotient;

    fixture()
    {
        problem->configure_finite_symmetry_actions( *actions );
        quotient = std::make_shared<quotient_type>( space, adapter, actions );
    }
};

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
        require( failed, "workspace failure was not injected" );
        require( space->allocations - space->frees == live, "partial construction leaked a vector" );
    }
}

void test_construction()
{
    auto space = std::make_shared<vector_space>();
    test_partial_construction( space, 11, [&]() { return std::make_shared<problem_type>( 2.0, 4.0, 128, space ); } );
    fixture f;
    test_partial_construction( f.space, 2, [&]() { return std::make_shared<operator_type>( f.problem ); } );
    test_partial_construction( f.space, 2, [&]() { return std::make_shared<preconditioner_type>( f.problem ); } );
    test_partial_construction( f.space, 2, [&]() { return std::make_shared<stability_operator_type>( f.problem ); } );
    auto base = std::make_shared<base_provider_type>( f.space, f.problem );
    test_partial_construction( f.space, 3, [&]() {
        return std::make_shared<provider_type>( f.space, f.problem, base );
    } );
    test_partial_construction( f.space, 8, [&]() {
        return std::make_shared<quotient_type>( f.space, f.adapter, f.actions );
    } );
    test_partial_construction( f.space, 4, [&]() {
        return std::make_shared<storage_type>( f.space, 2, f.space->get_l2_size(), 2.0, f.quotient );
    } );
    auto        storage = std::make_shared<storage_type>( f.space, 1, f.space->get_l2_size(), 2.0, f.quotient );
    buffer_type state( *f.space );
    state.init();
    state.start_use();
    gpu_file_operations<vector_space>( f.space.get() ).read_vector( state_file, *state );
    test_partial_construction( f.space, 1, [&]() { storage->push_back( *state ); } );
    require( storage->get_size() == 0, "failed insertion committed a solution" );
    storage->push_back( *state );
    require( storage->get_size() == 1, "storage did not recover after failed insertion" );
    f.space->scale( 1.1, *state );
    storage->push_back( *state );
    require( storage->get_size() == 2, "solution storage failed during buffer relocation" );
}

void configure( solver_type &solver )
{
    monitor_type::params p;
    p.rel_tol       = 1e-9;
    p.abs_tol       = 1e-11;
    p.max_iters_num = 500;
    solver.get_linsolver_handle()->monitor().configure( p );
    solver.get_linsolver_handle_original()->monitor().configure( p );
    solver.get_linsolver_handle()->set_basis_size( 4 );
    solver.get_linsolver_handle_original()->set_basis_size( 4 );
    solver.get_linsolver_handle()->set_use_precond_resid( false );
    solver.get_linsolver_handle_original()->set_use_precond_resid( false );
}

void test_numerics()
{
    fixture     f;
    auto       &space = *f.space;
    auto        op    = std::make_shared<operator_type>( f.problem );
    auto        prec  = std::make_shared<preconditioner_type>( f.problem );
    solver_type solver( f.space, op, prec );
    configure( solver );
    buffer_type state( space ), shifted( space ), exact( space ), solution( space ), rhs( space ), work( space ),
        c( space ), d( space ), candidate( space );
    for ( auto *v : { &state, &shifted, &exact, &solution, &rhs, &work, &c, &d, &candidate } )
    {
        v->init();
        v->start_use();
    }
    gpu_file_operations<vector_space>( f.space.get() ).read_vector( state_file, *state );
    f.problem->projected_F( *state, parameter, *rhs );
    // Archived states have small Fourier coefficients zeroed; Newton restores the root.
    require( space.norm( *rhs ) < 1e-2, "archived seed projected residual" );
    f.adapter->apply_shift( *state, *shifted, 0.37 );
    f.problem->projected_F( *shifted, parameter, *work );
    space.add_mul( -1.0, *rhs, 1.0, *work );
    require( space.norm( *work ) < 1e-6, "translated state changed projected residual" );
    storage_type storage( f.space, 1, space.get_l2_size(), 2.0, f.quotient, 1e-8 );
    storage.push_back( *state );
    storage.push_back( *shifted );
    f.actions->apply( 1, *state, *shifted );
    storage.push_back( *shifted );
    require(
        storage.get_size() == 1 && storage.duplicates_skipped_since_clear() == 2,
        "translated/reflected copies were counted as new solutions"
    );

    f.problem->set_projected_linearization_point( *state, parameter );
    space.assign_scalar( 1.0, *work );
    f.problem->project_current_tangent( *work, *shifted );
    space.add_mul( -1.0, *shifted, 1.0, *work );
    require( space.norm( *work ) > 0, "gauge test has no gauge component" );
    space.assign( *work, *exact );
    op->apply( *work, *solution );
    space.add_mul( -1.0, *exact, 1.0, *solution );
    require( space.norm( *solution ) < 1e-7, "Newton gauge completion changed" );
    prec->apply( *work );
    space.add_mul( -1.0, *exact, 1.0, *work );
    require( space.norm( *work ) < 1e-7, "projected preconditioner changed the gauge completion" );
    std::vector<real> direction( 126, 0.0 );
    for ( std::size_t i = 0; i < 8; ++i )
        direction[i] = 0.1 / real( ( i + 1 ) * ( i + 1 ) );
    space.set( direction.data(), *exact, direction.size() );
    f.problem->project_current_tangent( *exact, *exact );
    space.assign_mul( 0.2, *exact, *c );
    space.assign_mul( -0.1, *exact, *d );
    for ( int kind = 0; kind < 3; ++kind )
    {
        op->apply( *exact, *rhs );
        if ( kind == 1 )
            space.assign_mul( 0.8, *rhs, -space.scalar_prod( *c, *exact ) / 2.0, *d, *rhs );
        if ( kind == 2 )
            space.add_mul( 0.3, *d, 1.0, *rhs );
        space.assign_scalar( 0.0, *solution );
        real       border = 0.0;
        const real beta   = space.scalar_prod( *c, *exact ) + 0.6;
        const bool solved = kind == 0   ? solver.solve( *op, *rhs, *solution )
                            : kind == 1 ? solver.solve( 0.8, *op, 2.0, *c, *d, *rhs, *solution )
                                        : solver.solve( *op, *c, *d, 2.0, *rhs, beta, *solution, border );
        require( solved, "projected ordinary/rank-one/bordered solve failed" );
        op->apply( *solution, *work );
        if ( kind == 1 )
            space.assign_mul( 0.8, *work, -space.scalar_prod( *c, *solution ) / 2.0, *d, *work );
        if ( kind == 2 )
            space.add_mul( border, *d, 1.0, *work );
        space.add_mul( -1.0, *rhs, 1.0, *work );
        require( space.norm( *work ) < 1e-8 * ( 1.0 + space.norm( *rhs ) ), "projected true residual" );
        if ( kind == 2 )
            require(
                std::abs( space.scalar_prod( *c, *solution ) + 2.0 * border - beta ) < 1e-8, "projected border residual"
            );
        f.problem->project_current_tangent( *solution, *work );
        space.add_mul( -1.0, *solution, 1.0, *work );
        require( space.norm( *work ) < 1e-8, "linear correction left the tangent space" );
    }

    space.assign_mul( 1.00001, *state, *candidate );
    auto *space_ptr  = f.space.get();
    auto *op_ptr     = op.get();
    auto *solver_ptr = &solver;
    nonlinear_operators::projected_system_operator<vector_space, problem_type, operator_type, solver_type> correction(
        space_ptr, op_ptr, solver_ptr
    );
    for ( int iteration = 0; iteration < 8; ++iteration )
    {
        f.problem->projected_F( *candidate, parameter, *rhs );
        if ( space.norm( *rhs ) < 1e-8 )
            break;
        space.assign_scalar( 0.0, *solution );
        require(
            correction.solve( f.problem.get(), *candidate, parameter, *solution ), "projected Newton correction failed"
        );
        space.add_mul( 1.0, *solution, 1.0, *candidate );
    }
    f.problem->projected_F( *candidate, parameter, *rhs );
    require( space.norm( *rhs ) < 1e-8, "projected Newton failed to converge" );
    f.quotient->stabilize_closest_to_reference( *state, *candidate, *work );
    space.add_mul( -1.0, *state, 1.0, *work );
    require( space.norm( *work ) < 1e-5, "Newton changed the reference orbit" );
    f.problem->set_projected_linearization_point( *state, parameter );
    op->apply( *exact, *rhs );
    space.assign_scalar( 0.0, *solution );
    solver.get_linsolver_handle_original()->monitor().set_temp_max_iterations( 0 );
    require( !solver.solve( *op, *rhs, *solution ), "iteration exhaustion was reported as success" );
    solver.get_linsolver_handle_original()->monitor().restore_max_iterations();
    require( solver.solve( *op, *rhs, *solution ), "solver failed after restoring iteration budget" );
}

void test_lifetimes()
{
    fixture f;
    auto    storage       = std::make_shared<storage_type>( f.space, 2, f.space->get_l2_size(), 2.0, f.quotient );
    auto    op            = std::make_shared<operator_type>( f.problem );
    auto    prec          = std::make_shared<preconditioner_type>( f.problem );
    auto    base          = std::make_shared<base_provider_type>( f.space, f.problem );
    auto    provider      = std::make_shared<provider_type>( f.space, f.problem, base );
    auto    linearization = std::make_shared<linearization_type>( f.problem );
    std::weak_ptr<problem_type>  weak_problem  = f.problem;
    std::weak_ptr<vector_space>  weak_space    = f.space;
    std::weak_ptr<quotient_type> weak_quotient = f.quotient;
    std::weak_ptr<operator_type> weak_op       = op;
    prec->set_operator( op );
    op.reset();
    f.problem.reset();
    f.space.reset();
    f.quotient.reset();
    f.actions.reset();
    f.adapter.reset();
    require(
        !weak_problem.expired() && !weak_space.expired() && !weak_quotient.expired() && !weak_op.expired(),
        "shared projected graph lost a dependency"
    );
    prec->set_operator( std::shared_ptr<const operator_type>{} );
    require( weak_op.expired(), "preconditioner rebinding retained its old operator" );
    prec.reset();
    base.reset();
    provider.reset();
    linearization.reset();
    require( weak_problem.expired() && !weak_space.expired(), "problem leaked or quotient lost its space" );
    storage->set_symmetry_adapter( std::shared_ptr<quotient_type>{} );
    require( weak_quotient.expired(), "storage rebinding retained its quotient" );
    storage.reset();
    require( weak_space.expired(), "symmetry ownership cycle" );
}

void test_stability_ownership()
{
    fixture f;
    auto    state = std::make_unique<buffer_type>( *f.space );
    state->init();
    state->start_use();
    gpu_file_operations<vector_space>( f.space.get() ).read_vector( state_file, **state );
    f.problem->set_stateless_projected_linearization_point( **state, parameter );
    auto op           = std::make_shared<stability_operator_type>( f.problem, -1.0 );
    using lapack_type = nmfd::operations::linalg::host_small_dense_lapack<real>;
    using dense_type =
        stability::eigensolvers::host_dense_operator_eigensolver<vector_space, stability_operator_type, lapack_type>;
    using guarded_type = stability::analysis::dimension_guarded_eigensolver<dense_type, dense_type>;
    auto lapack        = std::make_shared<lapack_type>();
    auto primary       = std::make_shared<dense_type>( f.space, op, lapack );
    auto small         = std::make_shared<dense_type>( f.space, op, lapack );
    auto guarded       = std::make_shared<guarded_type>( primary, small, 126, 126, true );
    stability::analysis::transition_state_alignment<vector_space> alignment( f.space.get() );
    alignment.set( f.quotient );
    std::weak_ptr<problem_type>  weak_problem  = f.problem;
    std::weak_ptr<vector_space>  weak_space    = f.space;
    std::weak_ptr<quotient_type> weak_quotient = f.quotient;
    f.problem.reset();
    f.space.reset();
    f.adapter.reset();
    f.actions.reset();
    f.quotient.reset();
    op.reset();
    lapack.reset();
    primary.reset();
    small.reset();
    require(
        guarded->execute( **state ).succeeded(), "owned dense fallback failed after releasing caller dependencies"
    );
    require( !weak_problem.expired() && !weak_quotient.expired(), "stability graph lost its problem or aligner" );
    state.reset();
    guarded.reset();
    require( weak_problem.expired() && !weak_space.expired(), "stability dependency release order" );
    alignment.reset();
    require( weak_quotient.expired() && weak_space.expired(), "aligner reset leaked a dependency" );
}

} // namespace

int main()
{
    try
    {
#if defined( KS1D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        test_construction();
        test_numerics();
        test_lifetimes();
        test_stability_ownership();
        require( !vector_space::invalid_cleanup, "full KS1D workspace leak" );
        std::cout << "Full KS1D NMFD: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "Full KS1D NMFD failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
