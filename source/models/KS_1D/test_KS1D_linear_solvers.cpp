#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <common/gpu_file_operations.h>
#include <nmfd/detail/vector_wrap.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/linear_operator_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/preconditioner_KS_1D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/system_operator.h>
#include <stability/eigensolvers/transformations/nonlinear_operator_real_affine_inverse_provider.h>

#include "KS1D_backend_typedefs.h"
#include "KS1D_linear_solvers.h"

#if defined( KS1D_VECTOR_BACKEND_CUDA )
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

class vector_space : public vec_ops_real
{
public:
    mutable int allocations = 0, frees = 0, activations = 0;
    int         fail_activation = -1;

    vector_space() : vec_ops_real( 63 )
    {
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

using vector_type         = vector_space::vector_type;
using buffer_type         = nmfd::detail::vector_wrap<vector_space, false, false>;
using problem_type        = nonlinear_operators::kuramoto_sivashinskiy_1d<vector_space, fft_backend_t>;
using operator_type       = nonlinear_operators::linear_operator_KS_1D<vector_space, problem_type>;
using preconditioner_type = nonlinear_operators::preconditioner_KS_1D<vector_space, problem_type, operator_type>;
using monitor_type        = nmfd::solvers::default_monitor<vector_space, log_t>;
using solver_type         = nmfd::solvers::sherman_morrison<
            vector_space, monitor_type, log_t, operator_type, preconditioner_type, nmfd::solvers::bicgstabl>;
using provider_type = stability::eigensolvers::transformations::nonlinear_operator_real_affine_inverse_provider<
    vector_space, problem_type>;

template <class Solver>
void configure( Solver &solver )
{
    monitor_type::params parameters;
    parameters.rel_tol       = 1e-10;
    parameters.abs_tol       = 1e-12;
    parameters.max_iters_num = 300;
    solver.get_linsolver_handle()->monitor().configure( parameters );
    solver.get_linsolver_handle_original()->monitor().configure( parameters );
    solver.get_linsolver_handle()->set_basis_size( 4 );
    solver.get_linsolver_handle_original()->set_basis_size( 4 );
    solver.get_linsolver_handle()->set_use_precond_resid( false );
    solver.get_linsolver_handle_original()->set_use_precond_resid( false );
}

struct failing_preconditioner : preconditioner_type
{
    using preconditioner_type::preconditioner_type;
    mutable int failed_applications = 0;

    bool apply( vector_type & ) const
    {
        ++failed_applications;
        return false;
    }
};

void test_construction_failure()
{
    for ( int activation = 1; activation <= 7; ++activation )
    {
        auto space             = std::make_shared<vector_space>();
        space->fail_activation = activation;
        bool failed            = false;
        try
        {
            problem_type problem( 2.0, 4.0, 128, space );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        require( failed, "workspace activation failure was not injected" );
        require( space->allocations == space->frees, "partial KS1D construction leaked a vector" );
    }
}

void test_solves( const std::shared_ptr<vector_space> &space, const std::shared_ptr<problem_type> &problem )
{
    auto        op   = std::make_shared<operator_type>( problem );
    auto        prec = std::make_shared<preconditioner_type>( problem );
    solver_type solver( space, op, prec );
    configure( solver );
    buffer_type state( *space ), exact( *space ), rhs( *space ), solution( *space ), work( *space ), c( *space ),
        d( *space ), candidate( *space );
    for ( auto *buffer : { &state, &exact, &rhs, &solution, &work, &c, &d, &candidate } )
    {
        buffer->init();
        buffer->start_use();
    }
    gpu_file_operations<vector_space> files( space.get() );
    files.read_vector( "source/models/KS_1D/tests/data/symmetric_lambda6.dat", *state );
    problem->F( *state, 6.0, *rhs );
    require( space->norm( *rhs ) < 1e-7, "saved nonzero equilibrium is not a solution" );
    require( space->norm( *state ) > 90.0, "regression accidentally uses the zero branch" );
    space->assign_mul( 0.01, *state, *exact );
    space->assign_mul( 0.002, *state, *c );
    space->assign_mul( -0.001, *state, *d );

    for ( const real parameter : { real( 6 ), real( 6.1 ) } )
    {
        solver.get_linsolver_handle_original()->set_use_precond_resid( true );
        problem->set_linearization_point( *state, parameter );
        op->apply( *exact, *rhs );
        space->assign_scalar( 0.0, *solution );
        const bool solved = solver.solve( *op, *rhs, *solution );
        op->apply( *solution, *work );
        space->add_mul( -1.0, *rhs, 1.0, *work );
        if ( !solved )
            std::cerr << "Ordinary solve: lambda=" << parameter
                      << ", relative true residual=" << space->norm( *work ) / space->norm( *rhs )
                      << ", monitor residual=" << solver.get_linsolver_handle_original()->monitor().resid_norm()
                      << ", iterations=" << solver.get_linsolver_handle_original()->monitor().iters_performed() << '\n';
        require( solved, "ordinary Jacobian solve failed" );
        require( space->norm( *work ) < 1e-8 * space->norm( *rhs ), "ordinary true residual" );
        space->add_mul( -1.0, *exact, 1.0, *solution );
        require( space->norm( *solution ) < 1e-8, "ordinary solution error" );

        const real alpha = 2.0, gamma = 0.8, scalar_solution = 0.3;
        op->apply( *exact, *rhs );
        space->assign_mul( gamma, *rhs, -space->scalar_prod( *c, *exact ) / alpha, *d, *rhs );
        space->assign_scalar( 0.0, *solution );
        require( solver.solve( gamma, *op, alpha, *c, *d, *rhs, *solution ), "rank-one Jacobian solve failed" );
        op->apply( *solution, *work );
        space->assign_mul( gamma, *work, -space->scalar_prod( *c, *solution ) / alpha, *d, *work );
        space->add_mul( -1.0, *rhs, 1.0, *work );
        require( space->norm( *work ) < 1e-8 * space->norm( *rhs ), "rank-one true residual" );

        op->apply( *exact, *rhs );
        space->add_mul( scalar_solution, *d, 1.0, *rhs );
        const real beta   = space->scalar_prod( *c, *exact ) + alpha * scalar_solution;
        real       scalar = 0.0;
        space->assign_scalar( 0.0, *solution );
        require( solver.solve( *op, *c, *d, alpha, *rhs, beta, *solution, scalar ), "bordered Jacobian solve failed" );
        op->apply( *solution, *work );
        space->add_mul( scalar, *d, 1.0, *work );
        space->add_mul( -1.0, *rhs, 1.0, *work );
        require( space->norm( *work ) < 1e-8 * space->norm( *rhs ), "bordered true residual" );
        require( std::abs( space->scalar_prod( *c, *solution ) + alpha * scalar - beta ) < 1e-9, "border residual" );
        require( std::abs( scalar - scalar_solution ) < 1e-8, "bordered scalar error" );
    }

    space->assign_mul( 1.001, *state, *candidate );
    auto *space_ptr  = space.get();
    auto *op_ptr     = op.get();
    auto *solver_ptr = &solver;
    nonlinear_operators::system_operator<vector_space, problem_type, operator_type, solver_type> correction(
        space_ptr, op_ptr, solver_ptr
    );
    for ( int iteration = 0; iteration < 6; ++iteration )
    {
        problem->F( *candidate, 6.0, *rhs );
        if ( space->norm( *rhs ) < 1e-8 )
            break;
        space->assign_scalar( 0.0, *solution );
        require( correction.solve( problem.get(), *candidate, 6.0, *solution ), "nonzero Newton correction failed" );
        space->add_mul( 1.0, *solution, 1.0, *candidate );
    }
    problem->F( *candidate, 6.0, *rhs );
    require( space->norm( *rhs ) < 1e-8, "nonzero Newton correction did not converge" );
    space->add_mul( -1.0, *state, 1.0, *candidate );
    require( space->norm( *candidate ) < 1e-7, "Newton changed the reference branch" );

    problem->set_linearization_point( *state, 6.0 );
    op->apply( *exact, *rhs );
    space->assign_scalar( 0.0, *solution );
    solver.get_linsolver_handle_original()->monitor().set_temp_max_iterations( 0 );
    require( !solver.solve( *op, *rhs, *solution ), "iteration exhaustion reported success" );
    solver.get_linsolver_handle_original()->monitor().restore_max_iterations();
    require( solver.solve( *op, *rhs, *solution ), "solver did not recover after restoring iteration budget" );

    auto failing_prec = std::make_shared<failing_preconditioner>( problem );
    nmfd::solvers::sherman_morrison<
        vector_space, monitor_type, log_t, operator_type, failing_preconditioner, nmfd::solvers::bicgstabl>
        recovery_solver( space, op, failing_prec );
    configure( recovery_solver );
    recovery_solver.get_linsolver_handle_original()->monitor().set_temp_max_iterations( 10000 );
    int retries = 0;
    require(
        numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &recovery_solver, [&]() { return recovery_solver.solve( *op, *rhs, *solution ); },
            [&]() { space->assign_scalar( 0.0, *solution ); }, [&]() { ++retries; }
        ),
        "unpreconditioned recovery failed"
    );
    require(
        retries == 1 && failing_prec->failed_applications == 1, "preconditioner failure did not trigger one retry"
    );
    require( !recovery_solver.preconditioner_bypass_enabled(), "recovery did not restore preconditioning" );
    op->apply( *solution, *work );
    space->add_mul( -1.0, *rhs, 1.0, *work );
    require( space->norm( *work ) < 1e-8 * space->norm( *rhs ), "recovery true residual" );
}

void test_lifetimes()
{
    auto                         space        = std::make_shared<vector_space>();
    auto                         problem      = std::make_shared<problem_type>( 2.0, 4.0, 128, space );
    auto                         op           = std::make_shared<operator_type>( problem );
    auto                         prec         = std::make_shared<preconditioner_type>( problem );
    auto                         provider     = std::make_shared<provider_type>( space, problem );
    std::weak_ptr<vector_space>  weak_space   = space;
    std::weak_ptr<problem_type>  weak_problem = problem;
    std::weak_ptr<operator_type> weak_op      = op;
    prec->set_operator( op );
    op.reset();
    problem.reset();
    space.reset();
    require( !weak_space.expired() && !weak_problem.expired() && !weak_op.expired(), "retained dependencies expired" );
    prec->set_operator( std::shared_ptr<const operator_type>{} );
    require( weak_op.expired(), "preconditioner rebinding retained the old operator" );
    prec.reset();
    require( !weak_problem.expired(), "affine inverse provider lost its problem" );
    provider.reset();
    require( weak_space.expired() && weak_problem.expired(), "KS1D dependency cycle" );
}

} // namespace

int main()
{
    try
    {
#if defined( KS1D_VECTOR_BACKEND_CUDA )
        common::init_cuda_from_scfd_selector( "auto" );
#endif
        test_construction_failure();
        auto space = std::make_shared<vector_space>();
        {
            auto problem = std::make_shared<problem_type>( 2.0, 4.0, 128, space );
            test_solves( space, problem );
        }
        require( space->allocations == space->frees, "KS1D solver workspace leak" );
        test_lifetimes();
        std::cout << "KS1D NMFD linear solvers: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "KS1D NMFD linear solvers failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
