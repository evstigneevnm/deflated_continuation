#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <scfd/utils/log_std.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class Operations>
struct affine_problem
{
    using scalar_type = typename Operations::scalar_type;
    using vector_type = typename Operations::vector_type;
    struct jacobi_operator_type
    {
        Operations &operations;
        void        apply( const vector_type &x, vector_type &out ) const
        {
            operations.assign_lin_comb( scalar_type( 2 ), x, out );
        }
    };

    Operations                           &operations;
    bool                                  invalid_residual = false;
    unsigned int                          residual_calls   = 0;
    std::shared_ptr<jacobi_operator_type> jacobi;

    explicit affine_problem( Operations &ops )
        : operations( ops ), jacobi( std::make_shared<jacobi_operator_type>( jacobi_operator_type{ ops } ) )
    {
    }
    void apply( const vector_type &x, vector_type &out )
    {
        ++residual_calls;
        if ( invalid_residual )
            operations.assign_scalar( std::numeric_limits<scalar_type>::quiet_NaN(), out );
        else
        {
            operations.assign_lin_comb( scalar_type( 2 ), x, out );
            operations.add_mul_scalar( scalar_type( -2 ), scalar_type( 1 ), out );
        }
    }
    void set_linearization_point( const vector_type & )
    {
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi;
    }
};

template <class Operations, class Problem>
struct diagonal_solver
{
    using scalar_type = typename Operations::scalar_type;
    using vector_type = typename Operations::vector_type;
    Operations  &operations;
    bool         converged = true, invalid_correction = false;
    unsigned int calls = 0;

    void set_operator( std::shared_ptr<const typename Problem::jacobi_operator_type> )
    {
    }
    bool solve( const vector_type &rhs, vector_type &out )
    {
        ++calls;
        if ( invalid_correction )
            operations.assign_scalar( std::numeric_limits<scalar_type>::quiet_NaN(), out );
        else
            operations.assign_lin_comb( scalar_type( .5 ), rhs, out );
        return converged;
    }
};

template <class Operations>
struct trial_projection
{
    using scalar_type = typename Operations::scalar_type;
    using vector_type = typename Operations::vector_type;
    Operations  &operations;
    unsigned int calls = 0, invalid_trials = 0;

    void apply( vector_type &x )
    {
        if ( ++calls <= invalid_trials )
            operations.assign_scalar( std::numeric_limits<scalar_type>::infinity(), x );
    }
};

template <class T>
void test( bool full_check )
{
    using operations_type = nmfd::operations::scfd_vector_space<backend_type, T>;
    using problem_type    = affine_problem<operations_type>;
    using linear_type     = diagonal_solver<operations_type, problem_type>;
    using projection_type = trial_projection<operations_type>;
    using iteration_type  = nmfd::solvers::newton_iteration<operations_type, problem_type, linear_type>;
    using strategy_type   = nmfd::solvers::default_convergence_strategy<
          operations_type, scfd::utils::log_std, problem_type, projection_type>;
    using solver_type = nmfd::solvers::nonlinear_solver<
        operations_type, scfd::utils::log_std, problem_type, iteration_type, projection_type>;
    auto                           operations = std::make_shared<operations_type>( 1024 );
    auto                           linear     = std::make_shared<linear_type>( linear_type{ *operations } );
    auto                           iteration  = std::make_shared<iteration_type>( operations, linear );
    typename strategy_type::params p;
    p.perform_full_check      = full_check;
    p.max_backtracking_trials = 3;
    auto        strategy      = std::make_shared<strategy_type>( operations, nullptr, p );
    solver_type solver( operations, nullptr, iteration, strategy );
    auto        problem_owner = std::make_shared<problem_type>( *operations );
    auto       &problem       = *problem_owner;
    nmfd::detail::vector_wrap<operations_type, false, false> initial( *operations ), state( *operations ),
        residual( *operations );
    initial.init();
    initial.start_use();
    state.init();
    state.start_use();
    residual.init();
    residual.start_use();
    operations->assign_scalar( T( 2 ), *initial );
    for ( unsigned int attempt = 0; attempt < 3; ++attempt )
    {
        require( solver.solve( &problem, nullptr, nullptr, *initial, *state ), "backend Newton solve failed" );
        problem.apply( *state, *residual );
        require(
            operations->norm_l2( *residual ) == T( 0 ) && operations->is_valid_number( *state ),
            "backend Newton root is incorrect"
        );
    }
    for ( int expected : { 2, 6, 7 } )
    {
        problem.invalid_residual   = expected == 2;
        linear->converged          = expected != 6;
        linear->invalid_correction = expected == 7;
        const int status           = expected == 7 && !full_check ? 3 : expected;
        require(
            !solver.solve( &problem, nullptr, nullptr, *initial, *state ) && strategy->get_result_status() == status,
            "backend failure status is incorrect"
        );
        operations->assign_lin_comb( T( 1 ), *state, T( -1 ), *initial, *residual );
        require( operations->norm_l2( *residual ) == T( 0 ), "backend failure did not restore initial state" );
    }
    problem.invalid_residual = linear->invalid_correction = false;
    linear->converged                                     = true;
    require( solver.solve( &problem, nullptr, nullptr, *initial, *state ), "backend solver restart failed" );
    problem.apply( *state, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend restart root is incorrect" );

    for ( T value :
          { std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity() } )
    {
        operations->assign_scalar( value, *initial );
        const auto calls = linear->calls, residual_calls = problem.residual_calls;
        require(
            !solver.solve( &problem, nullptr, nullptr, *initial, *state ) &&
                strategy->get_result_status() == ( full_check ? 11 : 2 ) && linear->calls == calls,
            "backend invalid input did not terminate before the linear solve"
        );
        require(
            problem.residual_calls == residual_calls + ( full_check ? 0u : 1u ) &&
                !operations->is_valid_number( *state ),
            "backend input validation evaluated an invalid state or failed to restore it"
        );
    }
    operations->assign_scalar( T( 2 ), *initial );
    auto       projection_owner = std::make_shared<projection_type>( projection_type{ *operations, 0, 3 } );
    auto      &projection       = *projection_owner;
    const auto residual_calls   = problem.residual_calls;
    require(
        !solver.solve( &problem, &projection, nullptr, *initial, *state ) &&
            strategy->get_result_status() == ( full_check ? 10 : 3 ) && projection.calls == 3,
        "backend projection failure has an incorrect status or retry count"
    );
    require(
        problem.residual_calls == residual_calls + ( full_check ? 3u : 6u ),
        "backend validation did not bypass residual evaluation after invalid projection"
    );
    operations->assign_lin_comb( T( 1 ), *state, T( -1 ), *initial, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend projection failure did not restore input" );
    projection.calls          = 0;
    projection.invalid_trials = 1;
    require(
        solver.solve( &problem, &projection, nullptr, *initial, *state ) && projection.calls == 3,
        "backend failed to recover from a transient invalid projection"
    );
    problem.apply( *state, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend projection recovery root is incorrect" );

    solver.set_nonlinear_operator( problem_owner );
    require( solver.solve( *initial, *state ), "backend bound out-of-place solve failed" );
    problem.apply( *state, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend bound solve root is incorrect" );
    const auto projection_calls = projection.calls;
    solver.set_project_operator( projection_owner );
    operations->assign( *initial, *state );
    require(
        solver.solve( *state ) && projection.calls == projection_calls + 1,
        "backend bound in-place solve did not use the projection"
    );
    solver.set_project_operator( nullptr );
    require(
        solver.solve( *initial, *state ) && projection.calls == projection_calls + 1,
        "backend optional projection was not cleared"
    );
    linear->converged = false;
    require(
        !solver.solve( *initial, *state ) && strategy->get_result_status() == 6,
        "backend bound solve lost the linear failure status"
    );
    operations->assign_lin_comb( T( 1 ), *state, T( -1 ), *initial, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend bound failure did not restore input" );
    linear->converged = true;
    require( solver.solve( *initial, *state ), "backend bound solve did not restart after failure" );

    auto                        owned_problem    = std::make_shared<problem_type>( *operations );
    auto                        owned_projection = std::make_shared<projection_type>( projection_type{ *operations } );
    std::weak_ptr<problem_type> retained_problem = owned_problem;
    std::weak_ptr<projection_type> retained_projection = owned_projection;
    solver.set_nonlinear_operator( owned_problem );
    solver.set_project_operator( owned_projection );
    owned_problem.reset();
    owned_projection.reset();
    require(
        !retained_problem.expired() && !retained_projection.expired() && solver.solve( *initial, *state ),
        "backend shared bindings did not preserve lifetime or solve correctly"
    );
    problem.apply( *state, *residual );
    require( operations->norm_l2( *residual ) == T( 0 ), "backend shared solve root is incorrect" );
    solver.set_project_operator( nullptr );
    solver.set_nonlinear_operator( problem_owner );
    require(
        retained_problem.expired() && retained_projection.expired() && solver.solve( *initial, *state ),
        "backend shared rebinding did not release shared owners"
    );

    p.abs_tol                    = T( 0 );
    p.rel_tol                    = T( .125 );
    p.relax_tolerance_factor     = T( 4 );
    p.max_iters_num              = 1;
    p.newton_weight_initial      = T( .5 );
    auto        relaxed_strategy = std::make_shared<strategy_type>( operations, nullptr, p );
    solver_type relaxed_solver( operations, nullptr, iteration, relaxed_strategy );
    problem.apply( *initial, *residual );
    const T initial_norm = operations->norm_l2( *residual );
    require(
        relaxed_solver.solve( &problem, nullptr, nullptr, *initial, *state ) &&
            relaxed_strategy->used_relaxed_tolerance() && relaxed_strategy->get_result_status() == 0,
        "backend final correction was not accepted at relaxed tolerance"
    );
    problem.apply( *state, *residual );
    require( operations->norm_l2( *residual ) == T( .5 ) * initial_norm, "backend relaxed state is incorrect" );
    operations->assign_scalar( T( 1 ), *initial );
    require(
        relaxed_solver.solve( &problem, nullptr, nullptr, *initial, *state ) &&
            !relaxed_strategy->used_relaxed_tolerance(),
        "backend strict restart retained relaxed acceptance"
    );
}

int main()
{
    try
    {
        for ( bool full_check : { false, true } )
        {
            test<float>( full_check );
            test<double>( full_check );
        }
        std::printf( "%u backend checks, 0 failures\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Newton backend test failed: %s\n", error.what() );
        return 1;
    }
}
