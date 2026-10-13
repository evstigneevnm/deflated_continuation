#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
#endif
#include <common/scfd_vector_operations.h>
#include <scfd/utils/log_std.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

void require( bool passed, const char *message )
{
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
    std::shared_ptr<jacobi_operator_type> jacobi;

    explicit affine_problem( Operations &ops )
        : operations( ops ), jacobi( std::make_shared<jacobi_operator_type>( jacobi_operator_type{ ops } ) )
    {
    }
    void apply( const vector_type &x, vector_type &out )
    {
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

template <class T>
void test()
{
    using operations_type = scfd_vector_operations<backend_type, T>;
    using problem_type    = affine_problem<operations_type>;
    using linear_type     = diagonal_solver<operations_type, problem_type>;
    using iteration_type  = nmfd::solvers::newton_iteration<operations_type, problem_type, linear_type>;
    using strategy_type =
        nmfd::solvers::default_convergence_strategy<operations_type, scfd::utils::log_std, problem_type>;
    using solver_type =
        nmfd::solvers::nonlinear_solver<operations_type, scfd::utils::log_std, problem_type, iteration_type>;
    auto                           operations = std::make_shared<operations_type>( 1024 );
    auto                           linear     = std::make_shared<linear_type>( linear_type{ *operations } );
    auto                           iteration  = std::make_shared<iteration_type>( operations, linear );
    typename strategy_type::params p;
    p.check_linear_solver_fail = true;
    auto         strategy      = std::make_shared<strategy_type>( operations, nullptr, p );
    solver_type  solver( operations, nullptr, iteration, strategy );
    problem_type problem( *operations );
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
            operations->norm_l2( *residual ) == T( 0 ) && operations->check_is_valid_number( *state ),
            "backend Newton root is incorrect"
        );
    }
    for ( int expected : { 2, 6, 7 } )
    {
        problem.invalid_residual   = expected == 2;
        linear->converged          = expected != 6;
        linear->invalid_correction = expected == 7;
        require(
            !solver.solve( &problem, nullptr, nullptr, *initial, *state ) && strategy->get_result_status() == expected,
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
}

int main()
{
    try
    {
        test<float>();
        test<double>();
        std::printf( "28 backend checks, 0 failures\n" );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Newton backend test failed: %s\n", error.what() );
        return 1;
    }
}
