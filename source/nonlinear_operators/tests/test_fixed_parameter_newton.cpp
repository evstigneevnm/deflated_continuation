#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
#else
#    include <scfd/backend/serial_cpu.h>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/fixed_parameter_operator.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <scfd/utils/log_std.h>

namespace
{
#if defined( TEST_VECTOR_BACKEND_CUDA )
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif

unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class T>
class vector_space : public nmfd::operations::scfd_vector_space<backend_type, T>
{
    using parent_type = nmfd::operations::scfd_vector_space<backend_type, T>;

public:
    using typename parent_type::vector_type;
    mutable unsigned int allocations = 0, frees = 0;

    vector_space() : parent_type( 1 )
    {
    }
    void init_vector( vector_type &x ) const
    {
        parent_type::init_vector( x );
        ++allocations;
    }
    void free_vector( vector_type &x ) const
    {
        parent_type::free_vector( x );
        ++frees;
    }
};

template <class Space, class Operator>
struct controlled_preconditioner
{
    using vector_type          = typename Space::vector_type;
    mutable unsigned int calls = 0;
    bool                 fail = false, throw_on_failure = false;
    void                 set_operator( std::shared_ptr<const Operator> )
    {
    }
    bool apply( vector_type & ) const
    {
        ++calls;
        if ( fail && throw_on_failure )
            throw std::runtime_error( "Injected preconditioner exception" );
        return !fail;
    }
};

// F(u, lambda) = u^2 + lambda^2 - 1: fixed |lambda| < 1 has two roots.
// Test both branches, including near-fold conditioning, without changing lambda.
template <class T, template <class, class, class, class, class> class LinearSolver>
void test_circle()
{
    using space_type          = vector_space<T>;
    using problem_type        = nonlinear_operators::circle<space_type>;
    using jacobian_type       = nonlinear_operators::linear_operator_circle<space_type, problem_type>;
    using operator_type       = nonlinear_operators::fixed_parameter_operator<space_type, problem_type, jacobian_type>;
    using log_type            = scfd::utils::log_std;
    using monitor_type        = nmfd::solvers::default_monitor<space_type, log_type>;
    using preconditioner_type = controlled_preconditioner<space_type, jacobian_type>;
    using linear_type         = LinearSolver<space_type, monitor_type, log_type, jacobian_type, preconditioner_type>;
    using recovery_type       = numerical_algos::recovering_linear_solver<space_type, linear_type>;
    using iteration_type      = nmfd::solvers::newton_iteration<space_type, operator_type, recovery_type>;
    using solver_type         = nmfd::solvers::nonlinear_solver<space_type, log_type, operator_type, iteration_type>;
    using strategy_type       = nmfd::solvers::default_convergence_strategy<space_type, log_type, operator_type>;
    using storage_type        = nmfd::detail::vector_wrap<space_type, false, false>;
    const T                      tolerance = std::is_same_v<T, float> ? T( 2e-6 ) : T( 2e-12 );
    auto                         space     = std::make_shared<space_type>();
    std::weak_ptr<problem_type>  problem_lifetime;
    std::weak_ptr<jacobian_type> jacobian_lifetime;
    std::weak_ptr<operator_type> operator_lifetime;
    {
        auto       problem     = std::make_shared<problem_type>( T( 1 ), 1, space );
        auto       jacobian    = std::make_shared<jacobian_type>( problem );
        const auto allocations = space->allocations;
        auto       op          = std::make_shared<operator_type>( problem, jacobian, T( .4 ) );
        require( space->allocations == allocations, "Parameter binding allocated vector storage" );
        problem_lifetime                  = problem;
        jacobian_lifetime                 = jacobian;
        operator_lifetime                 = op;
        auto                         prec = std::make_shared<preconditioner_type>();
        typename linear_type::params linear_params;
        // Keep linear accuracy tighter than Newton's target, but attainable in float.
        linear_params.monitor.abs_tol            = tolerance / T( 10 );
        linear_params.monitor.rel_tol            = tolerance / T( 10 );
        linear_params.monitor.max_iters_num      = 20;
        linear_params.use_real_resid             = true;
        auto                           linear    = std::make_shared<linear_type>( space, nullptr, linear_params, prec );
        auto                           recovery  = std::make_shared<recovery_type>( space, linear );
        auto                           iteration = std::make_shared<iteration_type>( space, recovery );
        typename strategy_type::params convergence;
        convergence.abs_tol       = tolerance;
        convergence.rel_tol       = T( 0 );
        convergence.verbose       = false;
        convergence.max_iters_num = 30;
        auto        strategy      = std::make_shared<strategy_type>( space, nullptr, convergence );
        solver_type solver( space, nullptr, iteration, strategy );
        solver.set_nonlinear_operator( op );
        storage_type initial( *space ), state( *space ), residual( *space ), direction( *space ), plus( *space ),
            minus( *space ), action( *space );
        for ( auto *storage : { &initial, &state, &residual, &direction, &plus, &minus, &action } )
        {
            storage->init();
            storage->start_use();
        }

        bool unprepared = false;
        try
        {
            op->get_jacobi_operator();
        }
        catch ( const std::logic_error & )
        {
            unprepared = true;
        }
        require( unprepared, "Unprepared Jacobian was exposed" );
        space->assign_scalar( T( 1.3 ), *initial );
        space->assign_scalar( T( .7 ), *direction );
        op->set_linearization_point( *initial );
        const auto frozen = op->get_jacobi_operator();
        require( frozen.get() == jacobian.get(), "Jacobian handle copied or replaced the existing operator" );
        const T epsilon = std::is_same_v<T, float> ? T( .001 ) : T( 1e-5 );
        space->assign_lin_comb( T( 1 ), *initial, epsilon, *direction, *plus );
        space->assign_lin_comb( T( 1 ), *initial, -epsilon, *direction, *minus );
        op->apply( *plus, *state );
        op->apply( *minus, *residual );
        space->assign_lin_comb( T( .5 ) / epsilon, *state, -T( .5 ) / epsilon, *residual, *plus );
        frozen->apply( *direction, *action );
        space->add_lin_comb( T( -1 ), *action, *plus );
        require( space->norm_l2( *plus ) < T( 200 ) * tolerance, "Fixed-parameter Jacobian failed finite differences" );
        space->assign_scalar( T( 4 ), *state );
        op->apply( *state, *residual );
        frozen->apply( *direction, *residual );
        space->add_lin_comb( T( -1 ), *action, *residual );
        require( space->norm_l2( *residual ) == T( 0 ), "Trial residual changed the frozen Jacobian" );

        const auto solve_allocations = space->allocations;
        for ( T parameter : { T( 0 ), T( .5 ), T( -.5 ), T( .99 ) } )
        {
            op->set_parameter( parameter );
            require( op->parameter() == parameter, "Fixed parameter was not rebound" );
            for ( T sign : { T( 1 ), T( -1 ) } )
            {
                space->assign_scalar( sign * T( 1.5 ), *initial );
                const bool converged = solver.solve( *initial, *state );
                if ( !converged )
                    std::fprintf(
                        stderr, "Circle lambda=%g, sign=%g, scalar bytes=%zu: Newton status=%d, linear residual=%g\n",
                        double( parameter ), double( sign ), sizeof( T ), strategy->get_result_status(),
                        double( linear->monitor().resid_norm() )
                    );
                require( converged, "Circle correction did not converge" );
                op->apply( *state, *residual );
                T root;
                space->get( *state, &root, 1 );
                const T expected = sign * std::sqrt( T( 1 ) - parameter * parameter );
                require(
                    std::abs( root - expected ) <= T( 20 ) * tolerance, "Circle correction reached an incorrect root"
                );
                require(
                    space->norm_l2( *residual ) <= tolerance, "Circle correction failed true residual verification"
                );
                require( op->parameter() == parameter, "Newton changed the fixed parameter" );
                require(
                    solver.solve( *state ) && strategy->get_number_of_iterations() == 0,
                    "Exact-root restart did not terminate before applying a correction"
                );
            }
        }
        require( space->allocations == solve_allocations, "Newton allocated vectors during repeated solves" );

        op->set_parameter( T( .5 ) );
        space->assign_scalar( T( 1.5 ), *initial );
        prec->fail       = true;
        const auto calls = prec->calls;
        require(
            solver.solve( *initial, *state ) && prec->calls > calls,
            "Newton did not recover from a failed preconditioner"
        );
        require( !linear->preconditioner_bypass_enabled(), "Recovery did not restore preconditioning" );
        prec->throw_on_failure = true;
        bool propagated        = false;
        try
        {
            solver.solve( *initial, *state );
        }
        catch ( const std::runtime_error & )
        {
            propagated = true;
        }
        require(
            propagated && !linear->preconditioner_bypass_enabled(),
            "Backend exception was hidden or changed bypass state"
        );
        prec->fail = prec->throw_on_failure = false;

        space->assign_scalar( T( 0 ), *initial );
        require(
            !solver.solve( *initial, *state ) && strategy->get_result_status() == 6,
            "Singular circle Jacobian did not propagate linear failure"
        );
        space->add_lin_comb( T( -1 ), *initial, *state );
        require( space->norm_l2( *state ) == T( 0 ), "Failed solve did not restore the initial state" );
        require( !linear->preconditioner_bypass_enabled(), "Failed retry left preconditioning disabled" );

        convergence.max_iters_num    = 1;
        auto        limited_strategy = std::make_shared<strategy_type>( space, nullptr, convergence );
        solver_type limited( space, nullptr, iteration, limited_strategy );
        limited.set_nonlinear_operator( op );
        space->assign_scalar( T( 2 ), *initial );
        require(
            !limited.solve( *initial, *state ) && limited_strategy->get_result_status() == 9,
            "Iteration exhaustion did not return failure"
        );
        space->add_lin_comb( T( -1 ), *initial, *state );
        require( space->norm_l2( *state ) == T( 0 ), "Exhausted Newton did not roll back" );
        require( solver.solve( *initial, *state ), "Newton did not recover after a failed solve" );

        op.reset();
        jacobian.reset();
        problem.reset();
        require(
            !operator_lifetime.expired() && !problem_lifetime.expired(), "Retained problem bindings expired early"
        );
        space->assign_scalar( T( -1.5 ), *initial );
        require( solver.solve( *initial, *state ), "Shared bindings did not preserve a usable problem" );
    }
    require(
        problem_lifetime.expired() && jacobian_lifetime.expired() && operator_lifetime.expired(),
        "Fixed-parameter Newton retained an ownership cycle"
    );
    require( space->allocations == space->frees, "Fixed-parameter Newton leaked vector storage" );
}

// Synthetic projected residual P(u, lambda) = lambda*u - 1. Raw entry points
// deliberately throw, so the test detects bypassing the existing projection hooks.
template <class Space>
struct projected_problem
{
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    scalar_type            coefficient = 0;
    void                   F( const vector_type &, scalar_type, vector_type & )
    {
        throw std::logic_error( "Raw residual selected" );
    }
    void set_linearization_point( const vector_type &, scalar_type )
    {
        throw std::logic_error( "Raw Jacobian selected" );
    }
    void projected_F( const vector_type &x, scalar_type parameter, vector_type &out )
    {
        space->assign_lin_comb( parameter, x, out );
        space->add_mul_scalar( scalar_type( -1 ), scalar_type( 1 ), out );
    }
    void set_projected_linearization_point( const vector_type &, scalar_type parameter )
    {
        coefficient = parameter;
    }
    void jacobian_u( const vector_type &x, vector_type &out )
    {
        space->assign_lin_comb( coefficient, x, out );
    }
};

void test_projection_and_validation()
{
    using space_type       = vector_space<double>;
    using problem_type     = projected_problem<space_type>;
    using jacobian_type    = nonlinear_operators::linear_operator_circle<space_type, problem_type>;
    using operator_type    = nonlinear_operators::fixed_parameter_operator<space_type, problem_type, jacobian_type>;
    auto          space    = std::make_shared<space_type>();
    auto          problem  = std::make_shared<problem_type>( problem_type{ space } );
    auto          jacobian = std::make_shared<jacobian_type>( problem );
    operator_type op( problem, jacobian, 2. );
    nmfd::detail::vector_wrap<space_type, false, false> state( *space ), out( *space );
    state.init();
    state.start_use();
    out.init();
    out.start_use();
    space->assign_scalar( 1., *state );
    op.set_linearization_point( *state );
    auto frozen = op.get_jacobi_operator();
    op.set_parameter( 3. );
    op.apply( *state, *out );
    double value;
    space->get( *out, &value, 1 );
    require( value == 2., "Parameter rebinding did not reach projected residual" );
    bool invalidated = false;
    try
    {
        op.get_jacobi_operator();
    }
    catch ( const std::logic_error & )
    {
        invalidated = true;
    }
    require( invalidated, "Parameter rebinding exposed a stale Jacobian" );
    frozen->apply( *state, *out );
    space->get( *out, &value, 1 );
    require( value == 2., "Parameter change or residual evaluation modified the prepared Jacobian" );
    op.set_linearization_point( *state );
    op.get_jacobi_operator()->apply( *state, *out );
    space->get( *out, &value, 1 );
    require( value == 3., "Projected linearization did not use the new parameter" );
    for ( bool missing_problem : { false, true } )
    {
        bool rejected = false;
        try
        {
            operator_type invalid( missing_problem ? nullptr : problem, missing_problem ? jacobian : nullptr, 0. );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Missing fixed-parameter dependency was accepted" );
    }
}
}

int main()
{
    try
    {
        test_circle<float, nmfd::solvers::cgs>();
        test_circle<double, nmfd::solvers::cgs>();
        test_circle<float, nmfd::solvers::bicgstabl>();
        test_circle<double, nmfd::solvers::bicgstabl>();
        test_projection_and_validation();
        std::printf( "Fixed-parameter NMFD Newton: %u checks passed\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Fixed-parameter NMFD Newton failed after %u checks: %s\n", checks, error.what() );
        return 1;
    }
}
