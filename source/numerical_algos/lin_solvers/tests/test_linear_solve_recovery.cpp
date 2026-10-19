#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>

#include <scfd/utils/log_std.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/iter_solver_base.h>
#include <numerical_algos/lin_solvers/linear_solve_recovery.h>
#include <numerical_algos/lin_solvers/tests/nmfd_cpu_reference_vector_space.h>

namespace
{

void require( const bool condition, const std::string &message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

struct recoverable_solver
{
    void set_preconditioner_bypass( const bool value )
    {
        bypass = value;
        ++bypass_changes;
    }

    bool         bypass         = false;
    unsigned int bypass_changes = 0;
};

struct unsupported_solver
{
};

using vector_space_type = nmfd::tests::cpu_reference_vector_space<double>;

struct identity_operator
{
    using vector_type = vector_space_type::vector_type;

    void apply( const vector_type &input, vector_type &output ) const
    {
        output = input;
    }
};

struct identity_preconditioner
{
    using vector_type = vector_space_type::vector_type;

    void set_operator( const identity_operator * )
    {
    }

    void apply( vector_type & ) const
    {
    }
};

using log_type     = scfd::utils::log_std;
using monitor_type = numerical_algos::lin_solvers::default_monitor<vector_space_type, log_type>;

class base_recoverable_solver
    : public numerical_algos::lin_solvers::iter_solver_base<
          identity_operator, identity_preconditioner, vector_space_type, monitor_type, log_type>
{
    using base_type = numerical_algos::lin_solvers::iter_solver_base<
        identity_operator, identity_preconditioner, vector_space_type, monitor_type, log_type>;

public:
    base_recoverable_solver( const vector_space_type *vector_space, log_type *log )
        : base_type( vector_space, log, 0, "base_recovery_test::" )
    {
    }

    bool solve( const identity_operator &, const vector_type &, vector_type & ) const override
    {
        ++solve_count;
        return this->prec_ == nullptr;
    }

    bool has_preconditioner() const
    {
        return this->prec_ != nullptr;
    }

    mutable unsigned int solve_count = 0;
};

struct failing_preconditioner
{
    void set_operator( std::shared_ptr<const identity_operator> )
    {
    }
    bool apply( vector_space_type::vector_type & ) const
    {
        return false;
    }
};

void test_nmfd_recovery()
{
    using monitor       = nmfd::solvers::default_monitor<vector_space_type, log_type>;
    using linear_system = nmfd::solvers::sherman_morrison<
        vector_space_type, monitor, log_type, identity_operator, failing_preconditioner, nmfd::solvers::cgs>;
    auto                           space = std::make_shared<vector_space_type>( 1 );
    auto                           op    = std::make_shared<identity_operator>();
    auto                           prec  = std::make_shared<failing_preconditioner>();
    linear_system                  solver( space, op, prec );
    vector_space_type::vector_type c{ 1. }, d{ 1. }, rhs{ 1. }, x{ 0. };
    for ( int mode = 0; mode < 3; ++mode )
    {
        rhs[0]       = mode == 0 ? 1. : mode == 1 ? 3. : 4.;
        double v     = 0.;
        auto   solve = [&]() {
            if ( mode == 0 )
                return solver.solve( *op, rhs, x );
            if ( mode == 1 )
                return solver.solve( 2., *op, 2., c, d, rhs, x );
            return solver.solve( *op, c, d, 2., rhs, 5., x, v );
        };
        unsigned int resets = 0;
        const bool   recovered =
            numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry( &solver, solve, [&]() {
                x[0] = 0.;
                ++resets;
            } );
        require( recovered && resets == 1, "NMFD ordinary/rank-one/bordered unpreconditioned retry succeeds" );
        require( std::abs( x[0] - double( mode + 1 ) ) < 1e-12, "NMFD retry gives the analytical solution" );
        require( mode != 2 || std::abs( v - 1. ) < 1e-12, "NMFD retry recovers bordered scalar" );
        require( !solver.preconditioner_bypass_enabled(), "NMFD retry restores both preconditioners" );
    }
    bool threw = false;
    try
    {
        numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &solver, [&]() { return solver.solve( *op, rhs, x ); },
            []() { throw std::runtime_error( "injected retry failure" ); }
        );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    require( threw && !solver.preconditioner_bypass_enabled(), "NMFD retry restores bypass after an exception" );
}

} // namespace

int main()
{
    try
    {
        test_nmfd_recovery();
        recoverable_solver solver;
        unsigned int       solves    = 0;
        unsigned int       resets    = 0;
        unsigned int       retries   = 0;
        const bool         recovered = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &solver,
            [&]() {
                ++solves;
                return solver.bypass;
            },
            [&]() { ++resets; }, [&]() { ++retries; }
        );
        require( recovered, "unpreconditioned retry succeeds" );
        require( solves == 2, "primary and retry are both executed" );
        require( resets == 1, "retry output is reset once" );
        require( retries == 1, "retry callback executes once" );
        require( !solver.bypass, "bypass state is restored" );
        require( solver.bypass_changes == 2, "bypass is enabled and disabled transactionally" );

        unsupported_solver unsupported;
        solves                        = 0;
        const bool unsupported_result = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &unsupported,
            [&]() {
                ++solves;
                return false;
            },
            []() {}
        );
        require( !unsupported_result && solves == 1, "unsupported solvers retain their original path" );

        vector_space_type       vector_space( 1 );
        log_type                log;
        identity_operator       linear_operator;
        identity_preconditioner preconditioner;
        base_recoverable_solver base_solver( &vector_space, &log );
        base_solver.set_preconditioner( &preconditioner );
        vector_space_type::vector_type rhs( 1, 1.0 );
        vector_space_type::vector_type solution( 1, 0.0 );
        const bool base_recovered = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &base_solver, [&]() { return base_solver.solve( linear_operator, rhs, solution ); },
            [&]() { solution.assign( 1, 0.0 ); }
        );
        require(
            base_recovered && base_solver.solve_count == 2, "iterative solver base retries without preconditioning"
        );
        require(
            base_solver.has_preconditioner() && !base_solver.preconditioner_bypass_enabled(),
            "iterative solver base restores its configured preconditioner"
        );
    }
    catch ( const std::exception &exception )
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
