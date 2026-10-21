#include <iostream>
#include <stdexcept>
#include <string>
#include <memory>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>

#include <scfd/utils/log_std.h>

#include <numerical_algos/linear_solve_recovery.h>
#include "nmfd_cpu_reference_vector_space.h"

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

using log_type = scfd::utils::log_std;

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

        require(
            numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
                &solver, []() { return true; }, []() { throw std::runtime_error( "unexpected retry" ); }
            ) && solver.bypass_changes == 2,
            "successful primary solve does not reset output or change preconditioning"
        );
        solves = resets         = 0;
        const bool retry_failed = numerical_algos::lin_solvers::recovery::solve_with_unpreconditioned_retry(
            &solver,
            [&]() {
                ++solves;
                return false;
            },
            [&]() { ++resets; }
        );
        require( !retry_failed && solves == 2 && resets == 1, "failed retry propagates failure without looping" );
        require( !solver.bypass, "failed retry restores the configured preconditioner" );

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
    }
    catch ( const std::exception &exception )
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
