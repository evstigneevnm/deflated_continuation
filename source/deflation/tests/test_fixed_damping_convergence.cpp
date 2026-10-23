#include "deflation_newton_fixture.h"
#include <limits>
#include <main/nmfd_newton_types.h>
#include <nmfd/preconditioners/dummy.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <scfd/utils/log_std.h>

namespace
{
using deflation::tests::test_report;

// F(x)=x separates physical Euclidean stopping from RMS stopping and lets us
// prescribe growing Newton trials independently of the linear solve.
template <class Space>
struct probe_operator
{
    using T           = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    Space &space;
    bool   invalid = false;
    T      original_residual_norm( const vector_type &state, vector_type &work )
    {
        space.assign( state, work );
        return invalid ? std::numeric_limits<T>::infinity() : space.norm_l2( work );
    }
};

template <class Space>
struct probe_projector
{
    Space &space;
    void   apply( typename Space::vector_type &state )
    {
        space.assign_scalar( std::numeric_limits<typename Space::scalar_type>::infinity(), state );
    }
};

template <class T>
void test_acceptance( test_report &report )
{
    using space_type     = deflation::tests::vector_space<T>;
    using operator_type  = probe_operator<space_type>;
    using projector_type = probe_projector<space_type>;
    using quality_type   = nmfd::operations::zero_functional<space_type>;
    using strategy_type  = deflation::fixed_damping_convergence_strategy<
         space_type, scfd::utils::log_std, operator_type, projector_type, quality_type>;
    auto space = std::make_shared<space_type>( 2 );
    {
        nmfd::detail::vector_wrap<space_type, false, false> state( *space ), increment( *space ), saved( *space );
        for ( auto *buffer : { &state, &increment, &saved } )
        {
            buffer->init();
            buffer->start_use();
        }
        strategy_type strategy( space, nullptr );
        strategy.set_convergence_constants( T( 1.2 ), 40, T( .5 ), true, false );
        operator_type op{ *space };
        const auto    allocations = space->allocations;
        space->assign_scalar( T( 1 ), *state );
        space->assign_scalar( T( 2 ), *increment );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Used RMS instead of Euclidean stopping"
        );
        report.require( strategy.get_number_of_iterations() == 0, "Initialization counted as a correction" );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Rejected physical residual growth"
        );
        T values[2];
        space->get( *state, values, 2 );
        report.require( values[0] == T( 2 ) && values[1] == T( 2 ), "Changed the prescribed damping weight" );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ), "Second growing trial failed"
        );
        space->get( *state, values, 2 );
        report.require( values[0] == T( 3 ), "Weight changed between corrections" );
        space->assign_scalar( T( 0 ), *increment );
        for ( int i = 0; i < 15; ++i )
            report.require(
                !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
                "Applied stagnation termination"
            );
        report.require( strategy.get_norms_history_handle()->size() == 17, "Incorrect accepted-iteration history" );

        strategy.set_convergence_constants( T( 1e-8 ), 20, T( .5 ), false, false );
        strategy.reset_iterations();
        space->assign_scalar( T( 1e-4 ), *state );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Growth test initialized as converged"
        );
        space->assign_scalar( T( 1e15 ), *increment );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Applied a relative growth ceiling"
        );
        space->assign( *state, *saved );
        space->assign_scalar( T( 4e16 ), *increment );
        report.require(
            strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 12,
            "Ignored absolute residual ceiling"
        );
        space->add_lin_comb( T( -1 ), *saved, *state );
        report.require( space->norm( *state ) == T( 0 ), "Failed trial replaced committed state" );

        strategy.reset_iterations();
        space->assign_scalar( T( 1 ), *state );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Failed to reset after ceiling failure"
        );
        strategy.increment_ok( false );
        report.require(
            strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 6,
            "Ignored linear failure"
        );
        strategy.reset_iterations();
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ), "Failed linear-status reset"
        );
        space->assign_scalar( std::numeric_limits<T>::infinity(), *increment );
        report.require(
            strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 7,
            "Accepted nonfinite increment"
        );

        strategy.reset_iterations();
        space->assign_scalar( T( 0 ), *increment );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Invalid-projector initialization failed"
        );
        projector_type projector{ *space };
        report.require(
            strategy.check_convergence( &op, &projector, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 10,
            "Accepted nonfinite projection"
        );
        space->get( *state, values, 2 );
        report.require( values[0] == T( 1 ), "Invalid projection changed committed state" );
        strategy.reset_iterations();
        op.invalid = true;
        report.require(
            strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 2,
            "Ignored invalid residual or exclusion flag"
        );
        op.invalid = false;
        strategy.reset_iterations();
        space->assign_scalar( std::numeric_limits<T>::quiet_NaN(), *state );
        report.require(
            strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ) &&
                strategy.get_result_status() == 11,
            "Accepted invalid initial state"
        );

        for ( T weight : { T( 0 ), T( -1 ), T( 2 ), std::numeric_limits<T>::quiet_NaN() } )
        {
            bool rejected = false;
            try
            {
                strategy.set_convergence_constants( T( 2 ), 1, weight );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            report.require( rejected, "Accepted invalid damping configuration" );
        }
        strategy.reset_iterations();
        space->assign_scalar( T( 1 ), *state );
        report.require(
            !strategy.check_convergence( &op, nullptr, nullptr, *state, *increment ),
            "Rejected setter changed tolerance"
        );
        report.require( strategy.get_norms_history_handle()->empty(), "Reset retained convergence history" );
        for ( T tolerance : { T( -1 ), std::numeric_limits<T>::infinity() } )
        {
            bool rejected = false;
            try
            {
                strategy.set_convergence_constants( tolerance, 20, T( .5 ) );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            report.require( rejected, "Accepted invalid tolerance" );
        }
        bool rejected = false;
        try
        {
            strategy.set_convergence_constants( T( 1e-8 ), 0, T( .5 ) );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.require( rejected, "Accepted a zero iteration limit" );
        typename strategy_type::params p;
        p.max_iters_num = std::numeric_limits<unsigned int>::max();
        rejected        = false;
        try
        {
            strategy_type invalid( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.require( rejected, "Accepted an invalid constructor limit" );
        report.require( space->allocations == allocations, "Strategy allocated vectors while iterating" );
    }
    report.require( space->allocations == space->frees, "Strategy leaked scratch vectors" );
    {
        typename strategy_type::utils_hierarchy  u( space );
        typename strategy_type::params_hierarchy p;
        strategy_type                            constructed( u, p );
        report.require( constructed.get_result_status() == -1, "Hierarchy construction did not initialize status" );
#ifdef NMFD_ENABLE_NLOHMANN
        auto j      = p.to_json();
        j["weight"] = T( .25 );
        p.from_json( j );
        report.require( p.weight == T( .25 ), "Hierarchy JSON configuration lost damping" );
        j["max_iters_num"] = -1;
        bool rejected      = false;
        try
        {
            p.from_json( j );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.require( rejected && p.max_iters_num == 1000, "Invalid JSON changed parameters" );
#endif
    }
    report.require( space->allocations == space->frees, "Hierarchy construction leaked vectors" );
    space->fail_activation = space->activations + 2;
    bool failed            = false;
    try
    {
        strategy_type strategy( space, nullptr );
    }
    catch ( const std::runtime_error & )
    {
        failed = true;
    }
    report.require( failed && space->allocations == space->frees, "Construction failure leaked scratch vectors" );
}

// Deflated circle corrections must escape the stored positive root using the
// ordinary NMFD iteration/rank-one solver, while failed solves roll back.
template <class T>
void test_integration( test_report &report )
{
    using fixture_type        = deflation::tests::circle_fixture<T>;
    using space_type          = typename fixture_type::space_type;
    using log_type            = scfd::utils::log_std;
    using monitor_type        = nmfd::solvers::default_monitor<space_type, log_type>;
    using state_operator_type = typename fixture_type::state_operator_type;
    using prec_type           = nmfd::preconditioners::dummy<space_type, state_operator_type>;
    using system_type         = nmfd::solvers::sherman_morrison<
                space_type, monitor_type, log_type, state_operator_type, prec_type, nmfd::solvers::bicgstabl>;
    using default_types = main_classes::nmfd_newton_types<
        space_type, log_type, typename fixture_type::problem_type, state_operator_type, system_type,
        typename fixture_type::solutions_type>;
    using types = main_classes::nmfd_newton_types<
        space_type, log_type, typename fixture_type::problem_type, state_operator_type, system_type,
        typename fixture_type::solutions_type, main_classes::detail::newton_convergence,
        main_classes::detail::newton_convergence, main_classes::detail::fixed_damping_deflation_convergence>;
    static_assert(
        std::is_same_v<typename types::fixed_convergence_type, typename default_types::fixed_convergence_type>
    );
    static_assert(
        std::is_same_v<
            typename types::continuation_convergence_type, typename default_types::continuation_convergence_type>
    );
    static_assert(
        !std::is_same_v<typename types::deflation_convergence_type, typename default_types::deflation_convergence_type>
    );
    std::shared_ptr<space_type>                         space;
    std::weak_ptr<typename fixture_type::operator_type> lifetime;
    {
        fixture_type f;
        space    = f.space;
        lifetime = f.op;
        auto system =
            std::make_shared<system_type>( space, f.state_operator, std::make_shared<prec_type>( space ), nullptr );
        system->get_linsolver_handle()->monitor().set_tolerances(
            fixture_type::tolerance / T( 10 ), fixture_type::tolerance / T( 10 )
        );
        auto method = types::deflation_method::make(
            space, f.problem, f.op, space, std::make_shared<typename types::rank_one_type>( system ), nullptr,
            std::make_shared<typename types::deflated_operator_type::trial_projector>( f.problem )
        );
        struct settings
        {
            T    tolerance = fixture_type::tolerance, newton_wight = T( .5 );
            int  newton_max_it       = 100;
            bool store_norms_history = true, verbose = false;
        } p;
        types::configure_deflation( method->get_convergence_strategy_handle(), p );
        space->assign_scalar( T( 1 ), *f.initial );
        f.solutions->push_back( *f.initial );
        space->assign_scalar( T( .8 ), *f.initial );
        T          parameter   = T( 0 );
        const auto allocations = space->allocations;
        report.require(
            method->solve( f.problem.get(), *f.initial, parameter, *f.state, parameter ),
            "Fixed-damping deflation failed"
        );
        report.require(
            std::abs( f.value( *f.state ) + T( 1 ) ) < T( 10 ) * fixture_type::tolerance, "Returned the excluded root"
        );
        report.require(
            method->solve( f.problem.get(), *f.state, parameter ) &&
                method->get_convergence_strategy_handle()->get_number_of_iterations() == 0,
            "Admissible-root restart failed"
        );

        auto *strategy = method->get_convergence_strategy_handle();
        strategy->reset_iterations();
        space->assign_scalar( T( .8 ), *f.state );
        space->assign_scalar( T( .4 ), *f.direction );
        report.require(
            !strategy->check_convergence( f.op.get(), nullptr, nullptr, *f.state, *f.direction ),
            "Excluded-trial initialization failed"
        );
        report.require(
            strategy->check_convergence( f.op.get(), nullptr, nullptr, *f.state, *f.direction ) &&
                strategy->get_result_status() == 3 && f.value( *f.state ) == T( .8 ),
            "Excluded trial changed the weight or the committed state"
        );

        space->assign_scalar( T( 1 ), *f.initial );
        report.require(
            !method->solve( f.problem.get(), *f.initial, parameter, *f.state, parameter ),
            "Accepted an excluded initial root"
        );
        report.require( f.value( *f.state ) == T( 1 ), "Excluded-root failure did not roll back" );
        p.newton_max_it = 1;
        types::configure_deflation( method->get_convergence_strategy_handle(), p );
        space->assign_scalar( T( .8 ), *f.initial );
        report.require(
            !method->solve( f.problem.get(), *f.initial, parameter, *f.state, parameter ) &&
                method->get_convergence_strategy_handle()->get_result_status() == 9,
            "Iteration limit did not fail"
        );
        report.require( f.value( *f.state ) == T( .8 ), "Exhausted solve did not roll back" );
        p.newton_max_it = 100;
        types::configure_deflation( method->get_convergence_strategy_handle(), p );
        report.require(
            method->solve( f.problem.get(), *f.initial, parameter, *f.state, parameter ),
            "Repeated solve did not recover"
        );
        report.require( space->allocations == allocations, "Correction loop allocated vectors" );
    }
    report.require( lifetime.expired(), "Strategy binding formed an ownership cycle" );
    report.require( space->allocations == space->frees, "Integrated deflation leaked vectors" );
}
}

int main()
{
    test_report report;
    try
    {
        test_acceptance<float>( report );
        test_acceptance<double>( report );
        test_integration<float>( report );
        test_integration<double>( report );
        std::printf( "Fixed-damping deflation: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Fixed-damping deflation failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
