#include "deflation_newton_fixture.h"
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <deflation/rank_one_linear_solver.h>
#include <scfd/utils/log_std.h>

namespace
{
using deflation::tests::test_report;

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

// At fixed lambda the circle has two equilibria. Deflate the positive one and
// find the negative one, retaining physical-residual convergence and rank-one recovery.
template <class T, template <class, class, class, class, class> class LinearSolver>
void test_newton( test_report &report )
{
    using fixture_type        = deflation::tests::circle_fixture<T>;
    using space_type          = typename fixture_type::space_type;
    using state_operator_type = typename fixture_type::state_operator_type;
    using operator_type       = typename fixture_type::operator_type;
    using log_type            = scfd::utils::log_std;
    using monitor_type        = nmfd::solvers::default_monitor<space_type, log_type>;
    using preconditioner_type = controlled_preconditioner<space_type, state_operator_type>;
    using system_type         = nmfd::solvers::sherman_morrison<
                space_type, monitor_type, log_type, state_operator_type, preconditioner_type, LinearSolver>;
    using binding_type   = deflation::rank_one_linear_solver<typename operator_type::jacobi_operator_type, system_type>;
    using recovery_type  = numerical_algos::recovering_linear_solver<space_type, binding_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, operator_type, recovery_type>;
    using projector_type = typename operator_type::trial_projector;
    using quality_type   = nmfd::operations::zero_functional<space_type>;
    using strategy_type  = nmfd::solvers::default_convergence_strategy<
         space_type, log_type, operator_type, projector_type, quality_type, deflation::original_residual_norm>;
    using solver_type = nmfd::solvers::nonlinear_solver<
        space_type, log_type, operator_type, iteration_type, projector_type, quality_type, strategy_type>;
    std::shared_ptr<space_type>                          space;
    std::weak_ptr<operator_type>                         operator_lifetime;
    std::weak_ptr<typename fixture_type::solutions_type> storage_lifetime;
    {
        fixture_type f;
        space             = f.space;
        operator_lifetime = f.op;
        storage_lifetime  = f.solutions;
        const T tolerance = fixture_type::tolerance;
        auto    prec      = std::make_shared<preconditioner_type>();
        auto    system    = std::make_shared<system_type>( f.space, f.state_operator, prec, nullptr );
        system->get_linsolver_handle()->monitor().set_tolerances( tolerance / T( 10 ), tolerance / T( 10 ) );
        system->get_linsolver_handle_original()->monitor().set_tolerances( tolerance / T( 10 ), tolerance / T( 10 ) );
        auto                           binding   = std::make_shared<binding_type>( system );
        auto                           recovery  = std::make_shared<recovery_type>( f.space, binding );
        auto                           iteration = std::make_shared<iteration_type>( f.space, recovery );
        typename strategy_type::params convergence;
        convergence.abs_tol       = tolerance;
        convergence.max_iters_num = 40;
        convergence.verbose       = false;
        // Deflation can initially increase ||F|| when leaving an excluded root.
        // The old deflation corrector does not require monotone physical residuals.
        convergence.maximum_norm_increase = T( 1e6 );
        auto        strategy              = std::make_shared<strategy_type>( f.space, nullptr, convergence );
        solver_type solver( f.space, nullptr, iteration, strategy );
        solver.set_nonlinear_operator( f.op );
        solver.set_project_operator( std::make_shared<projector_type>( f.problem ) );

        bool unbound = false;
        try
        {
            binding->solve( *f.residual, *f.action );
        }
        catch ( const std::logic_error & )
        {
            unbound = true;
        }
        report.require( unbound, "Unbound rank-one solver was usable" );
        f.space->assign_scalar( T( 1 ), *f.initial );
        f.solutions->push_back( *f.initial );
        f.space->assign_scalar( T( .8 ), *f.state );
        f.op->set_linearization_point( *f.state );
        const auto jacobian = f.op->get_jacobi_operator();
        binding->set_operator( jacobian );
        f.space->assign_scalar( T( .3 ), *f.residual );
        f.space->assign_scalar( T( 0 ), *f.action );
        report.require( recovery->solve( *f.residual, *f.action ), "Rank-one solve failed" );
        const T expected = T( .3 ) / ( T( 26 ) * T( 1.6 ) - T( -.36 ) * T( -250 ) );
        report.require(
            std::abs( f.value( *f.action ) - expected ) < T( 10 ) * tolerance,
            "Rank-one solve disagreed with assembled scalar reference"
        );
        jacobian->apply( *f.action, *f.plus );
        f.space->add_lin_comb( T( -1 ), *f.residual, *f.plus );
        report.require( f.space->norm_l2( *f.plus ) < tolerance, "Rank-one solve failed true residual check" );
        prec->fail       = true;
        const auto calls = prec->calls;
        f.space->assign_scalar( T( 0 ), *f.action );
        report.require(
            recovery->solve( *f.residual, *f.action ) && prec->calls > calls,
            "Rank-one solver did not retry failed preconditioning"
        );
        report.require( !system->preconditioner_bypass_enabled(), "Rank-one retry retained bypass" );
        prec->throw_on_failure = true;
        f.space->assign_scalar( T( 0 ), *f.action );
        bool propagated = false;
        try
        {
            recovery->solve( *f.residual, *f.action );
        }
        catch ( const std::runtime_error & )
        {
            propagated = true;
        }
        report.require( propagated && !system->preconditioner_bypass_enabled(), "Rank-one recovery hid an exception" );
        prec->fail = prec->throw_on_failure = false;

        const auto allocations = f.space->allocations;
        for ( T parameter : { T( 0 ), T( .5 ), T( -.5 ) } )
        {
            f.op->set_parameter( parameter );
            f.solutions->clear();
            const T root = std::sqrt( T( 1 ) - parameter * parameter );
            f.space->assign_scalar( root, *f.initial );
            f.solutions->push_back( *f.initial );
            f.space->assign_scalar( T( .8 ) * root, *f.initial );
            const bool converged = solver.solve( *f.initial, *f.state );
            if ( !converged )
                std::fprintf(
                    stderr, "Deflation lambda=%g, scalar bytes=%zu: Newton status=%d\n", double( parameter ),
                    sizeof( T ), strategy->get_result_status()
                );
            report.require( converged, "Deflated Newton did not converge" );
            report.require(
                std::abs( f.value( *f.state ) + root ) < T( 10 ) * tolerance, "Deflation returned the excluded root"
            );
            report.require(
                f.op->original_residual_norm( *f.state, *f.residual ) <= tolerance,
                "Deflation accepted an inaccurate physical residual"
            );
            report.require(
                solver.solve( *f.state ) && strategy->get_number_of_iterations() == 0,
                "Deflated restart at an admissible root failed"
            );
            report.require( f.op->parameter() == parameter, "Deflated Newton changed the fixed parameter" );
        }
        // Push-back owns one vector each time; correction loops themselves allocate none.
        report.require( f.space->allocations == allocations + 3, "Deflated Newton allocated scratch during solves" );

        f.op->set_parameter( T( 0 ) );
        f.solutions->clear();
        f.space->assign_scalar( T( 1 ), *f.initial );
        f.solutions->push_back( *f.initial );
        for ( T start : { T( 1 ), T( 1 ) - tolerance } )
        {
            f.space->assign_scalar( start, *f.initial );
            report.require(
                !solver.solve( *f.initial, *f.state ) && strategy->get_result_status() == 2,
                "Excluded root passed physical-residual stopping test"
            );
            f.space->add_lin_comb( T( -1 ), *f.initial, *f.state );
            report.require( f.space->norm_l2( *f.state ) == 0, "Excluded-root failure changed the initial state" );
        }
        // An excluded trial is rejected locally: halve the weight, not the committed state.
        f.space->assign_scalar( T( .8 ), *f.state );
        f.space->assign_scalar( T( .2 ), *f.action );
        strategy->reset_iterations();
        report.require(
            !strategy->check_convergence( f.op.get(), nullptr, nullptr, *f.state, *f.action ),
            "Deflation did not initialize trial assessment"
        );
        report.require(
            !strategy->check_convergence( f.op.get(), nullptr, nullptr, *f.state, *f.action ) &&
                std::abs( f.value( *f.state ) - T( .9 ) ) < tolerance,
            "Excluded trial did not trigger bounded backtracking"
        );
        // With empty storage J_F(0)=0 is singular; no rank-one term can repair it.
        f.solutions->clear();
        f.space->assign_scalar( T( 0 ), *f.initial );
        report.require(
            !solver.solve( *f.initial, *f.state ) && strategy->get_result_status() == 6,
            "Singular physical Jacobian did not report linear failure"
        );
        report.require(
            f.space->norm_l2( *f.state ) == 0 && !system->preconditioner_bypass_enabled(),
            "Failed rank-one solve did not restore state and preconditioning"
        );

        convergence.max_iters_num    = 1;
        auto        limited_strategy = std::make_shared<strategy_type>( f.space, nullptr, convergence );
        solver_type limited( f.space, nullptr, iteration, limited_strategy );
        limited.set_nonlinear_operator( f.op );
        f.space->assign_scalar( T( 1 ), *f.initial );
        f.solutions->push_back( *f.initial );
        f.space->assign_scalar( T( .8 ), *f.initial );
        report.require(
            !limited.solve( *f.initial, *f.state ) && limited_strategy->get_result_status() == 9,
            "Deflation iteration exhaustion was not reported"
        );
        f.space->add_lin_comb( T( -1 ), *f.initial, *f.state );
        report.require( f.space->norm_l2( *f.state ) == 0, "Exhausted deflation did not roll back" );
        report.require( solver.solve( *f.initial, *f.state ), "Deflation did not recover after failed solves" );
    }
    report.require(
        operator_lifetime.expired() && storage_lifetime.expired(), "Deflation retained an ownership cycle"
    );
    report.require( space->allocations == space->frees, "Deflation leaked vector storage" );
}
}

int main()
{
    test_report report;
    try
    {
        test_newton<float, nmfd::solvers::cgs>( report );
        test_newton<double, nmfd::solvers::cgs>( report );
        test_newton<float, nmfd::solvers::bicgstabl>( report );
        test_newton<double, nmfd::solvers::bicgstabl>( report );
        std::printf( "Deflation NMFD Newton: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Deflation NMFD Newton failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
