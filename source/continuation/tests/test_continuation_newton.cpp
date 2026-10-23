#include "continuation_newton_fixture.h"
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <continuation/bordered_linear_solver.h>
#include <scfd/utils/log_std.h>

namespace
{
using continuation::tests::test_report;

template <class Space, class Operator>
struct controlled_preconditioner
{
    using vector_type          = typename Space::vector_type;
    bool                 fail  = false;
    mutable unsigned int calls = 0;
    void                 set_operator( std::shared_ptr<const Operator> )
    {
    }
    bool apply( vector_type & ) const
    {
        ++calls;
        return !fail;
    }
};

// Continue u^2+lambda^2=1 through both fixed-parameter folds. Standard NMFD
// Newton solves the pair residual using the existing scalar-border elimination.
template <class T, template <class, class, class, class, class> class LinearSolver>
void test_newton( test_report &report )
{
    using fixture_type        = continuation::tests::circle_fixture<T>;
    using space_type          = typename fixture_type::space_type;
    using state_operator_type = typename fixture_type::state_operator_type;
    using operator_type       = typename fixture_type::operator_type;
    using pair_space_type     = typename fixture_type::pair_space_type;
    using log_type            = scfd::utils::log_std;
    using monitor_type        = nmfd::solvers::default_monitor<space_type, log_type>;
    using preconditioner_type = controlled_preconditioner<space_type, state_operator_type>;
    using system_type         = nmfd::solvers::sherman_morrison<
                space_type, monitor_type, log_type, state_operator_type, preconditioner_type, LinearSolver>;
    using binding_type =
        continuation::bordered_linear_solver<typename operator_type::jacobi_operator_type, system_type>;
    using recovery_type  = numerical_algos::recovering_linear_solver<pair_space_type, binding_type>;
    using iteration_type = nmfd::solvers::newton_iteration<pair_space_type, operator_type, recovery_type>;
    using projector_type = typename operator_type::trial_projector;
    using solver_type =
        nmfd::solvers::nonlinear_solver<pair_space_type, log_type, operator_type, iteration_type, projector_type>;
    using strategy_type =
        nmfd::solvers::default_convergence_strategy<pair_space_type, log_type, operator_type, projector_type>;
    std::shared_ptr<space_type>                        operations;
    std::weak_ptr<operator_type>                       operator_lifetime;
    std::weak_ptr<typename fixture_type::problem_type> problem_lifetime;
    {
        fixture_type f;
        operations        = f.operations;
        operator_lifetime = f.op;
        problem_lifetime  = f.problem;
        const T tolerance = fixture_type::tolerance;
        auto    prec      = std::make_shared<preconditioner_type>();
        auto    system    = std::make_shared<system_type>( f.operations, f.state_operator, prec, nullptr );
        system->is_small_alpha( true );
        system->get_linsolver_handle()->monitor().set_tolerances( T( 0 ), tolerance / T( 10 ) );
        auto                           binding   = std::make_shared<binding_type>( system );
        auto                           recovery  = std::make_shared<recovery_type>( f.space, binding );
        auto                           iteration = std::make_shared<iteration_type>( f.space, recovery );
        typename strategy_type::params convergence;
        convergence.abs_tol       = tolerance;
        convergence.rel_tol       = T( 0 );
        convergence.max_iters_num = 20;
        convergence.verbose       = false;
        auto        strategy      = std::make_shared<strategy_type>( f.space, nullptr, convergence );
        solver_type solver( f.space, nullptr, iteration, strategy );
        solver.set_nonlinear_operator( f.op );
        auto projector = std::make_shared<projector_type>( f.op );
        solver.set_project_operator( projector );

        f.configure( T( .2 ), T( .03 ) );
        f.set( *f.state, T( .8 ), T( .6 ) );
        f.op->set_linearization_point( *f.state );
        const auto jacobian = f.op->get_jacobi_operator();
        binding->set_operator( jacobian );
        f.set( *f.rhs, T( .3 ), T( -.2 ) );
        f.space->assign_scalar( T( 0 ), *f.action );
        report.require( recovery->solve( *f.rhs, *f.action ), "Bordered solve failed" );
        const T row = f.value( *f.tangent ), alpha = ( *f.tangent ).second[0];
        const T determinant     = T( 1.6 ) * alpha - T( 1.2 ) * row;
        const T exact_state     = ( T( .3 ) * alpha + T( .24 ) ) / determinant;
        const T exact_parameter = ( T( -.32 ) - row * T( .3 ) ) / determinant;
        report.require(
            std::abs( f.value( *f.action ) - exact_state ) < T( 10 ) * tolerance &&
                std::abs( ( *f.action ).second[0] - exact_parameter ) < T( 10 ) * tolerance,
            "Bordered solve disagrees with the assembled two-by-two reference"
        );
        jacobian->apply( *f.action, *f.plus );
        report.require(
            f.difference( *f.plus, *f.rhs ) < tolerance, "Bordered correction failed full-system residual verification"
        );

        prec->fail       = true;
        const auto calls = prec->calls;
        f.space->assign_scalar( T( 0 ), *f.action );
        report.require(
            recovery->solve( *f.rhs, *f.action ) && prec->calls > calls,
            "Bordered solve did not recover from preconditioner failure"
        );
        report.require( !system->preconditioner_bypass_enabled(), "Bordered retry did not restore preconditioning" );
        prec->fail = false;

        // The anchor tangent is fixed within each corrector, not recomputed at trials.
        const auto allocations = operations->allocations;
        f.configure( T( .15 ), T( .08 ) );
        bool positive_to_negative = false, negative_to_positive = false;
        for ( unsigned int step = 0; step < 90; ++step )
        {
            const T u = f.value( *f.reference ), parameter = ( *f.reference ).second[0];
            f.set( *f.tangent, -parameter, u );
            f.op->set_tangent_space( ( *f.reference ).first, parameter, ( *f.tangent ).first, u, T( .08 ) );
            f.space->assign_lin_comb( T( 1 ), *f.reference, T( .08 ), *f.tangent, *f.state );
            report.require( solver.solve( *f.state ), "Circle continuation corrector failed" );
            f.op->apply( *f.state, *f.rhs );
            report.require(
                f.space->norm_l2( *f.rhs ) <= tolerance,
                "Continuation accepted an inaccurate physical or arclength residual"
            );
            const T next_u       = f.value( *f.state );
            positive_to_negative = positive_to_negative || ( u > 0 && next_u < 0 );
            negative_to_positive = negative_to_positive || ( u < 0 && next_u > 0 );
            f.space->assign( *f.state, *f.reference );
        }
        report.require( positive_to_negative && negative_to_positive, "Continuation did not pass both folds" );
        report.require(
            allocations == operations->allocations, "Continuation allocated vectors inside corrector solves"
        );

        if constexpr ( std::is_same_v<T, double> )
        {
            // Exercise the existing small-pivot scaling without demanding float sub-roundoff solves.
            f.configure( std::acos( T( 1e-5 ) ), T( .02 ) );
            f.space->assign_lin_comb( T( 1 ), *f.reference, T( .02 ), *f.tangent, *f.state );
            report.require( solver.solve( *f.state ), "Scaled small-border solve regressed" );
            f.op->apply( *f.state, *f.rhs );
            report.require(
                f.space->norm_l2( *f.rhs ) <= tolerance && f.value( *f.state ) < 0,
                "Small-border correction did not cross the fold accurately"
            );
        }

        // The current elimination requires a nonzero scalar pivot, even when
        // the full matrix [0 2; 1 0] is nonsingular. Report failure, never false success.
        f.set( *f.reference, T( 0 ), T( 1 ) );
        f.set( *f.tangent, T( 1 ), T( 0 ) );
        f.op->set_tangent_space( ( *f.reference ).first, T( 1 ), ( *f.tangent ).first, T( 0 ), T( .02 ) );
        f.space->assign( *f.reference, *f.state );
        report.require(
            !solver.solve( *f.reference, *f.state ) && strategy->get_result_status() == 6,
            "Unsupported zero scalar pivot did not report linear failure"
        );
        report.require(
            f.difference( *f.state, *f.reference ) == 0, "Failed corrector did not restore both unknown components"
        );
        report.require( !system->preconditioner_bypass_enabled(), "Failed bordered solve left bypass enabled" );

        f.configure( T( .4 ), T( .03 ) );
        f.set( *f.state, T( 2 ), T( 2 ) );
        convergence.max_iters_num    = 1;
        auto        limited_strategy = std::make_shared<strategy_type>( f.space, nullptr, convergence );
        solver_type limited( f.space, nullptr, iteration, limited_strategy );
        limited.set_nonlinear_operator( f.op );
        limited.set_project_operator( projector );
        report.require(
            !limited.solve( *f.state, *f.plus ) && limited_strategy->get_result_status() == 9,
            "Continuation iteration limit did not report exhaustion"
        );
        report.require(
            f.difference( *f.plus, *f.state ) == 0, "Exhausted corrector did not roll back state and parameter"
        );
        f.space->assign_lin_comb( T( 1 ), *f.reference, T( .03 ), *f.tangent, *f.state );
        report.require( solver.solve( *f.state ), "Continuation did not recover after failed solves" );
    }
    report.require(
        operator_lifetime.expired() && problem_lifetime.expired(), "Continuation ownership cycle retained a problem"
    );
    report.require( operations->allocations == operations->frees, "Continuation leaked vector storage" );
}
}

int main()
{
    continuation::tests::test_report report;
    try
    {
        test_newton<float, nmfd::solvers::cgs>( report );
        test_newton<double, nmfd::solvers::cgs>( report );
        test_newton<float, nmfd::solvers::bicgstabl>( report );
        test_newton<double, nmfd::solvers::bicgstabl>( report );
        std::printf( "Continuation NMFD Newton: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Continuation NMFD Newton failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
