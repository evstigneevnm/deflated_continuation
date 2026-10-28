#include "continuation_newton_fixture.h"
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <continuation/augmented_linear_solver.h>
#include <continuation/augmented_tangent.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <main/nmfd_newton_types.h>
#include <scfd/utils/log_std.h>

namespace
{

void test_preconditioner_wrapper( continuation::tests::test_report &report )
{
    struct preconditioner_type
    {
        using vector_type          = scfd::static_vec::vec<double, 1>;
        mutable unsigned int calls = 0;
        virtual bool         apply( vector_type &x ) const
        {
            ++calls;
            x[0] *= 2.;
            return true;
        }
    };
    numerical_algos::bypassable_preconditioner<preconditioner_type> preconditioner;
    preconditioner_type::vector_type                                x{ 3. };
    report.require(
        preconditioner.apply( x ) && x[0] == 6. && preconditioner.calls == 1,
        "Wrapper calls a virtual base preconditioner without recursive dispatch"
    );
    preconditioner.enabled = true;
    report.require(
        preconditioner.apply( x ) && x[0] == 6. && preconditioner.calls == 1,
        "Project bypass applies identity without touching the underlying inverse"
    );
}

// Exercise both GMRES preconditioning sides with project-owned bypass control.
void test_gmres_retry( continuation::tests::test_report &report )
{
    using fixture_type  = continuation::tests::circle_fixture<double>;
    using space_type    = fixture_type::pair_space_type;
    using jacobian_type = fixture_type::operator_type::jacobi_operator_type;
    struct inverse_type
    {
        mutable unsigned int calls = 0;
        bool                 fail  = false;
        void                 set_operator( std::shared_ptr<const fixture_type::state_operator_type> )
        {
        }
        bool apply( fixture_type::space_type::vector_type & ) const
        {
            ++calls;
            return !fail;
        }
    };
    using assembly_type = main_classes::detail::augmented_solver_types<
        space_type, jacobian_type, inverse_type, scfd::utils::log_std, main_classes::detail::augmented_gmres>;
    for ( char side : { 'L', 'R' } )
    {
        fixture_type f;
        auto         inverse        = std::make_shared<inverse_type>();
        auto         preconditioner = std::make_shared<assembly_type::preconditioner_type>(
            f.space, inverse, std::make_shared<assembly_type::schur_inverse_type>( f.space->second() )
        );
        assembly_type::solver_type::params parameters;
        parameters.preconditioner_side = side;
        auto native = std::make_shared<assembly_type::solver_type>( f.space, nullptr, parameters, preconditioner );
        auto linear = std::make_shared<assembly_type::linear_type>( f.space, native, preconditioner );
        numerical_algos::recovering_linear_solver<space_type, assembly_type::linear_type> recovery( f.space, linear );
        linear->monitor().set_tolerances( 0., 1e-12 );
        f.set( *f.state, 1., 0. );
        f.set( *f.tangent, 0., 1. );
        f.op->prepare_tangent( ( *f.state ).first, 0., ( *f.tangent ).first, 1. );
        recovery.set_operator( f.op->get_jacobi_operator() );
        f.set( *f.rhs, .4, -.3 );
        inverse->fail = true;
        report.require( recovery.solve( *f.rhs, *f.action ), "Project preconditioner recovers on either GMRES side" );
        report.require(
            !linear->preconditioner_bypass_enabled() && !preconditioner->enabled,
            "Recovery restores the shared project control"
        );
        linear->set_preconditioner_bypass( true );
        inverse->calls = 0;
        report.require(
            linear->solve( *f.rhs, *f.action ) && inverse->calls == 0,
            "Bypass skips every inverse action without changing canonical GMRES"
        );
        f.op->get_jacobi_operator()->apply( *f.action, *f.plus );
        report.require( f.difference( *f.plus, *f.rhs ) < 1e-12, "Bypassed solve retains original-system accuracy" );
        report.require( !linear->supports_unpreconditioned_retry(), "Do not retry an already bypassed solve" );
        linear->set_preconditioner_bypass( false );
    }
}

// u^2 + lambda^2 = 1. Its augmented Jacobian remains nonsingular at a fold
// although J = 0 and the lower-right scalar block is exactly zero.
// Identity H is the circle's existing fixed linear state preconditioner.
template <class T, template <class, class, class, class, class> class Krylov>
void test_augmented( continuation::tests::test_report &report )
{
    using fixture_type  = continuation::tests::circle_fixture<T>;
    using space_type    = typename fixture_type::pair_space_type;
    using operator_type = typename fixture_type::operator_type;
    using jacobian_type = typename operator_type::jacobi_operator_type;
    struct inverse_type
    {
        bool fail_setup = false, fail_apply = false;
        bool set_operator( std::shared_ptr<const typename fixture_type::state_operator_type> )
        {
            return !fail_setup;
        }
        bool apply( typename fixture_type::space_type::vector_type & ) const
        {
            return !fail_apply;
        }
    };
    using assembly_type = main_classes::detail::augmented_solver_types<
        space_type, jacobian_type, inverse_type, scfd::utils::log_std, Krylov>;
    using linear_type    = typename assembly_type::linear_type;
    using recovery_type  = numerical_algos::recovering_linear_solver<space_type, linear_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, operator_type, recovery_type>;
    using projector_type = typename operator_type::trial_projector;
    using quality_type   = nmfd::operations::zero_functional<space_type>;
    using strategy_type  = nmfd::solvers::default_convergence_strategy<
         space_type, scfd::utils::log_std, operator_type, projector_type, quality_type,
         continuation::corrector_residual_norm>;
    using newton_type = nmfd::solvers::nonlinear_solver<
        space_type, scfd::utils::log_std, operator_type, iteration_type, projector_type, quality_type, strategy_type>;
    std::weak_ptr<operator_type>                       operator_lifetime;
    std::weak_ptr<linear_type>                         solver_lifetime;
    std::shared_ptr<typename fixture_type::space_type> operations;
    {
        fixture_type f;
        operations        = f.operations;
        operator_lifetime = f.op;
        const T tolerance = f.tolerance;
        auto    inverse   = std::make_shared<inverse_type>();
        auto    linear    = assembly_type::make( f.space, inverse, nullptr );
        solver_lifetime   = linear;
        linear->monitor().set_tolerances( T( 0 ), tolerance / T( 10 ) );
        auto                           recovery  = std::make_shared<recovery_type>( f.space, linear );
        auto                           iteration = std::make_shared<iteration_type>( f.space, recovery );
        typename strategy_type::params parameters;
        parameters.abs_tol   = tolerance;
        parameters.rel_tol   = T( 0 );
        parameters.verbose   = false;
        auto        strategy = std::make_shared<strategy_type>( f.space, nullptr, parameters );
        newton_type newton( f.space, nullptr, iteration, strategy );
        newton.set_nonlinear_operator( f.op );
        newton.set_project_operator( std::make_shared<projector_type>( f.op ) );
        continuation::augmented_tangent<operator_type, linear_type> tangent( f.op, linear );

        // [0 2; 1 0] at the exact fold, and [1.2 1.6; 1 0] away from it.
        for ( T u : { T( 0 ), T( .6 ) } )
        {
            f.set( *f.state, u, std::sqrt( T( 1 ) - u * u ) );
            f.set( *f.tangent, T( 1 ), T( 0 ) );
            f.op->prepare_tangent( ( *f.state ).first, ( *f.state ).second[0], ( *f.tangent ).first, T( 0 ) );
            auto jacobian = f.op->get_jacobi_operator();
            linear->set_operator( jacobian );
            f.set( *f.direction, T( .2 ), T( -.3 ) );
            jacobian->apply( *f.direction, *f.rhs );
            report.require( recovery->solve( *f.rhs, *f.action ), "Zero-border augmented solve" );
            report.require(
                f.difference( *f.action, *f.direction ) < tolerance, "Augmented solution agrees with assembled system"
            );
            jacobian->apply( *f.action, *f.plus );
            report.require( f.difference( *f.plus, *f.rhs ) < tolerance, "True augmented residual" );
            inverse->fail_setup = true;
            linear->set_operator( jacobian );
            f.space->assign( *f.direction, *f.action );
            report.require( !linear->solve( *f.rhs, *f.action ), "Failed preconditioner setup" );
            report.require( f.difference( *f.action, *f.direction ) == T( 0 ), "Failed solve rollback" );
            report.require(
                recovery->solve( *f.rhs, *f.action ) && !linear->preconditioner_bypass_enabled(),
                "Explicit unpreconditioned recovery restores mode"
            );
            inverse->fail_setup = false;
        }

        // An inverse action failure is recoverable without changing the original equation.
        f.set( *f.state, T( 1 ), T( 0 ) );
        f.set( *f.tangent, T( 0 ), T( 1 ) );
        f.op->prepare_tangent( ( *f.state ).first, T( 0 ), ( *f.tangent ).first, T( 1 ) );
        linear->set_operator( f.op->get_jacobi_operator() );
        inverse->fail_apply = true;
        f.set( *f.rhs, T( .4 ), T( -.3 ) );
        report.require( recovery->solve( *f.rhs, *f.action ), "Recover from inverse action failure" );
        inverse->fail_apply = false;

        // A genuinely singular augmented system must fail without publishing a partial result.
        f.set( *f.tangent, T( 0 ), T( 0 ) );
        f.op->prepare_tangent( ( *f.state ).first, T( 0 ), ( *f.tangent ).first, T( 0 ) );
        linear->set_operator( f.op->get_jacobi_operator() );
        f.space->assign( *f.direction, *f.action );
        linear->set_preconditioner_bypass( true );
        report.require( !linear->solve( *f.rhs, *f.action ), "Singular augmented system is not converged" );
        report.require( f.difference( *f.action, *f.direction ) == T( 0 ), "Singular solve preserves correction" );
        linear->set_preconditioner_bypass( false );

        f.set( *f.state, T( 0 ), T( 1 ) );
        report.require(
            tangent.initial( ( *f.state ).first, T( 1 ), T( 1 ), ( *f.action ).first, ( *f.action ).second[0] ),
            "Initial tangent at exact fold"
        );
        report.require(
            std::abs( f.value( *f.action ) - T( 1 ) ) < tolerance && std::abs( ( *f.action ).second[0] ) < tolerance,
            "Fold tangent orientation and equation"
        );
        report.require(
            linear->monitor().rel_tol() == T( 0 ) && linear->monitor().abs_tol() == tolerance / T( 10 ),
            "Tangent preparation restores correction tolerances"
        );
        f.set( *f.tangent, T( 1 ), T( 0 ) );
        for ( T beta : { T( 2 ), T( -2 ) } )
        {
            report.require(
                tangent.solve(
                    ( *f.state ).first, T( 1 ), ( *f.tangent ).first, T( 0 ), beta, ( *f.action ).first,
                    ( *f.action ).second[0], tolerance / T( 10 ), 1000
                ),
                "Unnormalized zero-border tangent solve"
            );
            report.require(
                std::abs( f.value( *f.action ) - beta ) < tolerance && std::abs( ( *f.action ).second[0] ) < tolerance,
                "Tangent solve preserves signed normalization RHS"
            );
        }
        f.space->assign( *f.action, *f.direction );
        report.require(
            !tangent.solve(
                ( *f.state ).first, T( 1 ), ( *f.tangent ).first, T( 0 ), T( 1 ), ( *f.action ).first,
                ( *f.action ).second[0], tolerance / T( 10 ), 0
            ),
            "Exhausted raw tangent solve fails"
        );
        report.require(
            f.difference( *f.action, *f.direction ) == T( 0 ), "Raw tangent failure preserves both outputs"
        );
        report.require(
            !linear->preconditioner_bypass_enabled() && linear->monitor().rel_tol() == T( 0 ) &&
                linear->monitor().abs_tol() == tolerance / T( 10 ),
            "Raw tangent failure restores recovery and monitor settings"
        );

        f.configure( T( .15 ), T( .08 ) );
        const auto allocations = operations->allocations;
        bool       first_fold = false, second_fold = false;
        for ( unsigned int step = 0; step < 90; ++step )
        {
            const T u = f.value( *f.reference ), parameter = ( *f.reference ).second[0];
            f.set( *f.tangent, -parameter, u );
            f.op->set_tangent_space( ( *f.reference ).first, parameter, ( *f.tangent ).first, u, T( .08 ) );
            f.space->assign_lin_comb( T( 1 ), *f.reference, T( .08 ), *f.tangent, *f.state );
            report.require( newton.solve( *f.state ), "Augmented circle corrector" );
            f.op->apply( *f.state, *f.rhs );
            report.require( f.space->norm_l2( *f.rhs ) <= tolerance, "Physical and arclength accuracy" );
            report.require(
                tangent.apply(
                    ( *f.state ).first, ( *f.state ).second[0], ( *f.tangent ).first, u, ( *f.action ).first,
                    ( *f.action ).second[0], T( 1e-5 ), 1000
                ),
                "Augmented tangent update"
            );
            f.state_operator->apply( ( *f.action ).first, ( *f.plus ).first );
            report.require(
                std::abs( f.value( *f.plus ) + T( 2 ) * ( *f.state ).second[0] * ( *f.action ).second[0] ) <
                    T( 10 ) * tolerance,
                "Tangent equation residual"
            );
            report.require(
                f.operations->scalar_prod( ( *f.action ).first, ( *f.tangent ).first ) + u * ( *f.action ).second[0] >
                    T( 0 ),
                "Tangent orientation"
            );
            const T next = f.value( *f.state );
            first_fold   = first_fold || ( u > T( 0 ) && next < T( 0 ) );
            second_fold  = second_fold || ( u < T( 0 ) && next > T( 0 ) );
            f.space->assign( *f.state, *f.reference );
        }
        report.require( first_fold && second_fold, "Circle traverses both folds" );
        report.require( operations->allocations == allocations, "No state-vector allocation in solves" );

        parameters.max_iters_num     = 1;
        auto        limited_strategy = std::make_shared<strategy_type>( f.space, nullptr, parameters );
        newton_type limited_newton( f.space, nullptr, iteration, limited_strategy );
        limited_newton.set_nonlinear_operator( f.op );
        limited_newton.set_project_operator( std::make_shared<projector_type>( f.op ) );
        f.configure( T( .3 ), T( .08 ) );
        f.set( *f.reference, T( 10 ), T( 10 ) );
        report.require( !limited_newton.solve( *f.reference, *f.state ), "Exhausted augmented Newton solve fails" );
        report.require(
            f.difference( *f.state, *f.reference ) == T( 0 ), "Failed augmented Newton solve restores both components"
        );
    }
    report.require( operator_lifetime.expired() && solver_lifetime.expired(), "Augmented ownership has no cycles" );
    report.require( operations->allocations == operations->frees, "Augmented workspace is released" );
}

}

int main()
{
    try
    {
        continuation::tests::test_report report;
        test_preconditioner_wrapper( report );
        test_gmres_retry( report );
        test_augmented<float, main_classes::detail::augmented_gmres>( report );
        test_augmented<double, main_classes::detail::augmented_gmres>( report );
        test_augmented<float, nmfd::solvers::bicgstabl>( report );
        test_augmented<double, nmfd::solvers::bicgstabl>( report );
        std::printf( "Augmented continuation: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "%s\n", error.what() );
        return 1;
    }
}
