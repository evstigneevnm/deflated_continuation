#include <deflation/tests/deflation_newton_fixture.h>
#include <deflation/deflation.hpp>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/sherman_morrison.h>
#include <main/nmfd_newton_types.h>
#include <scfd/utils/log_std.h>

namespace
{
using deflation::tests::test_report;

// A replaceable strategy with its own configuration interface. The algorithms
// must never call the default strategy's setters on this type.
template <class Space, class Log, class Operator, class Projector, class Quality>
class selected_deflation_strategy : public nmfd::solvers::default_convergence_strategy<
                                        Space, Log, Operator, Projector, Quality, deflation::original_residual_norm>
{
    using base_type = nmfd::solvers::default_convergence_strategy<
        Space, Log, Operator, Projector, Quality, deflation::original_residual_norm>;
    using T           = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;

public:
    using base_type::base_type;
    unsigned int assessments = 0;
    template <class Parameters>
    void configure( const Parameters &parameters )
    {
        base_type::set_convergence_constants(
            parameters.tolerance, parameters.newton_max_it, T( 1 ), 0, parameters.newton_wight,
            parameters.store_norms_history, parameters.verbose, 10, T( 1e16 )
        );
    }
    void set_convergence_constants() = delete;
    bool
    check_convergence( Operator *op, Projector *projector, Quality *quality, vector_type &state, vector_type &delta )
    {
        ++assessments;
        return base_type::check_convergence( op, projector, quality, state, delta );
    }
};

template <class T>
struct settings
{
    T            tolerance     = deflation::tests::circle_fixture<T>::tolerance;
    unsigned int newton_max_it = 40;
    T            newton_wight = 1, relax_tolerance_factor = 1, maximum_norm_increase = T( .1 ),
      newton_wight_threshold           = T( 1e-12 );
    int          relax_tolerance_steps = 0;
    unsigned int stagnation_max        = 10;
    bool         store_norms_history = true, verbose = false;
};

template <class Space, class Operator>
struct controlled_preconditioner
{
    using vector_type = typename Space::vector_type;
    bool throws       = false;
    void set_operator( std::shared_ptr<const Operator> )
    {
    }
    bool apply( vector_type & ) const
    {
        if ( throws )
            throw std::runtime_error( "Injected backend failure" );
        return true;
    }
};

template <class Fixture, class System>
struct selection : main_classes::nmfd_newton_types<
                       typename Fixture::space_type, scfd::utils::log_std, typename Fixture::problem_type,
                       typename Fixture::state_operator_type, System, typename Fixture::solutions_type,
                       main_classes::detail::newton_convergence, main_classes::detail::newton_convergence,
                       selected_deflation_strategy>
{
    using base_type = main_classes::nmfd_newton_types<
        typename Fixture::space_type, scfd::utils::log_std, typename Fixture::problem_type,
        typename Fixture::state_operator_type, System, typename Fixture::solutions_type,
        main_classes::detail::newton_convergence, main_classes::detail::newton_convergence,
        selected_deflation_strategy>;
    template <class Parameters>
    static void
    configure_deflation( typename base_type::deflation_convergence_type *strategy, const Parameters &parameters )
    {
        strategy->configure( parameters );
    }
};

// F(u,lambda)=u^2+lambda^2-1 exercises all production correction roles:
// fixed lambda, the coupled arclength equation, and deflation of the positive root.
template <class T>
void test_assembly( test_report &report )
{
    using fixture_type  = deflation::tests::circle_fixture<T>;
    using space_type    = typename fixture_type::space_type;
    using jacobian_type = typename fixture_type::state_operator_type;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::default_monitor<space_type, log_type>;
    using prec_type     = controlled_preconditioner<space_type, jacobian_type>;
    using system_type   = nmfd::solvers::sherman_morrison<
          space_type, monitor_type, log_type, jacobian_type, prec_type, nmfd::solvers::cgs>;
    using types = selection<fixture_type, system_type>;
    std::shared_ptr<space_type>                                 space;
    std::weak_ptr<typename types::fixed_correction_type>        fixed_lifetime;
    std::weak_ptr<typename types::continuation_correction_type> coupled_lifetime;
    std::weak_ptr<typename types::deflation_correction_type>    deflated_lifetime;
    {
        fixture_type f;
        space = f.space;
        settings<T> parameters;
        const T     tolerance = parameters.tolerance;
        auto        prec      = std::make_shared<prec_type>();
        auto        system    = std::make_shared<system_type>( space, f.state_operator, prec, nullptr );
        system->get_linsolver_handle()->monitor().set_tolerances( tolerance / T( 10 ), tolerance / T( 10 ) );
        system->get_linsolver_handle_original()->monitor().set_tolerances( tolerance / T( 10 ), tolerance / T( 10 ) );
        auto ordinary = std::shared_ptr<typename system_type::linear_solver_original_type>(
            system, system->get_linsolver_handle_original()
        );
        auto fixed_operator =
            std::make_shared<typename types::fixed_operator_type>( f.problem, f.state_operator, T( 0 ) );
        auto fixed = types::fixed_method::make(
            space, f.problem, fixed_operator, space, ordinary, nullptr,
            std::make_shared<typename types::projector_type>( f.problem )
        );
        fixed_lifetime = fixed;
        types::configure_fixed( fixed->get_convergence_strategy_handle(), parameters );
        T parameter = T( .5 );
        f.space->assign_scalar( T( 1.5 ), *f.state );
        report.require(
            fixed->solve( f.problem.get(), *f.state, parameter ) && parameter == T( .5 ),
            "Fixed correction changed the bound parameter"
        );
        report.require(
            std::abs( f.value( *f.state ) - std::sqrt( T( .75 ) ) ) < T( 10 ) * tolerance,
            "Fixed application binding returned an incorrect root"
        );
        f.space->assign_scalar( T( 0 ), *f.state );
        report.require(
            !fixed->solve( f.problem.get(), *f.state, parameter ) && f.space->norm_l2( *f.state ) == 0 &&
                parameter == T( .5 ),
            "Fixed failure modified the committed state or parameter"
        );
        prec->throws = true;
        f.space->assign_scalar( T( 1.5 ), *f.state );
        bool propagated = false;
        try
        {
            fixed->solve( f.problem.get(), *f.state, parameter );
        }
        catch ( const std::runtime_error & )
        {
            propagated = true;
        }
        report.require(
            propagated && f.value( *f.state ) == T( 1.5 ) && !system->preconditioner_bypass_enabled(),
            "Backend exception was hidden or modified the committed state"
        );
        prec->throws = false;

        auto corrector_operator =
            std::make_shared<typename types::corrector_operator_type>( space, f.problem, f.state_operator );
        auto corrector = types::continuation_method::make(
            space, f.problem, corrector_operator, corrector_operator->space(),
            std::make_shared<typename types::bordered_type>( system ), nullptr,
            std::make_shared<typename types::corrector_operator_type::trial_projector>( corrector_operator )
        );
        coupled_lifetime = corrector;
        types::configure_continuation( corrector->get_convergence_strategy_handle(), parameters );
        f.space->assign_scalar( T( .8 ), *f.initial );
        f.space->assign_scalar( T( -.6 ), *f.direction );
        corrector->set_tangent_space( *f.initial, T( .6 ), *f.direction, T( .8 ), T( .03 ), 'S' );
        f.space->assign_scalar( T( .782 ), *f.state );
        parameter = T( .624 );
        report.require(
            corrector->solve( f.problem.get(), *f.state, parameter ), "Coupled application binding failed"
        );
        const T u = f.value( *f.state );
        report.require(
            std::abs( u * u + parameter * parameter - T( 1 ) ) < tolerance &&
                std::abs( ( u - T( .8 ) ) * T( -.6 ) + ( parameter - T( .6 ) ) * T( .8 ) - T( .03 ) ) < tolerance,
            "Coupled binding did not commit both unknown components"
        );
        parameters.newton_max_it = 1;
        types::configure_continuation( corrector->get_convergence_strategy_handle(), parameters );
        f.space->assign_scalar( T( 2 ), *f.state );
        parameter = T( 2 );
        report.require(
            !corrector->solve( f.problem.get(), *f.state, parameter ) && f.value( *f.state ) == T( 2 ) &&
                parameter == T( 2 ),
            "Failed coupled correction changed committed unknowns"
        );
        parameters.newton_max_it = 40;

        f.solutions->clear();
        f.space->assign_scalar( T( 1 ), *f.initial );
        f.solutions->push_back( *f.initial );
        auto deflated = types::deflation_method::make(
            space, f.problem, f.op, space, std::make_shared<typename types::rank_one_type>( system ), nullptr,
            std::make_shared<typename types::deflated_operator_type::trial_projector>( f.problem )
        );
        deflated_lifetime = deflated;
        types::configure_deflation( deflated->get_convergence_strategy_handle(), parameters );
        const auto allocations = space->allocations;
        for ( unsigned int run = 0; run < 4; ++run )
        {
            f.space->assign_scalar( T( .8 ), *f.state );
            parameter = T( 0 );
            report.require(
                deflated->solve( f.problem.get(), *f.state, parameter ) &&
                    std::abs( f.value( *f.state ) + T( 1 ) ) < T( 10 ) * tolerance,
                "Selected deflation convergence strategy was not operational"
            );
        }
        report.require(
            deflated->get_convergence_strategy_handle()->assessments > 0 &&
                !deflated->get_convergence_strategy_handle()->get_norms_history_handle()->empty(),
            "Selected strategy was bypassed or lost residual history"
        );
        report.require( allocations == space->allocations, "Application correction allocated vectors during solves" );
        f.space->assign_scalar( T( 1 ), *f.state );
        report.require(
            !deflated->solve( f.problem.get(), *f.state, parameter ) && f.value( *f.state ) == T( 1 ),
            "Excluded-root failure changed the committed state"
        );

        // Instantiate the actual deflation algorithm with the non-default strategy.
        {
            using engine_type = deflation::deflation<
                space_type, int, log_type, typename fixture_type::problem_type, jacobian_type, system_type,
                typename fixture_type::solutions_type, types>;
            int         files         = 0;
            auto       *file_pointer  = &files;
            auto       *ops           = space.get();
            auto       *problem       = f.problem.get();
            auto       *jacobian      = f.state_operator.get();
            auto       *linear_system = system.get();
            auto       *solutions     = f.solutions.get();
            log_type    logger;
            auto       *log = &logger;
            engine_type engine( ops, file_pointer, log, problem, jacobian, linear_system, solutions, deflated );
            engine.set_max_retries( 0 );
            report.require(
                !engine.find_solution( T( 0 ) ), "Injected deflation did not preserve retry configuration"
            );
        }

        struct throwing_correction
        {
            typename types::deflation_convergence_type *strategy;
            bool                                        solve(
                                                       typename fixture_type::problem_type *, const typename space_type::vector_type &, T,
                                                       typename space_type::vector_type &, T &
                                                   )
            {
                throw std::runtime_error( "Injected backend failure" );
            }
            auto get_convergence_strategy_handle()
            {
                return strategy;
            }
        } throwing{ deflated->get_convergence_strategy_handle() };
        {
            log_type logger;
            deflation::deflation_operator<
                space_type, throwing_correction, typename fixture_type::problem_type,
                typename fixture_type::solutions_type, log_type, false>
                  algorithm( space.get(), &logger, &throwing, 1 );
            auto *problem = f.problem.get();
            bool  caught  = false;
            try
            {
                algorithm.find_solution( T( 0 ), problem );
            }
            catch ( const std::runtime_error & )
            {
                caught = true;
            }
            report.require( caught, "Deflation retried a backend exception as a numerical failure" );
        }

        for ( unsigned int activation = 1; activation <= 6; ++activation )
        {
            const auto live        = space->allocations - space->frees;
            space->fail_activation = space->activations + activation;
            bool caught            = false;
            try
            {
                types::fixed_method::make(
                    space, f.problem, fixed_operator, space, ordinary, nullptr,
                    std::make_shared<typename types::projector_type>( f.problem )
                );
            }
            catch ( const std::runtime_error & )
            {
                caught = true;
            }
            report.require(
                caught && space->allocations - space->frees == live, "Failed correction assembly leaked scratch"
            );
        }
        space->fail_activation = 0;
    }
    report.require(
        fixed_lifetime.expired() && coupled_lifetime.expired() && deflated_lifetime.expired(),
        "Application solver bindings formed an ownership cycle"
    );
    report.require( space->allocations == space->frees, "Application correction bindings leaked vectors" );
}
}

int main()
{
    test_report report;
    try
    {
        test_assembly<float>( report );
        test_assembly<double>( report );
        std::printf( "NMFD application corrections: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf(
            stderr, "NMFD application corrections failed after %u checks: %s\n", report.checks, error.what()
        );
        return 1;
    }
}
