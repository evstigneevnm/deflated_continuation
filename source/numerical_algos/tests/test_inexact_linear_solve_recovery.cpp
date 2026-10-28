#include <contrib/nmfd-newton/test/solvers/newton_test_fixture.h>
#include <numerical_algos/recovering_linear_solver.h>
#include <main/nmfd_newton_types.h>

namespace
{
using nmfd::tests::test_report;

template <class T>
struct controlled_solver
{
    using fixture_type         = nmfd::tests::newton_fixture<T>;
    using linear_operator_type = typename fixture_type::jacobi_type;
    using vector_type          = typename fixture_type::vector_type;
    std::shared_ptr<const linear_operator_type> op;
    bool         bypass = false, converged = false, throws_on_retry = false, reset_observed = false;
    T            primary_fraction = T( 19 ) / T( 20 ), retry_fraction = T( 1 );
    bool         allow_retry = true;
    unsigned int calls       = 0;
    void         set_operator( std::shared_ptr<const linear_operator_type> value )
    {
        op = std::move( value );
    }
    void set_preconditioner_bypass( bool value )
    {
        bypass = value;
    }
    bool supports_unpreconditioned_retry() const
    {
        return allow_retry && !bypass;
    }
    bool solve( const vector_type &rhs, vector_type &out )
    {
        ++calls;
        if ( bypass )
        {
            reset_observed = out[0] == T( 0 );
            if ( throws_on_retry )
                throw std::runtime_error( "injected retry failure" );
        }
        out[0] = ( bypass ? retry_fraction : primary_fraction ) * rhs[0] / op->coefficient;
        return converged;
    }
};

template <class T>
void test_recovery( test_report &report )
{
    using fixture       = nmfd::tests::newton_fixture<T>;
    using vector_type   = typename fixture::vector_type;
    using linear_type   = controlled_solver<T>;
    using recovery_type = numerical_algos::recovering_linear_solver<
        typename fixture::space_type, linear_type, typename fixture::strategy_type>;
    fixture f;
    auto    linear   = std::make_shared<linear_type>();
    auto    recovery = std::make_shared<recovery_type>( f.space, linear, f.strategy );
    recovery->set_operator( f.problem.get_jacobi_operator() );
    vector_type rhs{ T( 10 ) }, delta{ T( 0 ) };
    report.check<T>(
        !recovery->solve( rhs, delta ) && linear->calls == 2 && linear->reset_observed && !linear->bypass,
        "strict recovery preserves retry, correction reset, and preconditioner restoration"
    );
    f.strategy->set_allow_inexact_increment( true );
    linear->calls = 0;
    report.check<T>(
        recovery->solve( rhs, delta ) && linear->calls == 1 &&
            std::abs( delta[0] - T( 9.5 ) ) <= T( 10 ) * std::numeric_limits<T>::epsilon() &&
            f.strategy->used_inexact_increment() && !linear->bypass,
        "acceptable failed primary correction survives without retry overwrite"
    );
    linear->primary_fraction = T( .5 );
    linear->calls            = 0;
    report.check<T>(
        recovery->solve( rhs, delta ) && linear->calls == 2 && delta[0] == T( 10 ) && !linear->bypass,
        "unacceptable primary correction can recover through assessed retry"
    );
    linear->retry_fraction = T( .5 );
    report.check<T>( !recovery->solve( rhs, delta ) && !linear->bypass, "both inadequate directions fail" );
    linear->throws_on_retry = true;
    bool propagated         = false;
    try
    {
        recovery->solve( rhs, delta );
    }
    catch ( const std::runtime_error & )
    {
        propagated = true;
    }
    report.check<T>( propagated && !linear->bypass, "retry exceptions propagate with restored preconditioner" );
    linear->throws_on_retry  = false;
    linear->primary_fraction = std::numeric_limits<T>::quiet_NaN();
    report.check<T>( !recovery->solve( rhs, delta ), "nonfinite primary correction is never accepted" );

    linear->primary_fraction = T( 19 ) / T( 20 );
    using iteration_type =
        nmfd::solvers::newton_iteration<typename fixture::space_type, typename fixture::problem_type, recovery_type>;
    using solver_type = nmfd::solvers::nonlinear_solver<
        typename fixture::space_type, scfd::utils::log_std, typename fixture::problem_type, iteration_type>;
    auto        iteration = std::make_shared<iteration_type>( f.space, recovery );
    solver_type solver( f.space, nullptr, iteration, f.strategy );
    vector_type state{ T( 2 ) };
    linear->calls          = 0;
    const auto allocations = f.space->allocations;
    report.check<T>(
        solver.solve( &f.problem, nullptr, nullptr, state ) && std::abs( state[0] - T( 1 ) ) <= f.strategy->abs_tol() &&
            linear->calls > 1 && f.strategy->used_inexact_increment(),
        "Newton and recovery share the policy without changing nonlinear convergence"
    );
    report.check<T>( f.space->allocations == allocations, "recovery fallback allocates no vector storage" );

    // Existing wrappers without an assessment policy retain their strict interface.
    numerical_algos::recovering_linear_solver<typename fixture::space_type, linear_type> strict( f.space, linear );
    strict.set_operator( f.problem.get_jacobi_operator() );
    report.check<T>( !strict.solve( rhs, delta ), "two-argument recovery remains strict" );
    linear->allow_retry = false;
    linear->calls       = 0;
    report.check<T>(
        !strict.solve( rhs, delta ) && linear->calls == 1 && !linear->bypass,
        "An inverse-only solver is not bypassed or reset after failure"
    );
}

void test_lifetime( test_report &report )
{
    using fixture = nmfd::tests::newton_fixture<double>;
    std::weak_ptr<typename fixture::strategy_type> strategy_lifetime;
    std::weak_ptr<controlled_solver<double>>       linear_lifetime;
    {
        fixture f;
        auto    linear    = std::make_shared<controlled_solver<double>>();
        strategy_lifetime = f.strategy;
        linear_lifetime   = linear;
        numerical_algos::recovering_linear_solver<
            fixture::space_type, controlled_solver<double>, fixture::strategy_type>
            recovery( f.space, linear, f.strategy );
    }
    report.check<double>( strategy_lifetime.expired() && linear_lifetime.expired(), "recovery has no ownership cycle" );
}

void test_configuration( test_report &report )
{
    using fixture = nmfd::tests::newton_fixture<double>;
    using types   = main_classes::nmfd_fixed_newton_types<
          fixture::space_type, scfd::utils::log_std, fixture::problem_type, fixture::jacobi_type,
          controlled_solver<double>>;
    struct settings
    {
        double       tolerance = 1e-6, newton_wight = 1;
        unsigned int newton_max_it       = 30;
        bool         store_norms_history = false, verbose = false, allow_inexact_increment = true;
    } parameters;
    fixture                                f;
    typename types::fixed_convergence_type strategy( f.space, nullptr );
    fixture::vector_type                   rhs{ 1 }, delta{ 1 };
    types::configure_fixed( &strategy, parameters );
    report.check<double>(
        strategy.accept_inexact_increment( *f.problem.jacobi, rhs, delta ), "fixed-role configuration enables fallback"
    );
    parameters.allow_inexact_increment = false;
    types::configure_fixed( &strategy, parameters );
    report.check<double>(
        !strategy.accept_inexact_increment( *f.problem.jacobi, rhs, delta ),
        "fixed-role reconfiguration disables fallback"
    );
}
}

int main()
{
    test_report report;
    test_recovery<float>( report );
    test_recovery<double>( report );
    test_recovery<long double>( report );
    test_lifetime( report );
    test_configuration( report );
    return report.finish();
}
