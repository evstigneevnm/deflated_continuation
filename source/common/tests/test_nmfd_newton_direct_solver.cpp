#include <cmath>
#include <memory>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <contrib/nmfd-newton/test/solvers/newton_test_fixture.h>

namespace
{
template <class Space, class Operator>
struct scalar_inverse
{
    using vector_type = typename Space::vector_type;
    using T           = typename Space::scalar_type;
    std::shared_ptr<const Operator> op;
    bool                            fail = false;

    void set_operator( std::shared_ptr<const Operator> value )
    {
        op = std::move( value );
    }
    bool apply( vector_type &x ) const
    {
        if ( fail || op->coefficient == T( 0 ) )
            return false;
        x[0] /= op->coefficient;
        return true;
    }
};

template <class T>
void test( nmfd::tests::test_report &report )
{
    using fixture_type  = nmfd::tests::newton_fixture<T>;
    using space_type    = typename fixture_type::space_type;
    using problem_type  = typename fixture_type::problem_type;
    using operator_type = typename fixture_type::jacobi_type;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::default_monitor<space_type, log_type>;
    using inverse_type  = scalar_inverse<space_type, operator_type>;
    using linear_type   = nmfd::solvers::direct_solver<space_type, monitor_type, log_type, operator_type, inverse_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, linear_type>;
    using newton_type    = nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, iteration_type>;
    auto space           = std::make_shared<space_type>();
    {
        auto inverse   = std::make_shared<inverse_type>();
        auto linear    = std::make_shared<linear_type>( space, nullptr, typename linear_type::params(), inverse );
        auto iteration = std::make_shared<iteration_type>( space, linear );
        auto problem   = std::make_shared<problem_type>();
        problem->model = nmfd::tests::equation::cubic;
        newton_type solver( space, nullptr, iteration );
        solver.set_nonlinear_operator( problem );
        typename space_type::vector_type x, residual;
        x[0] = T( -2 );
        report.check<T>( solver.solve( x ), "Newton converges with canonical direct solver" );
        problem->apply( x, residual );
        report.check<T>( std::abs( residual[0] ) < T( 2e-6 ), "nonlinear residual at direct-Newton root" );
        inverse->fail = true;
        x[0]          = T( -2 );
        report.check<T>( !solver.solve( x ) && x[0] == T( -2 ), "inverse failure preserves Newton iterate" );
        inverse->fail = false;
        report.check<T>( solver.solve( x ), "repeated Newton solve recovers after inverse failure" );
    }
    report.check<T>( space->balanced(), "direct-Newton workspace lifetimes balanced" );
}
}

int main()
{
    nmfd::tests::test_report report;
    test<float>( report );
    test<double>( report );
    return report.finish();
}
