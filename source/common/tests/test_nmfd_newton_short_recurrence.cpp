#include <cmath>
#include <memory>
#include <nmfd/solvers/bicgstab.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/cgs.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <contrib/nmfd-newton/test/solvers/newton_test_fixture.h>

namespace
{

template <template <class, class, class, class, class> class LinearSolver, class T>
void test_solver( nmfd::tests::test_report &report )
{
    using fixture_type = nmfd::tests::newton_fixture<T, scfd::static_vec::vec<T, 1>>;
    struct space_type : fixture_type::space_type
    {
        using norm_type = T;
    };
    using operator_type = typename fixture_type::jacobi_type;
    using problem_type  = typename fixture_type::problem_type;
    using log_type      = scfd::utils::log_std;
    using monitor_type  = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using prec_type     = nmfd::preconditioners::dummy<space_type, operator_type>;
    using linear_type   = LinearSolver<space_type, monitor_type, log_type, operator_type, prec_type>;
    using base_type     = nmfd::solvers::iter_solver_base<space_type, monitor_type, log_type, operator_type, prec_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, base_type>;
    using newton_type    = nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, iteration_type>;
    auto                         space   = std::make_shared<space_type>();
    auto                         problem = std::make_shared<problem_type>();
    typename linear_type::params p;
    p.monitor.rel_tol                          = T( 1e-5 );
    p.monitor.abs_tol                          = T( 0 );
    p.monitor.max_iters_num                    = 20;
    auto                       linear          = std::make_shared<linear_type>( space, nullptr, p );
    std::shared_ptr<base_type> abstract_linear = linear;
    newton_type                newton( space, nullptr, std::make_shared<iteration_type>( space, abstract_linear ) );
    newton.set_nonlinear_operator( problem );
    typename space_type::vector_type x{ T( 0 ) };
    report.check<T>(
        newton.solve( x ) && std::abs( x[0] - T( 1 ) ) < T( 1e-5 ), "Newton with canonical short-recurrence solver"
    );
    problem->model = nmfd::tests::equation::cubic;
    x[0]           = T( -2 );
    report.check<T>(
        newton.solve( x ) && std::abs( x[0] - T( -1.7692923542386314 ) ) < T( 1e-5 ),
        "Nonlinear cubic solve through canonical solver base"
    );
    p.monitor.max_iters_num = 0;
    auto        failing     = std::make_shared<linear_type>( space, nullptr, p );
    newton_type limited( space, nullptr, std::make_shared<iteration_type>( space, failing ) );
    limited.set_nonlinear_operator( problem );
    x[0] = T( -2 );
    report.check<T>( !limited.solve( x ) && x[0] == T( -2 ), "Linear failure preserves the Newton iterate" );

    struct backend_type
    {
        log_type  logger;
        log_type &log()
        {
            return logger;
        }
    } backend;
    using concrete_iteration = nmfd::solvers::newton_iteration<space_type, problem_type, linear_type>;
    using hierarchy_newton   = nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, concrete_iteration>;
    typename hierarchy_newton::utils_hierarchy  utilities( backend, space );
    typename hierarchy_newton::params_hierarchy parameters;
    hierarchy_newton                            recursive( utilities, parameters );
    x[0] = T( -2 );
    report.check<T>(
        recursive.solve( problem.get(), nullptr, nullptr, x ), "Recursive Newton hierarchy with short-recurrence solver"
    );
}

template <class T>
void test( nmfd::tests::test_report &report )
{
    test_solver<nmfd::solvers::bicgstab, T>( report );
    test_solver<nmfd::solvers::bicgstabl, T>( report );
    test_solver<nmfd::solvers::cgs, T>( report );
}

}

int main()
{
    nmfd::tests::test_report report;
    test<float>( report );
    test<double>( report );
    return report.finish();
}
