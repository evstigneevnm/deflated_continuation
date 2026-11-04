#ifndef PERIODIC_ORBIT_NEWTON_TEST_SUPPORT_H
#define PERIODIC_ORBIT_NEWTON_TEST_SUPPORT_H

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <scfd/utils/log_std.h>

namespace section_tests
{
template <class Space, class Problem>
struct newton_types
{
    using log         = scfd::utils::log_std;
    using monitor     = nmfd::solvers::monitor_krylov<Space, log>;
    using linear      = nmfd::solvers::gmres<Space, monitor, log, typename Problem::jacobi_operator_type>;
    using iteration   = nmfd::solvers::newton_iteration<Space, Problem, linear>;
    using convergence = nmfd::solvers::default_convergence_strategy<Space, log, Problem>;
    using solver      = nmfd::solvers::nonlinear_solver<Space, log, Problem, iteration>;
};

template <class Space, class Problem>
auto make_newton(
    const std::shared_ptr<Space> &space, const std::shared_ptr<Problem> &problem, double tolerance = 1e-10,
    unsigned int iterations = 12
)
{
    using types = newton_types<Space, Problem>;
    typename types::linear::params lp;
    lp.basis_size                              = 16;
    lp.batch_size                              = 1;
    lp.monitor.abs_tol                         = 1e-13;
    lp.monitor.rel_tol                         = 1e-10;
    auto                                linear = std::make_shared<typename types::linear>( space, nullptr, lp );
    typename types::convergence::params cp;
    cp.abs_tol             = tolerance;
    cp.rel_tol             = 0;
    cp.max_iters_num       = iterations;
    cp.verbose             = false;
    cp.store_norms_history = true;
    auto result            = std::make_shared<typename types::solver>(
        space, nullptr, std::make_shared<typename types::iteration>( space, linear ),
        std::make_shared<typename types::convergence>( space, nullptr, cp )
    );
    result->set_nonlinear_operator( problem );
    return result;
}
}
#endif
