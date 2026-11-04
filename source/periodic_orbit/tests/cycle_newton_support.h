#ifndef PERIODIC_ORBIT_CYCLE_NEWTON_SUPPORT_H
#define PERIODIC_ORBIT_CYCLE_NEWTON_SUPPORT_H

#include "prepared_cycle_support.h"
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <scfd/utils/log_std.h>

namespace cycle_tests
{
using cycle_log         = scfd::utils::log_std;
using cycle_monitor     = nmfd::solvers::monitor_krylov<blocks, cycle_log>;
using cycle_linear      = nmfd::solvers::gmres<blocks, cycle_monitor, cycle_log, orbit_problem::jacobi_operator_type>;
using cycle_iteration   = nmfd::solvers::newton_iteration<blocks, orbit_problem, cycle_linear>;
using cycle_convergence = nmfd::solvers::default_convergence_strategy<blocks, cycle_log, orbit_problem>;
using cycle_solver      = nmfd::solvers::nonlinear_solver<blocks, cycle_log, orbit_problem, cycle_iteration>;

inline std::shared_ptr<cycle_solver> make_newton(
    const std::shared_ptr<blocks> &space, const std::shared_ptr<orbit_problem> &problem, double tolerance = 1e-10,
    unsigned int iterations = 12
)
{
    cycle_linear::params lp;
    lp.basis_size                    = 16;
    lp.batch_size                    = 1;
    lp.monitor.abs_tol               = 1e-13;
    lp.monitor.rel_tol               = 1e-10;
    auto                      linear = std::make_shared<cycle_linear>( space, nullptr, lp );
    cycle_convergence::params cp;
    cp.abs_tol             = tolerance;
    cp.rel_tol             = 0;
    cp.max_iters_num       = iterations;
    cp.verbose             = false;
    cp.store_norms_history = true;
    auto result            = std::make_shared<cycle_solver>(
        space, nullptr, std::make_shared<cycle_iteration>( space, linear ),
        std::make_shared<cycle_convergence>( space, nullptr, cp )
    );
    result->set_nonlinear_operator( problem );
    return result;
}
}
#endif
