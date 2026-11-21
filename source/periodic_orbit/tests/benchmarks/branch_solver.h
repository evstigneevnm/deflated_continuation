#ifndef PERIODIC_ORBIT_BENCHMARK_BRANCH_SOLVER_H
#define PERIODIC_ORBIT_BENCHMARK_BRANCH_SOLVER_H
#include "../newton_test_support.h"
#include <periodic_orbit/matching/parameterized_multiple_section_problem.h>
#include <continuation/advance_solution.h>
#include <continuation/augmented_tangent.h>
#include <continuation/corrector_operator.h>
#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <nonlinear_operators/fixed_parameter_operator.h>
#include <numerical_algos/parameterized_correction.h>

namespace orbit_benchmarks
{
template <class Setup>
struct branch_corrections
{
    using space      = typename decltype( std::declval<Setup>().coords )::element_type;
    using orbit      = typename decltype( std::declval<Setup>().op )::element_type;
    using corrector  = continuation::corrector_operator<space, orbit, orbit>;
    using pair_space = typename corrector::vector_space_type;
    using fixed_op   = nonlinear_operators::fixed_parameter_operator<space, orbit, orbit>;
    template <class Op, class S>
    using correction = periodic_orbit::orbit_correction<numerical_algos::parameterized_correction<
        space, orbit, Op, typename section_tests::newton_types<S, Op>::solver>>;
    using alg        = section_tests::newton_types<pair_space, corrector>;
    using tangent    = periodic_orbit::orbit_tangent<continuation::augmented_tangent<corrector, typename alg::linear>>;
    using continuation_correction_type  = correction<corrector, pair_space>;
    using continuation_convergence_type = typename alg::convergence;
    using tangent_solver_type           = tangent;
    using fixed_correction_type         = correction<fixed_op, space>;

    Setup                                &s;
    double                                tolerance;
    unsigned                              max_iterations;
    typename alg::log                     log;
    std::shared_ptr<corrector>            op    = std::make_shared<corrector>( s.coords, s.op, s.op );
    std::shared_ptr<fixed_op>             fixed = std::make_shared<fixed_op>( s.op, s.op, 1. );
    std::shared_ptr<typename alg::solver> newton =
        section_tests::make_newton( op->space(), op, tolerance, max_iterations );
    std::shared_ptr<continuation_correction_type> coupled =
        std::make_shared<continuation_correction_type>( s.coords, s.op, op, op->space(), newton );
    fixed_correction_type ordinary{
        s.coords, s.op, fixed, s.coords, section_tests::make_newton( s.coords, fixed, tolerance, max_iterations )
    };
    std::shared_ptr<typename alg::linear> linear;
    std::shared_ptr<tangent>              tan;
    explicit branch_corrections( Setup &s_, double tol = 1e-10, unsigned iterations = 12 )
        : s( s_ ), tolerance( tol ), max_iterations( iterations )
    {
        typename alg::linear::params lp;
        lp.basis_size      = std::min<std::size_t>( 64, std::max<std::size_t>( 10, s.coords->get_vector_size() + 1 ) );
        lp.batch_size      = 1;
        lp.monitor.abs_tol = 1e-13;
        lp.monitor.rel_tol = 1e-11;
        linear             = std::make_shared<typename alg::linear>( op->space(), nullptr, lp );
        tan                = std::make_shared<tangent>( op, linear );
    }
};

template <class Setup>
struct branch_solver : branch_corrections<Setup>
{
    using base = branch_corrections<Setup>;
    using typename base::alg;
    using typename base::orbit;
    using typename base::space;
    using predictor = continuation::predictor_adaptive<space, typename alg::log>;
    using system_type =
        continuation::system_operator_continuation<space, orbit, orbit, typename alg::linear, typename alg::log>;
    using advance = continuation::advance_solution<
        space, typename alg::log, typename base::continuation_correction_type, typename base::fixed_correction_type,
        orbit, system_type, predictor, typename alg::convergence, typename base::tangent>;

    predictor   pred;
    system_type sys;
    advance     driver;
    explicit branch_solver( Setup &s, double ds = .05, double tol = 1e-10, unsigned iterations = 12 )
        : base( s, tol, iterations ), pred( s.coords.get(), &this->log, ds, ds ),
          sys( s.coords.get(), &this->log, s.op.get(), nullptr ),
          driver(
              s.coords.get(), &this->log, &sys, this->coupled.get(), &this->ordinary, &pred,
              this->newton->get_convergence_strategy_handle()
          )
    {
        driver.set_tangent_solver( this->tan );
        driver.set_verbose( false );
        sys.set_verbose( false );
    }
};
}
#endif
