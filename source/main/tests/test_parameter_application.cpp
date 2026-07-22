#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <main/deflation_continuation/parameter_application.h>
#include <main/parameters.hpp>

namespace
{

struct fake_monitor
{
    void init(double tolerance_, double absolute_tolerance_, unsigned int iterations_)
    {
        tolerance = tolerance_;
        absolute_tolerance = absolute_tolerance_;
        iterations = iterations_;
    }
    void set_save_convergence_history(bool value) { save_history = value; }
    void set_divide_out_norms_by_rel_base(bool value) { relative_base = value; }
    void set_verbose(bool value) { verbose = value; }
    void out_min_resid_norm() { minimum_residual_requested = true; }

    double tolerance = 0.0;
    double absolute_tolerance = -1.0;
    unsigned int iterations = 0;
    bool save_history = false;
    bool relative_base = false;
    bool verbose = false;
    bool minimum_residual_requested = false;
};

struct fake_solver
{
    fake_monitor& monitor() { return monitor_value; }
    void set_use_precond_resid(int value) { use_preconditioned_residual = value; }
    void set_resid_recalc_freq(int value) { residual_recalculation_frequency = value; }
    void set_basis_size(int value) { basis_size = value; }

    fake_monitor monitor_value;
    int use_preconditioned_residual = -1;
    int residual_recalculation_frequency = -1;
    int basis_size = -1;
};

struct fake_convergence
{
    void set_convergence_constants(
        double tolerance_,
        unsigned int iterations_,
        double weight_,
        bool store_history_,
        bool verbose_)
    {
        tolerance = tolerance_;
        iterations = iterations_;
        weight = weight_;
        store_history = store_history_;
        verbose = verbose_;
    }

    double tolerance = 0.0;
    unsigned int iterations = 0;
    double weight = 0.0;
    bool store_history = false;
    bool verbose = false;
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

bool close_value(double left, double right)
{
    return std::abs(left - right) < 1.0e-14;
}

}

int main()
{
    try
    {
        main_classes::parameters<double> parameters;
        parameters.set_default();

        auto& linear = parameters.nonlinear_operator.linear_solver;
        linear.lin_solver_tol = 2.5e-7;
        linear.lin_solver_max_it = 87;
        linear.save_convergence_history = true;
        linear.divide_out_norms_by_rel_base = false;
        linear.verbose = true;
        linear.use_precond_resid = 0;
        linear.resid_recalc_freq = 5;
        linear.basis_size = 13;

        fake_solver solver;
        auto* monitor =
            main_classes::deflation_continuation_detail::configure_linear_solver<double>(
                &solver,
                linear);
        require(monitor == &solver.monitor_value, "monitor handle");
        require(close_value(monitor->tolerance, 2.5e-7), "linear tolerance");
        require(monitor->iterations == 87, "linear iterations");
        require(monitor->save_history, "linear history");
        require(!monitor->relative_base, "linear relative-base flag");
        require(monitor->verbose, "linear verbosity");
        require(solver.use_preconditioned_residual == 0, "preconditioned residual");
        require(solver.residual_recalculation_frequency == 5, "residual frequency");
        require(solver.basis_size == 13, "basis size");

        auto& newton_parameters = parameters.nonlinear_operator.newton;
        newton_parameters.tolerance = 3.0e-9;
        newton_parameters.newton_max_it = 41;
        newton_parameters.newton_wight = 0.75;
        newton_parameters.store_norms_history = true;
        newton_parameters.verbose = false;
        fake_convergence convergence;
        main_classes::deflation_continuation_detail::configure_newton_convergence(
            &convergence,
            newton_parameters);
        require(close_value(convergence.tolerance, 3.0e-9), "Newton tolerance");
        require(convergence.iterations == 41, "Newton iterations");
        require(close_value(convergence.weight, 0.75), "Newton weight");
        require(convergence.store_history, "Newton history");
        require(!convergence.verbose, "Newton verbosity");

        const auto retry =
            main_classes::deflation_continuation_detail::make_corrector_retry_policy<double>(
                parameters.deflation_continuation.corrector_retry_policy);
        require(
            retry.maximum_retries ==
                parameters.deflation_continuation.corrector_retry_policy.maximum_retries,
            "retry mapping");

        const auto chart =
            main_classes::deflation_continuation_detail::make_predictor_chart_policy<double>(
                parameters.deflation_continuation.predictor_chart_policy);
        require(
            chart.maximum_retries ==
                parameters.deflation_continuation.predictor_chart_policy.maximum_retries,
            "chart mapping");

        const auto branch =
            main_classes::deflation_continuation_detail::make_branch_intersection_policy<double>(
                parameters.deflation_continuation.branch_intersection_policy);
        require(
            close_value(
                branch.state_tolerance,
                parameters.deflation_continuation.branch_intersection_policy.state_tolerance),
            "branch mapping");

        const auto isotropy =
            main_classes::deflation_continuation_detail::make_isotropy_transition_policy<double>(
                parameters.deflation_continuation.isotropy_transition_policy);
        require(
            isotropy.maximum_order ==
                parameters.deflation_continuation.isotropy_transition_policy.maximum_order,
            "isotropy mapping");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
