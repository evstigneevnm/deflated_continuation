#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <continuation/advance_solution.h>

namespace
{

struct vector_ops
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    explicit vector_ops(const std::size_t size_): size(size_)
    {
    }

    void init_vector(vector_type& value) const { value.assign(size, 0.0); }
    void start_use_vector(vector_type&) const {}
    void stop_use_vector(vector_type&) const {}
    void free_vector(vector_type& value) const { value.clear(); }
    void assign(const vector_type& source, vector_type& destination) const { destination = source; }

    void assign_mul(
        const double left_value,
        const vector_type& left,
        const double right_value,
        const vector_type& right,
        vector_type& destination) const
    {
        destination.resize(size);
        for(std::size_t i = 0; i < size; ++i)
        {
            destination[i] = left_value*left[i] + right_value*right[i];
        }
    }

    void add_mul(const double value, const vector_type& source, vector_type& destination) const
    {
        for(std::size_t i = 0; i < size; ++i)
        {
            destination[i] += value*source[i];
        }
    }

    void scale(const double value, vector_type& destination) const
    {
        for(auto& entry: destination)
        {
            entry *= value;
        }
    }

    double scalar_prod(const vector_type& left, const vector_type& right) const
    {
        double result = 0.0;
        for(std::size_t i = 0; i < size; ++i)
        {
            result += left[i]*right[i];
        }
        return result;
    }

    double norm_l2(const vector_type& value) const { return std::sqrt(scalar_prod(value, value)); }
    double norm(const vector_type& value) const { return norm_l2(value); }
    double norm_rank1(const vector_type& value, const double scalar) const
    {
        return std::sqrt(scalar_prod(value, value) + scalar*scalar);
    }

    bool check_is_valid_number(const vector_type& value) const
    {
        for(const auto entry: value)
        {
            if(!std::isfinite(entry))
            {
                return false;
            }
        }
        return true;
    }

    std::size_t size;
};

struct test_log
{
    void info(const char*) {}
    template<class... Args> void info_f(const char*, Args...) {}
    void warning(const char*) {}
    template<class... Args> void warning_f(const char*, Args...) {}
    void error(const char*) {}
};

struct mock_convergence
{
    std::vector<double>* get_norms_history_handle() { return &history; }
    std::vector<double> history;
};

struct mock_newton_extended
{
    template<class Operator>
    bool solve(Operator*, std::vector<double>&, double&)
    {
        ++solve_calls;
        return solve_calls != fail_on_call;
    }

    mock_convergence* get_convergence_strategy_handle() { return &convergence; }

    int solve_calls = 0;
    int fail_on_call = -1;
    mock_convergence convergence;
};

struct mock_newton
{
    template<class Operator>
    bool solve(Operator*, std::vector<double>&, const double&)
    {
        ++solve_calls;
        return false;
    }
    int solve_calls = 0;
};

struct mock_predictor
{
    using vector_type = std::vector<double>;

    void reset_all()
    {
        ds = initial_ds;
    }

    void reset_tangent_space(
        const vector_type& x0_,
        const double lambda0_,
        const vector_type& tangent_,
        const double lambda_tangent_)
    {
        x0 = x0_;
        lambda0 = lambda0_;
        tangent = tangent_;
        lambda_tangent = lambda_tangent_;
    }

    void apply(vector_type& predictor, double& predictor_lambda, vector_type& trial, double& trial_lambda)
    {
        predictor = x0;
        for(std::size_t i = 0; i < predictor.size(); ++i)
        {
            predictor[i] += ds*tangent[i];
        }
        predictor_lambda = lambda0 + ds*lambda_tangent;
        trial = predictor;
        trial_lambda = predictor_lambda;
    }

    continuation::step_retry_result retry_after_chart_rejection(const double factor)
    {
        ds *= factor;
        ++monotone_decreases;
        return continuation::step_retry_result::retry;
    }

    continuation::step_retry_result retry_after_failure()
    {
        ++adaptive_decreases;
        return failure_retry_result;
    }

    continuation::step_retry_result retry_at_step(const double requested_step)
    {
        ds = requested_step;
        ++targeted_retries;
        return continuation::step_retry_result::retry;
    }

    void accept_step(const bool recovered)
    {
        ++accept_calls;
        last_accept_recovered = recovered;
    }
    double get_ds() const { return ds; }
    double get_ds_max() const { return 1.0; }

    double initial_ds = 0.1;
    double ds = initial_ds;
    vector_type x0;
    vector_type tangent;
    double lambda0 = 0.0;
    double lambda_tangent = 0.0;
    int monotone_decreases = 0;
    int adaptive_decreases = 0;
    int targeted_retries = 0;
    int accept_calls = 0;
    bool last_accept_recovered = false;
    continuation::step_retry_result failure_retry_result =
        continuation::step_retry_result::retry;
};

struct mock_system_operator
{
    template<class Operator>
    void set_tangent_space(
        std::vector<double>&,
        double&,
        std::vector<double>&,
        double&,
        const double,
        const char,
        Operator*)
    {
    }

    double arclength_residual(const std::vector<double>&, const double&) const { return 0.0; }

    template<class Operator>
    bool update_tangent_space(
        Operator*,
        const std::vector<double>&,
        const double&,
        std::vector<double>& tangent,
        double& lambda_tangent)
    {
        tangent = {1.0, 0.0};
        lambda_tangent = 0.0;
        return true;
    }
};

struct plain_operator
{
};

struct reject_once_chart_operator
{
    void begin_continuation_chart(
        const std::vector<double>&,
        const double&,
        const std::vector<double>&,
        const double&)
    {
        ++begin_calls;
    }

    void restore_continuation_chart(
        const std::vector<double>&,
        const double&,
        const std::vector<double>&,
        const double&)
    {
        ++restore_calls;
    }

    void stabilize_predictor_for_continuation(
        const std::vector<double>&,
        const double&,
        const std::vector<double>&,
        const double&,
        const std::vector<double>& predictor,
        const double& predictor_lambda,
        std::vector<double>& trial,
        double& trial_lambda)
    {
        trial = predictor;
        trial_lambda = predictor_lambda;
        ++stabilize_calls;
        if(stabilize_calls == 1)
        {
            trial[1] += 20.0;
        }
    }

    int begin_calls = 0;
    int restore_calls = 0;
    int stabilize_calls = 0;
};

struct isotropy_event_operator
{
    symmetry::continuation::isotropy_transition_result<double>
    detect_continuation_isotropy_transition(
        const std::vector<double>&,
        const std::vector<double>&,
        const symmetry::continuation::isotropy_transition_policy<double>&)
    {
        ++detection_calls;
        symmetry::continuation::isotropy_transition_result<double> result;
        result.supported = true;
        result.detected = persistent || detection_calls <= detected_calls;
        result.previous_order = 1;
        result.candidate_order = 2;
        result.transition_order = 2;
        result.previous_transverse_ratio = 1.0e-4;
        result.candidate_transverse_ratio = 1.0e-12;
        return result;
    }

    void accept_continuation_step(
        std::vector<double>&,
        const double&,
        std::vector<double>&,
        double&)
    {
        ++accept_calls;
    }

    bool persistent = false;
    int detected_calls = 0;
    int detection_calls = 0;
    int accept_calls = 0;
};

int checks = 0;
int failures = 0;

void require_true(const bool value, const std::string& label)
{
    ++checks;
    if(!value)
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

void require_close(const double value, const double expected, const double tolerance, const std::string& label)
{
    ++checks;
    if(std::abs(value - expected) > tolerance)
    {
        ++failures;
        std::cerr << "FAIL " << label << " value=" << value << " expected=" << expected << std::endl;
    }
}

template<class Operator>
void run_solve(
    Operator& op,
    mock_newton_extended& newton_extended,
    mock_predictor& predictor,
    std::vector<double>& x1,
    double& lambda1)
{
    vector_ops ops(2);
    test_log log;
    mock_system_operator system;
    mock_newton newton;
    mock_convergence convergence;
    continuation::advance_solution<
        vector_ops,
        test_log,
        mock_newton_extended,
        mock_newton,
        Operator,
        mock_system_operator,
        mock_predictor,
        mock_convergence> advance(
            &ops,
            &log,
            &system,
            &newton_extended,
            &newton,
            &predictor,
            &convergence);

    continuation::predictor_chart_policy<double> policy;
    policy.maximum_retries = 2;
    policy.step_reduction_factor = 0.2;
    advance.set_predictor_chart_policy(policy);
    symmetry::continuation::isotropy_transition_policy<double> isotropy_policy;
    isotropy_policy.enabled = true;
    advance.set_isotropy_transition_policy(isotropy_policy);

    std::vector<double> x0{0.0, 0.0};
    std::vector<double> tangent{1.0, 0.0};
    std::vector<double> next_tangent(2, 0.0);
    double next_lambda_tangent = 0.0;
    advance.solve(
        &op,
        x0,
        0.0,
        tangent,
        0.0,
        x1,
        lambda1,
        next_tangent,
        next_lambda_tangent);
}

void test_rejected_chart_retries_before_newton()
{
    reject_once_chart_operator op;
    mock_newton_extended newton;
    mock_predictor predictor;
    std::vector<double> x1;
    double lambda1 = 0.0;
    run_solve(op, newton, predictor, x1, lambda1);

    require_true(op.begin_calls == 1, "chart initialized once");
    require_true(op.restore_calls == 1, "rejected chart restored once");
    require_true(op.stabilize_calls == 2, "predictor stabilized twice");
    require_true(predictor.monotone_decreases == 1, "chart retry decreases step monotonically");
    require_true(predictor.adaptive_decreases == 0, "chart retry does not use Newton retry schedule");
    require_true(newton.solve_calls == 1, "Newton sees only accepted predictor");
    require_close(predictor.ds, 0.02, 1.0e-14, "accepted retry step");
    require_close(x1[0], 0.02, 1.0e-14, "accepted retry state");
}

void test_plain_operator_keeps_single_pass_path()
{
    plain_operator op;
    mock_newton_extended newton;
    mock_predictor predictor;
    std::vector<double> x1;
    double lambda1 = 0.0;
    run_solve(op, newton, predictor, x1, lambda1);

    require_true(predictor.monotone_decreases == 0, "plain operator has no chart retry");
    require_true(newton.solve_calls == 1, "plain operator calls Newton once");
    require_close(x1[0], 0.1, 1.0e-14, "plain operator keeps original predictor step");
}

void test_isotropy_event_stays_latched_after_non_event_trial()
{
    vector_ops ops(2);
    test_log log;
    mock_system_operator system;
    mock_newton_extended newton_extended;
    mock_newton newton;
    mock_predictor predictor;
    mock_convergence convergence;
    isotropy_event_operator op;
    op.detected_calls = 1;

    using advance_t = continuation::advance_solution<
        vector_ops,
        test_log,
        mock_newton_extended,
        mock_newton,
        isotropy_event_operator,
        mock_system_operator,
        mock_predictor,
        mock_convergence>;
    advance_t advance(
        &ops,
        &log,
        &system,
        &newton_extended,
        &newton,
        &predictor,
        &convergence);

    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled = true;
    policy.maximum_refinements = 2;
    policy.refinement_step_factor = 0.5;
    advance.set_isotropy_transition_policy(policy);

    std::vector<double> x0{0.0, 0.0};
    std::vector<double> tangent{1.0, 0.0};
    std::vector<double> x1(2, 0.0);
    std::vector<double> next_tangent(2, 0.0);
    double lambda1 = 0.0;
    double next_lambda_tangent = 0.0;
    const bool solved = advance.solve(
        &op,
        x0,
        0.0,
        tangent,
        0.0,
        x1,
        lambda1,
        next_tangent,
        next_lambda_tangent);

    require_true(solved, "latched isotropy event returns a valid endpoint");
    require_true(advance.has_isotropy_transition(), "isotropy event remains latched");
    require_true(op.detection_calls == 3, "latched event exhausts bracket refinements");
    require_true(newton_extended.solve_calls == 3, "latched event corrects bracket trials");
    require_true(predictor.targeted_retries == 2, "latched event schedules bracket trials");
    require_true(op.accept_calls == 0, "latched event does not commit a chart state");
    require_true(predictor.accept_calls == 0, "latched event does not accept predictor state");
    require_close(x1[0], 0.1, 1.0e-14, "latched event preserves event-side endpoint");
}

void test_persistent_isotropy_event_terminates_without_commit()
{
    vector_ops ops(2);
    test_log log;
    mock_system_operator system;
    mock_newton_extended newton_extended;
    mock_newton newton;
    mock_predictor predictor;
    mock_convergence convergence;
    isotropy_event_operator op;
    op.persistent = true;

    using advance_t = continuation::advance_solution<
        vector_ops,
        test_log,
        mock_newton_extended,
        mock_newton,
        isotropy_event_operator,
        mock_system_operator,
        mock_predictor,
        mock_convergence>;
    advance_t advance(
        &ops,
        &log,
        &system,
        &newton_extended,
        &newton,
        &predictor,
        &convergence);

    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled = true;
    policy.maximum_refinements = 2;
    policy.refinement_step_factor = 0.5;
    advance.set_isotropy_transition_policy(policy);

    std::vector<double> x0{0.0, 0.0};
    std::vector<double> tangent{1.0, 0.0};
    std::vector<double> x1(2, 0.0);
    std::vector<double> next_tangent(2, 0.0);
    double lambda1 = 0.0;
    double next_lambda_tangent = 0.0;
    const bool solved = advance.solve(
        &op,
        x0,
        0.0,
        tangent,
        0.0,
        x1,
        lambda1,
        next_tangent,
        next_lambda_tangent);

    require_true(solved, "persistent isotropy event returns a valid endpoint");
    require_true(advance.has_isotropy_transition(), "persistent isotropy event is reported");
    require_true(op.detection_calls == 3, "persistent event exhausts refinement budget");
    require_true(newton_extended.solve_calls == 3, "persistent event corrects every refined trial");
    require_true(predictor.targeted_retries == 2, "persistent event refines twice");
    require_true(op.accept_calls == 0, "persistent event does not commit chart state");
    require_true(predictor.accept_calls == 0, "persistent event does not accept predictor state");
    require_close(x1[0], 0.025, 1.0e-14, "persistent event returns refined endpoint");
    require_true(
        advance.last_isotropy_transition().refinements == 2,
        "persistent event reports refinement count");
}

void test_bracket_failure_preserves_detected_endpoint()
{
    vector_ops ops(2);
    test_log log;
    mock_system_operator system;
    mock_newton_extended newton_extended;
    newton_extended.fail_on_call = 2;
    mock_newton newton;
    mock_predictor predictor;
    predictor.failure_retry_result = continuation::step_retry_result::retry_limit;
    mock_convergence convergence;
    isotropy_event_operator op;
    op.persistent = true;

    using advance_t = continuation::advance_solution<
        vector_ops,
        test_log,
        mock_newton_extended,
        mock_newton,
        isotropy_event_operator,
        mock_system_operator,
        mock_predictor,
        mock_convergence>;
    advance_t advance(
        &ops,
        &log,
        &system,
        &newton_extended,
        &newton,
        &predictor,
        &convergence);

    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled = true;
    policy.maximum_refinements = 3;
    policy.refinement_step_factor = 0.5;
    advance.set_isotropy_transition_policy(policy);

    std::vector<double> x0{0.0, 0.0};
    std::vector<double> tangent{1.0, 0.0};
    std::vector<double> x1(2, 0.0);
    std::vector<double> next_tangent(2, 0.0);
    double lambda1 = 0.0;
    double next_lambda_tangent = 0.0;
    const bool solved = advance.solve(
        &op,
        x0,
        0.0,
        tangent,
        0.0,
        x1,
        lambda1,
        next_tangent,
        next_lambda_tangent);

    require_true(solved, "failed bracket trial still returns detected endpoint");
    require_true(advance.has_isotropy_transition(), "failed bracket keeps event metadata");
    require_true(newton_extended.solve_calls == 2, "failed bracket stops after retry exhaustion");
    require_true(op.accept_calls == 0, "failed bracket does not commit chart state");
    require_true(predictor.accept_calls == 0, "failed bracket does not accept predictor state");
    require_close(x1[0], 0.1, 1.0e-14, "failed bracket restores event-side endpoint");
}

} // namespace

int main()
{
    test_rejected_chart_retries_before_newton();
    test_plain_operator_keeps_single_pass_path();
    test_isotropy_event_stays_latched_after_non_event_trial();
    test_persistent_isotropy_event_terminates_without_commit();
    test_bracket_failure_preserves_detected_endpoint();
    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
