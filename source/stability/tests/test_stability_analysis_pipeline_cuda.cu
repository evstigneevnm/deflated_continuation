#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <scfd/backend/cuda.h>

#include <common/cuda_init_scfd.h>
#include <common/scfd_vector_operations.h>
#include <stability/analysis/initial_vector_policy.h>
#include <stability/analysis/spectrum_classifier.h>
#include <stability/analysis/stability_evaluator.h>
#include <stability/analysis/stability_transition_refiner.h>

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

using vector_space_type =
    scfd_vector_operations<scfd::backend::cuda, double>;
using vector_type = typename vector_space_type::vector_type;

class parameterized_problem
{
public:
    explicit parameterized_problem(vector_space_type* vector_space)
        : vector_space_(vector_space)
    {
    }

    void set_linearization_point(
        const vector_type&,
        double parameter)
    {
        parameter_ = parameter;
        ++linearization_calls_;
    }

    void randomize_vector(vector_type& vector)
    {
        vector_space_->assign_scalar(1.0, vector);
        ++randomize_calls_;
    }

    double parameter() const
    {
        return parameter_;
    }

    std::size_t linearization_calls() const
    {
        return linearization_calls_;
    }

    std::size_t randomize_calls() const
    {
        return randomize_calls_;
    }

private:
    vector_space_type* vector_space_;
    double parameter_ = 0.0;
    std::size_t linearization_calls_ = 0;
    std::size_t randomize_calls_ = 0;
};

class parameter_spectrum_adapter
{
public:
    using real_type = double;
    using result_type =
        stability::eigensolvers::eigensolver_result<double>;

    explicit parameter_spectrum_adapter(
        parameterized_problem* problem)
        : problem_(problem)
    {
    }

    result_type execute(const vector_type&)
    {
        result_type result;
        if(fail_)
        {
            result.status =
                stability::eigensolvers::eigensolver_status::
                    inner_solver_failure;
            result.diagnostic = "expected CUDA adapter failure";
            return result;
        }

        result.status =
            stability::eigensolvers::eigensolver_status::success;
        result.eigenpairs.push_back(
            make_estimate({problem_->parameter(), 0.0}));
        result.eigenpairs.push_back(
            make_estimate({-2.0, 0.0}));
        return result;
    }

    void fail(bool value)
    {
        fail_ = value;
    }

private:
    parameterized_problem* problem_;
    bool fail_ = false;

    static stability::eigensolvers::eigenpair_estimate<double>
    make_estimate(std::complex<double> value)
    {
        stability::eigensolvers::eigenpair_estimate<double> result;
        result.value = value;
        result.residual = 1.0e-12;
        result.relative_residual = 1.0e-12;
        result.converged = true;
        return result;
    }
};

class coalescence_hopf_spectrum_adapter
{
public:
    using real_type = double;
    using result_type =
        stability::eigensolvers::eigensolver_result<double>;

    explicit coalescence_hopf_spectrum_adapter(
        parameterized_problem* problem)
        : problem_(problem)
    {
    }

    result_type execute(const vector_type&)
    {
        result_type result;
        result.status =
            stability::eigensolvers::eigensolver_status::success;

        const double parameter = problem_->parameter();
        if(parameter < -0.25)
        {
            result.eigenpairs.push_back(
                make_estimate({3.0, 0.0}));
            result.eigenpairs.push_back(
                make_estimate({2.0, 0.0}));
        }
        else
        {
            result.eigenpairs.push_back(
                make_estimate({2.5, 1.0}));
            result.eigenpairs.push_back(
                make_estimate({2.5, -1.0}));
        }

        const double crossing_real = 0.25 - parameter;
        result.eigenpairs.push_back(
            make_estimate({crossing_real, 2.0}));
        result.eigenpairs.push_back(
            make_estimate({crossing_real, -2.0}));
        result.eigenpairs.push_back(
            make_estimate({-5.0, 0.0}));
        return result;
    }

private:
    parameterized_problem* problem_;

    static stability::eigensolvers::eigenpair_estimate<double>
    make_estimate(std::complex<double> value)
    {
        stability::eigensolvers::eigenpair_estimate<double> result;
        result.value = value;
        result.residual = 1.0e-12;
        result.relative_residual = 1.0e-12;
        result.converged = true;
        return result;
    }
};

class exact_branch_newton
{
public:
    explicit exact_branch_newton(vector_space_type* vector_space)
        : vector_space_(vector_space)
    {
    }

    bool solve(
        parameterized_problem*,
        vector_type& state,
        double parameter)
    {
        if(fail_)
            return false;
        vector_space_->assign_scalar(parameter, state);
        return true;
    }

    void fail(bool value)
    {
        fail_ = value;
    }

private:
    vector_space_type* vector_space_;
    bool fail_ = false;
};

int run()
{
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            parameterized_problem>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            parameterized_problem,
            parameter_spectrum_adapter,
            classifier_type,
            initial_policy_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            parameterized_problem,
            exact_branch_newton,
            evaluator_type>;

    vector_space_type vector_space(32);
    parameterized_problem problem(&vector_space);
    parameter_spectrum_adapter adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));

    vector_type state;
    vector_type lower;
    vector_type upper;
    vector_type refined;
    vector_space.init_vector(state);
    vector_space.init_vector(lower);
    vector_space.init_vector(upper);
    vector_space.init_vector(refined);
    vector_space.start_use_vector(state);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(upper);
    vector_space.start_use_vector(refined);
    vector_space.assign_scalar(0.0, state);
    vector_space.assign_scalar(-1.0, lower);
    vector_space.assign_scalar(1.0, upper);

    const auto positive = evaluator.analyze(state, 0.5);
    require(
        positive.classification_complete(),
        "CUDA evaluator classifies spectrum");
    require(
        positive.unstable.real == 1,
        "CUDA evaluator detects unstable eigenvalue");
    require(
        problem.linearization_calls() == 1,
        "CUDA evaluator sets linearization point");
    require(
        problem.randomize_calls() == 1,
        "CUDA evaluator initializes device vector");

    adapter.fail(true);
    const auto failed = evaluator.analyze(state, 0.5);
    require(
        failed.classification_status ==
            stability::analysis::spectrum_classification_status::
                eigensolver_failure,
        "CUDA evaluator propagates solver failure");
    adapter.fail(false);

    const auto lower_stability = evaluator.analyze(lower, -1.0);
    const auto upper_stability = evaluator.analyze(upper, 1.0);
    exact_branch_newton newton(&vector_space);
    refiner_type refiner(
        &vector_space,
        &problem,
        &newton,
        &evaluator);
    typename refiner_type::options_type options;
    options.maximum_iterations = 24;
    options.parameter_tolerance = 1.0e-6;
    const auto transition = refiner.refine(
        lower,
        -1.0,
        lower_stability,
        upper,
        1.0,
        upper_stability,
        refined,
        options);
    require(
        transition.succeeded(),
        "CUDA transition refinement succeeds");
    require(
        std::abs(transition.parameter) <= 1.0e-6,
        "CUDA transition parameter");

    std::vector<double> refined_host(32);
    vector_space.get(
        refined,
        refined_host.data(),
        refined_host.size());
    require(
        std::abs(refined_host.front() - transition.parameter) <=
            1.0e-12,
        "CUDA refined state remains on branch");

    using coalescence_evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            parameterized_problem,
            coalescence_hopf_spectrum_adapter,
            classifier_type,
            initial_policy_type>;
    using coalescence_refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            parameterized_problem,
            exact_branch_newton,
            coalescence_evaluator_type>;
    coalescence_hopf_spectrum_adapter coalescence_adapter(
        &problem);
    coalescence_evaluator_type coalescence_evaluator(
        &vector_space,
        &problem,
        &coalescence_adapter,
        classifier_type{},
        initial_policy_type(&problem));
    coalescence_refiner_type coalescence_refiner(
        &vector_space,
        &problem,
        &newton,
        &coalescence_evaluator);

    const auto coalescence_lower =
        coalescence_evaluator.analyze(lower, -1.0);
    const auto coalescence_upper =
        coalescence_evaluator.analyze(upper, 1.0);
    typename coalescence_refiner_type::options_type
        coalescence_options;
    coalescence_options.maximum_iterations = 32;
    coalescence_options.parameter_tolerance = 1.0e-8;
    const auto coalescence_transition =
        coalescence_refiner.refine(
            lower,
            -1.0,
            coalescence_lower,
            upper,
            1.0,
            coalescence_upper,
            refined,
            coalescence_options);
    require(
        coalescence_transition.succeeded(),
        "CUDA coalescence followed by Hopf refinement succeeds");
    require(
        std::abs(coalescence_transition.parameter - 0.25) <=
            2.0e-7,
        "CUDA Hopf transition parameter");
    require(
        coalescence_transition.before_stability.unstable.as_pair() ==
            std::make_pair(0, 2),
        "CUDA refined local pre-Hopf signature");
    require(
        coalescence_transition.after_stability.unstable.as_pair() ==
            std::make_pair(0, 1),
        "CUDA refined local post-Hopf signature");

    newton.fail(true);
    const auto nonlinear_failure = refiner.refine(
        lower,
        -1.0,
        lower_stability,
        upper,
        1.0,
        upper_stability,
        refined,
        options);
    require(
        nonlinear_failure.status ==
            stability::analysis::stability_transition_status::
                nonlinear_solver_failure,
        "CUDA transition propagates Newton failure");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.stop_use_vector(state);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
    vector_space.free_vector(state);

    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string device_selector =
        argc > 1 ? argv[1] : "auto";
    const int device =
        common::init_cuda_from_scfd_selector(device_selector);
    std::cout << "CUDA device: " << device << '\n';
    return run();
}
