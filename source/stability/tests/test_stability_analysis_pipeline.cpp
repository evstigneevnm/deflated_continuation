#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <stability/analysis/initial_vector_policy.h>
#include <stability/analysis/spectrum_classifier.h>
#include <stability/analysis/stability_evaluator.h>
#include <stability/analysis/stability_transition_refiner.h>
#include <stability/eigensolvers/direct_scalar_eigensolver.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>
#include <stability/stability_analysis.hpp>

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
        std::cout << "FAIL " << message << std::endl;
    }
}

using eigensolver_result =
    stability::eigensolvers::eigensolver_result<double>;

stability::eigensolvers::eigenpair_estimate<double> estimate(
    std::complex<double> value,
    bool converged = true)
{
    stability::eigensolvers::eigenpair_estimate<double> result;
    result.value = value;
    result.residual = converged ? 1.0e-12 : 1.0;
    result.relative_residual = result.residual;
    result.converged = converged;
    return result;
}

eigensolver_result successful_spectrum(
    std::initializer_list<
        stability::eigensolvers::eigenpair_estimate<double>> values)
{
    eigensolver_result result;
    result.status =
        stability::eigensolvers::eigensolver_status::success;
    result.eigenpairs.assign(values.begin(), values.end());
    return result;
}

void test_spectrum_classifier()
{
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    classifier_type classifier;

    const auto mixed = classifier.classify(successful_spectrum({
        estimate({-2.0, 0.0}),
        estimate({1.0, 0.0}),
        estimate({0.5, 2.0}),
        estimate({0.5, -2.0}),
        estimate({-1.0, 3.0}),
        estimate({-1.0, -3.0}),
        estimate({1.0e-10, 0.0}),
        estimate({0.0, 4.0}),
        estimate({0.0, -4.0})
    }));
    require(mixed.classification_complete(), "mixed spectrum classified");
    require(mixed.unstable.real == 1, "mixed unstable real count");
    require(
        mixed.unstable.complex_pairs == 1,
        "mixed unstable complex-pair count");
    require(mixed.stable_real == 1, "mixed stable real count");
    require(
        mixed.stable_complex_pairs == 1,
        "mixed stable complex-pair count");
    require(mixed.neutral_real == 1, "mixed neutral real count");
    require(
        mixed.neutral_complex_pairs == 1,
        "mixed neutral complex-pair count");

    auto right_options = classifier.options();
    right_options.stable =
        stability::analysis::stable_halfplane::right;
    classifier.set_options(right_options);
    const auto right_stable = classifier.classify(successful_spectrum({
        estimate({-2.0, 0.0}),
        estimate({1.0, 0.0}),
        estimate({0.5, 2.0}),
        estimate({0.5, -2.0}),
        estimate({-1.0, 3.0}),
        estimate({-1.0, -3.0})
    }));
    require(
        right_stable.unstable.real == 1,
        "right-stable half-plane flips real classification");
    require(
        right_stable.unstable.complex_pairs == 1,
        "right-stable half-plane flips complex classification");

    classifier.set_options(classifier_type::options_type{});
    const auto unmatched = classifier.classify(successful_spectrum({
        estimate({0.5, 2.0})
    }));
    require(
        !unmatched.classification_complete(),
        "unmatched complex eigenvalue is rejected");
    require(
        unmatched.unmatched_complex_eigenvalues == 1,
        "unmatched complex eigenvalue is reported");

    const auto unconverged = classifier.classify(successful_spectrum({
        estimate({1.0, 0.0}, false)
    }));
    require(
        !unconverged.classification_complete(),
        "unconverged eigenpair is rejected");
    require(
        unconverged.unclassified_eigenvalues == 1,
        "unconverged eigenpair is reported");

    eigensolver_result failure;
    failure.status =
        stability::eigensolvers::eigensolver_status::
            inner_solver_failure;
    failure.diagnostic = "expected inner failure";
    const auto failed = classifier.classify(std::move(failure));
    require(
        failed.classification_status ==
            stability::analysis::spectrum_classification_status::
                eigensolver_failure,
        "eigensolver failure propagates to classification");

    eigensolver_result empty;
    empty.status =
        stability::eigensolvers::eigensolver_status::success;
    require(
        !classifier.classify(std::move(empty)).
            classification_complete(),
        "empty spectrum is incomplete");
}

template<class VectorSpace>
class parameterized_problem
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    explicit parameterized_problem(VectorSpace* vector_space)
        : vector_space_(vector_space)
    {
    }

    void set_linearization_point(
        const vector_type& state,
        scalar_type parameter)
    {
        if(fail_linearization_)
            throw std::runtime_error("expected linearization failure");
        vector_space_->get(
            state,
            &linearization_state_value_,
            std::size_t(1));
        parameter_ = parameter;
        ++linearization_calls_;
    }

    void randomize_vector(vector_type& vector)
    {
        vector_space_->assign_scalar(scalar_type(1), vector);
        ++randomize_calls_;
    }

    scalar_type parameter() const
    {
        return parameter_;
    }

    scalar_type linearization_state_value() const
    {
        return linearization_state_value_;
    }

    std::size_t linearization_calls() const
    {
        return linearization_calls_;
    }

    std::size_t randomize_calls() const
    {
        return randomize_calls_;
    }

    void fail_linearization(bool value)
    {
        fail_linearization_ = value;
    }

private:
    VectorSpace* vector_space_;
    scalar_type parameter_ = scalar_type{};
    scalar_type linearization_state_value_ = scalar_type{};
    std::size_t linearization_calls_ = 0;
    std::size_t randomize_calls_ = 0;
    bool fail_linearization_ = false;
};

template<class VectorSpace>
class distinct_stability_probe_problem
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    explicit distinct_stability_probe_problem(VectorSpace* vector_space)
        : vector_space_(vector_space)
    {
    }

    void randomize_vector(vector_type& vector)
    {
        vector_space_->assign_scalar(scalar_type(1), vector);
        ++deflation_probe_calls_;
    }

    void randomize_stability_vector(vector_type& vector)
    {
        vector_space_->assign_scalar(scalar_type(2), vector);
        ++stability_probe_calls_;
    }

    std::size_t deflation_probe_calls() const
    {
        return deflation_probe_calls_;
    }

    std::size_t stability_probe_calls() const
    {
        return stability_probe_calls_;
    }

private:
    VectorSpace* vector_space_;
    std::size_t deflation_probe_calls_ = 0;
    std::size_t stability_probe_calls_ = 0;
};

template<class Backend>
void test_distinct_stability_probe_policy(const std::string& label)
{
    using vector_space_type = scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = distinct_stability_probe_problem<vector_space_type>;
    using policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    policy_type policy(&problem);
    vector_type vector;
    vector_space.init_vector(vector);
    vector_space.start_use_vector(vector);

    policy(vector);
    double first = 0.0;
    vector_space.get(vector, &first, std::size_t(1));
    require(
        first == 2.0,
        label + " stability-specific probe is selected");
    require(
        problem.stability_probe_calls() == 1,
        label + " stability-specific probe is called once");
    require(
        problem.deflation_probe_calls() == 0,
        label + " deflation probe remains unused by stability");

    vector_space.stop_use_vector(vector);
    vector_space.free_vector(vector);
}

template<class VectorSpace>
class separate_linearization_provider
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    void set_linearization_point(
        const vector_type&,
        scalar_type parameter)
    {
        parameter_ = parameter;
        ++linearization_calls_;
    }

    scalar_type parameter() const
    {
        return parameter_;
    }

    std::size_t linearization_calls() const
    {
        return linearization_calls_;
    }

private:
    scalar_type parameter_ = scalar_type{};
    std::size_t linearization_calls_ = 0;
};

template<class VectorSpace, class Problem>
class parameter_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;
    using probe_generator_type =
        std::function<void(
            std::size_t,
            const vector_type&,
            vector_type&)>;

    explicit parameter_spectrum_adapter(Problem* problem)
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
            result.diagnostic = "expected spectrum failure";
            return result;
        }
        result.status =
            stability::eigensolvers::eigensolver_status::success;
        result.eigenpairs.push_back(
            estimate({problem_->parameter(), 0.0}));
        result.eigenpairs.push_back(estimate({-2.0, 0.0}));
        return result;
    }

    void fail(bool value)
    {
        fail_ = value;
    }

    void set_probe_generator(probe_generator_type probe_generator)
    {
        probe_generator_ = std::move(probe_generator);
    }

    bool probe_generator_configured() const
    {
        return static_cast<bool>(probe_generator_);
    }

    void generate_probe(
        std::size_t probe_index,
        const vector_type& initial_vector,
        vector_type& vector) const
    {
        probe_generator_(
            probe_index,
            initial_vector,
            vector);
    }

private:
    Problem* problem_;
    bool fail_ = false;
    probe_generator_type probe_generator_;
};

template<class VectorSpace, class Problem>
class transient_unmatched_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit transient_unmatched_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++execute_calls_;
        result_type result;
        result.status =
            stability::eigensolvers::eigensolver_status::success;
        result.eigenpairs.push_back(estimate({0.5, 2.0}));
        if(execute_calls_ > incomplete_attempts_)
            result.eigenpairs.push_back(estimate({0.5, -2.0}));
        return result;
    }

    void set_incomplete_attempts(std::size_t value)
    {
        incomplete_attempts_ = value;
    }

    std::size_t execute_calls() const
    {
        return execute_calls_;
    }

private:
    std::size_t incomplete_attempts_ = 1;
    std::size_t execute_calls_ = 0;
};

template<class VectorSpace, class Problem>
class classification_fallback_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit classification_fallback_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++primary_calls_;
        return successful_spectrum({
            estimate({0.5, 2.0})
        });
    }

    bool classification_fallback_available() const
    {
        return true;
    }

    result_type execute_classification_fallback(
        const vector_type&) const
    {
        ++fallback_calls_;
        return successful_spectrum({
            estimate({0.5, 2.0}),
            estimate({0.5, -2.0})
        });
    }

    std::size_t primary_calls() const
    {
        return primary_calls_;
    }

    std::size_t fallback_calls() const
    {
        return fallback_calls_;
    }

private:
    std::size_t primary_calls_ = 0;
    mutable std::size_t fallback_calls_ = 0;
};

template<class VectorSpace, class Problem>
class classification_confirmation_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit classification_confirmation_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++primary_calls_;
        return successful_spectrum({
            estimate({2.0, 0.0}),
            estimate({2.0, 0.0})
        });
    }

    bool classification_confirmation_available() const
    {
        return true;
    }

    result_type execute_classification_confirmation(
        const vector_type&) const
    {
        ++confirmation_calls_;
        return successful_spectrum({
            estimate({2.0, 0.0})
        });
    }

    std::size_t primary_calls() const
    {
        return primary_calls_;
    }

    std::size_t confirmation_calls() const
    {
        return confirmation_calls_;
    }

private:
    std::size_t primary_calls_ = 0;
    mutable std::size_t confirmation_calls_ = 0;
};

template<class VectorSpace, class Problem>
class inconsistent_confirmation_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit inconsistent_confirmation_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++calls_;
        result_type result = successful_spectrum({
            estimate({2.0, 0.0})
        });
        if(calls_ % 2 == 0)
            result.eigenpairs.push_back(estimate({3.0, 0.0}));
        return result;
    }

    std::size_t calls() const
    {
        return calls_;
    }

private:
    std::size_t calls_ = 0;
};

template<class VectorSpace, class Problem>
class recovering_confirmation_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit recovering_confirmation_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++calls_;
        result_type result = successful_spectrum({
            estimate({2.0, 0.0})
        });
        if(calls_ >= 2)
            result.eigenpairs.push_back(estimate({3.0, 0.0}));
        return result;
    }

    std::size_t calls() const
    {
        return calls_;
    }

private:
    std::size_t calls_ = 0;
};

template<class VectorSpace, class Problem>
class transactional_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit transactional_spectrum_adapter(Problem*)
    {
    }

    result_type execute(const vector_type&)
    {
        ++execute_calls_;
        if(fail_)
        {
            result_type result;
            result.status =
                stability::eigensolvers::eigensolver_status::
                    inner_solver_failure;
            return result;
        }
        return successful_spectrum({estimate({-1.0, 0.0})});
    }

    void begin_recycling_transaction() const
    {
        ++begin_calls_;
    }

    void commit_recycling_transaction() const
    {
        ++commit_calls_;
    }

    void rollback_recycling_transaction() const
    {
        ++rollback_calls_;
    }

    void fail(bool value)
    {
        fail_ = value;
    }

    std::size_t begin_calls() const { return begin_calls_; }
    std::size_t commit_calls() const { return commit_calls_; }
    std::size_t rollback_calls() const { return rollback_calls_; }
    std::size_t execute_calls() const { return execute_calls_; }

private:
    bool fail_ = false;
    std::size_t execute_calls_ = 0;
    mutable std::size_t begin_calls_ = 0;
    mutable std::size_t commit_calls_ = 0;
    mutable std::size_t rollback_calls_ = 0;
};

template<class VectorSpace, class Problem>
class coalescence_hopf_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit coalescence_hopf_spectrum_adapter(Problem* problem)
        : problem_(problem)
    {
    }

    result_type execute(const vector_type&)
    {
        result_type result;
        result.status =
            stability::eigensolvers::eigensolver_status::success;

        const real_type parameter = problem_->parameter();
        if(parameter < real_type(-0.25))
        {
            result.eigenpairs.push_back(
                estimate({3.0, 0.0}));
            result.eigenpairs.push_back(
                estimate({2.0, 0.0}));
        }
        else
        {
            result.eigenpairs.push_back(
                estimate({2.5, 1.0}));
            result.eigenpairs.push_back(
                estimate({2.5, -1.0}));
        }

        const real_type crossing_real =
            real_type(0.25) - parameter;
        result.eigenpairs.push_back(
            estimate({crossing_real, 2.0}));
        result.eigenpairs.push_back(
            estimate({crossing_real, -2.0}));
        result.eigenpairs.push_back(
            estimate({-5.0, 0.0}));
        return result;
    }

private:
    Problem* problem_;
};

template<class VectorSpace, class Problem>
class two_real_crossings_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    explicit two_real_crossings_spectrum_adapter(Problem* problem)
        : problem_(problem)
    {
    }

    result_type execute(const vector_type&)
    {
        const real_type parameter = problem_->parameter();
        return successful_spectrum({
            estimate({parameter + real_type(0.25), 0.0}),
            estimate({parameter - real_type(0.25), 0.0}),
            estimate({-5.0, 0.0})
        });
    }

private:
    Problem* problem_;
};

template<class VectorSpace, class Problem>
class exact_branch_newton
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    explicit exact_branch_newton(VectorSpace* vector_space)
        : vector_space_(vector_space)
    {
    }

    bool solve(Problem*, vector_type& state, scalar_type parameter)
    {
        if(fail_)
            return false;
        vector_space_->assign_scalar(parameter, state);
        ++solve_calls_;
        return true;
    }

    void fail(bool value)
    {
        fail_ = value;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

private:
    VectorSpace* vector_space_;
    bool fail_ = false;
    std::size_t solve_calls_ = 0;
};

template<class VectorSpace, class Problem>
class curved_branch_artifact_spectrum_adapter
{
public:
    using real_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using result_type =
        stability::eigensolvers::eigensolver_result<real_type>;

    curved_branch_artifact_spectrum_adapter(
        Problem* problem)
        : problem_(problem)
    {
    }

    result_type execute(const vector_type&)
    {
        const real_type parameter = problem_->parameter();
        result_type result;
        result.status =
            stability::eigensolvers::eigensolver_status::success;
        result.eigenpairs.push_back(
            estimate({parameter, 0.0}));
        result.eigenpairs.push_back(
            estimate({-5.0, 0.0}));

        using std::abs;
        if(
            abs(
                problem_->linearization_state_value() -
                parameter*parameter) >
            real_type(1.0e-10))
        {
            result.eigenpairs.push_back(
                estimate({2.0, 0.0}));
            result.eigenpairs.push_back(
                estimate({3.0, 0.0}));
        }
        return result;
    }

private:
    Problem* problem_;
};

template<class VectorSpace, class Problem>
class quadratic_branch_newton
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    explicit quadratic_branch_newton(VectorSpace* vector_space)
        : vector_space_(vector_space)
    {
    }

    bool solve(Problem*, vector_type& state, scalar_type parameter)
    {
        vector_space_->assign_scalar(
            parameter*parameter,
            state);
        ++solve_calls_;
        return true;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

private:
    VectorSpace* vector_space_;
    std::size_t solve_calls_ = 0;
};

template<class VectorSpace>
class sign_quotient_aligner
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    explicit sign_quotient_aligner(VectorSpace* vector_space)
        : vector_space_(vector_space)
    {
    }

    void stabilize_closest_to_reference(
        const vector_type& reference,
        const vector_type& source,
        vector_type& destination)
    {
        scalar_type reference_value = scalar_type{};
        scalar_type source_value = scalar_type{};
        vector_space_->get(
            reference,
            &reference_value,
            std::size_t(1));
        vector_space_->get(
            source,
            &source_value,
            std::size_t(1));
        const scalar_type sign =
            reference_value*source_value < scalar_type(0)
                ? scalar_type(-1)
                : scalar_type(1);
        vector_space_->assign_mul(
            sign,
            source,
            destination);
    }

private:
    VectorSpace* vector_space_;
};

template<class Backend>
void test_transition_state_alignment(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        parameter_spectrum_adapter<vector_space_type, problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            problem_type,
            newton_type,
            evaluator_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    newton_type newton(&vector_space);
    refiner_type refiner(
        &vector_space,
        &problem,
        &newton,
        &evaluator);
    sign_quotient_aligner<vector_space_type> aligner(
        &vector_space);
    refiner.set_transition_state_aligner(&aligner);

    vector_type lower;
    vector_type upper;
    vector_type refined;
    vector_space.init_vector(lower);
    vector_space.init_vector(upper);
    vector_space.init_vector(refined);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(upper);
    vector_space.start_use_vector(refined);
    vector_space.assign_scalar(2.0, lower);
    vector_space.assign_scalar(-2.0, upper);

    const auto lower_stability =
        evaluator.analyze(lower, -1.0);
    const auto upper_stability =
        evaluator.analyze(upper, 1.0);
    typename refiner_type::options_type options;
    options.maximum_iterations = 24;
    options.parameter_tolerance = 1.0e-6;
    options.correct_with_fixed_parameter_newton = false;
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
        label + " quotient-aligned transition succeeds");
    require(
        std::abs(transition.parameter) <= 1.0e-6,
        label + " quotient-aligned transition parameter");
    std::vector<double> refined_host(4);
    vector_space.get(
        refined,
        refined_host.data(),
        refined_host.size());
    require(
        std::abs(refined_host.front() - 2.0) <= 1.0e-12,
        label + " transition remains in the reference representative");
    require(
        newton.solve_calls() == 0,
        label + " alignment does not require nonlinear correction");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
}

template<class Backend>
void test_evaluator_and_refiner(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        parameter_spectrum_adapter<vector_space_type, problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            problem_type,
            newton_type,
            evaluator_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
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
        label + " evaluator classifies spectrum");
    require(
        positive.unstable.real == 1,
        label + " evaluator detects instability");
    require(
        problem.linearization_calls() == 1,
        label + " evaluator sets linearization point");
    require(
        problem.randomize_calls() == 1,
        label + " evaluator initializes Arnoldi vector");

    const auto negative = evaluator.analyze(state, -0.5);
    require(
        negative.unstable.real == 0,
        label + " evaluator detects stable side");

    adapter.fail(true);
    const auto solver_failure = evaluator.analyze(state, 0.5);
    require(
        solver_failure.classification_status ==
            stability::analysis::spectrum_classification_status::
                eigensolver_failure,
        label + " evaluator propagates eigensolver failure");
    adapter.fail(false);

    problem.fail_linearization(true);
    const auto linearization_failure =
        evaluator.analyze(state, 0.5);
    require(
        linearization_failure.eigensolver_status ==
            stability::eigensolvers::eigensolver_status::
                operator_failure,
        label + " evaluator propagates linearization failure");
    problem.fail_linearization(false);

    const auto lower_stability = evaluator.analyze(lower, -1.0);
    const auto upper_stability = evaluator.analyze(upper, 1.0);
    newton_type newton(&vector_space);
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
        label + " transition refinement succeeds");
    require(
        std::abs(transition.parameter) <= 1.0e-6,
        label + " transition parameter");
    std::vector<double> refined_host(4);
    vector_space.get(
        refined,
        refined_host.data(),
        refined_host.size());
    require(
        std::abs(refined_host.front() - transition.parameter) <=
            1.0e-12,
        label + " refined state matches branch");

    newton.fail(true);
    const auto failed_transition = refiner.refine(
        lower,
        -1.0,
        lower_stability,
        upper,
        1.0,
        upper_stability,
        refined,
        options);
    require(
        failed_transition.status ==
            stability::analysis::stability_transition_status::
                nonlinear_solver_failure,
        label + " transition propagates Newton failure");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.stop_use_vector(state);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
    vector_space.free_vector(state);
}

template<class Backend>
void test_classification_retry(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        transient_unmatched_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    evaluator.set_classification_retry_count(2);

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto recovered = evaluator.analyze(state, 0.5);
    require(
        recovered.succeeded(),
        label + " incomplete classification is retried");
    require(
        recovered.classification_attempts == 2,
        label + " successful retry count");
    require(
        recovered.unstable.complex_pairs == 1,
        label + " retry returns the conjugate pair");
    require(
        adapter.execute_calls() == 2,
        label + " eigensolver is executed twice");
    require(
        problem.linearization_calls() == 1,
        label + " retry preserves the frozen linearization");
    require(
        problem.randomize_calls() == 2,
        label + " retry uses an independent initial vector");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_classification_fallback(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        classification_fallback_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto recovered = evaluator.analyze(state, 0.5);
    require(
        recovered.succeeded(),
        label + " incomplete classification uses exact fallback");
    require(
        recovered.classification_attempts == 2,
        label + " fallback attempt count");
    require(
        recovered.unstable.complex_pairs == 1,
        label + " fallback returns the conjugate pair");
    require(
        adapter.primary_calls() == 1 &&
            adapter.fallback_calls() == 1,
        label + " fallback runs after one primary attempt");
    require(
        problem.linearization_calls() == 1,
        label + " fallback preserves the frozen linearization");
    require(
        problem.randomize_calls() == 1,
        label + " fallback reuses the prepared initial vector");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_classification_confirmation(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        classification_confirmation_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto primary = evaluator.analyze(state, 0.5);
    const auto confirmed = evaluator.analyze_confirmed(state, 0.5);
    require(
        primary.succeeded() && primary.unstable.real == 2,
        label + " primary classification exposes false multiplicity");
    require(
        confirmed.succeeded() && confirmed.unstable.real == 1,
        label + " dedicated confirmation corrects multiplicity");
    require(
        adapter.primary_calls() == 1 &&
            adapter.confirmation_calls() == 1,
        label + " confirmation bypasses the primary eigensolver");
    require(
        problem.linearization_calls() == 2,
        label + " confirmation freezes its own linearization");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_classification_confirmation_consensus(
    const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        inconsistent_confirmation_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    evaluator.set_classification_confirmation_count(2);

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto result = evaluator.analyze_confirmed(state, 0.5);
    require(
        !result.succeeded() &&
            result.classification_status ==
                stability::analysis::
                    spectrum_classification_status::incomplete,
        label + " inconsistent transition confirmation is rejected");
    require(
        result.diagnostic.find("inconsistent") != std::string::npos,
        label + " inconsistent confirmation diagnostic");
    require(
        adapter.calls() == 2,
        label + " confirmation uses independent eigensolver runs");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_classification_confirmation_retry_consensus(
    const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        recovering_confirmation_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    evaluator.set_classification_confirmation_count(2);
    evaluator.set_classification_retry_count(1);

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto result = evaluator.analyze_confirmed(state, 0.5);
    require(
        result.succeeded() && result.unstable.real == 2,
        label + " confirmation retry recovers repeated higher rank");
    require(
        result.classification_attempts == 3 && adapter.calls() == 3,
        label + " confirmation retry consumes one extra run");
    require(
        result.diagnostic.find("consensus recovered") !=
            std::string::npos,
        label + " confirmation retry diagnostic");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_recycling_transaction(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type = transactional_spectrum_adapter<
        vector_space_type,
        problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type = stability::analysis::stability_evaluator<
        vector_space_type,
        problem_type,
        adapter_type,
        classifier_type,
        initial_policy_type>;

    vector_space_type vector_space(3);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));

    typename vector_space_type::vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto successful = evaluator.analyze(state, 0.0);
    require(
        successful.succeeded() &&
            adapter.begin_calls() == 1 &&
            adapter.commit_calls() == 1 &&
            adapter.rollback_calls() == 0,
        label + " successful classification commits one recycle transaction");

    adapter.fail(true);
    const auto failed = evaluator.analyze(state, 0.0);
    require(
        !failed.succeeded() &&
            adapter.begin_calls() == 2 &&
            adapter.commit_calls() == 1 &&
            adapter.rollback_calls() == 1,
        label + " failed classification rolls back recycle transaction");

    adapter.fail(false);
    evaluator.set_classification_confirmation_count(2);
    const std::size_t executions_before = adapter.execute_calls();
    const auto confirmed = evaluator.analyze_confirmed(state, 0.0);
    require(
        confirmed.succeeded() &&
            adapter.execute_calls() - executions_before == 2 &&
            adapter.begin_calls() == 3 &&
            adapter.commit_calls() == 2 &&
            adapter.rollback_calls() == 1,
        label + " independent confirmations share one recycle transaction");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

template<class Backend>
void test_unexpected_secant_signature_recovery(
    const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        curved_branch_artifact_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;
    using newton_type =
        quadratic_branch_newton<
            vector_space_type,
            problem_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            problem_type,
            newton_type,
            evaluator_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    newton_type newton(&vector_space);
    refiner_type refiner(
        &vector_space,
        &problem,
        &newton,
        &evaluator);

    vector_type lower;
    vector_type upper;
    vector_type refined;
    vector_space.init_vector(lower);
    vector_space.init_vector(upper);
    vector_space.init_vector(refined);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(upper);
    vector_space.start_use_vector(refined);
    vector_space.assign_scalar(1.0, lower);
    vector_space.assign_scalar(1.0, upper);

    const auto lower_stability =
        evaluator.analyze(lower, -1.0);
    const auto upper_stability =
        evaluator.analyze(upper, 1.0);
    typename refiner_type::options_type options;
    options.maximum_iterations = 24;
    options.parameter_tolerance = 1.0e-6;
    options.correct_with_fixed_parameter_newton = false;
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
        label + " unexpected secant signature is recovered: status=" +
            stability::analysis::stability_transition_status_name(
                transition.status) +
            ", diagnostic=" + transition.diagnostic);
    require(
        transition.consistency_restarts == 1,
        label + " recovery records one consistency restart");
    require(
        newton.solve_calls() == transition.iterations,
        label + " corrected restart applies Newton to every midpoint");
    require(
        std::abs(transition.parameter) <= 1.0e-6,
        label + " recovered transition parameter");
    std::vector<double> refined_host(4);
    vector_space.get(
        refined,
        refined_host.data(),
        refined_host.size());
    require(
        std::abs(
            refined_host.front() -
            transition.parameter*transition.parameter) <=
            1.0e-12,
        label + " recovered state lies on the nonlinear branch");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
}

template<class Backend>
void test_coalescence_before_hopf_refinement(
    const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using adapter_type =
        coalescence_hopf_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            adapter_type,
            classifier_type,
            initial_policy_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            problem_type,
            newton_type,
            evaluator_type>;

    vector_space_type vector_space(4);
    problem_type problem(&vector_space);
    adapter_type adapter(&problem);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &adapter,
        classifier_type{},
        initial_policy_type(&problem));
    newton_type newton(&vector_space);
    refiner_type refiner(
        &vector_space,
        &problem,
        &newton,
        &evaluator);

    vector_type lower;
    vector_type coalesced;
    vector_type upper;
    vector_type refined;
    vector_space.init_vector(lower);
    vector_space.init_vector(coalesced);
    vector_space.init_vector(upper);
    vector_space.init_vector(refined);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(coalesced);
    vector_space.start_use_vector(upper);
    vector_space.start_use_vector(refined);
    vector_space.assign_scalar(-1.0, lower);
    vector_space.assign_scalar(0.0, coalesced);
    vector_space.assign_scalar(1.0, upper);

    const auto lower_stability =
        evaluator.analyze(lower, -1.0);
    const auto coalesced_stability =
        evaluator.analyze(coalesced, 0.0);
    const auto upper_stability =
        evaluator.analyze(upper, 1.0);
    require(
        lower_stability.unstable.as_pair() ==
            std::make_pair(2, 1),
        label + " coalescence test lower signature");
    require(
        coalesced_stability.unstable.as_pair() ==
            std::make_pair(0, 2),
        label + " coalescence test intermediate signature");
    require(
        upper_stability.unstable.as_pair() ==
            std::make_pair(0, 1),
        label + " coalescence test upper signature");
    require(
        lower_stability.unstable.real_subspace_dimension() ==
            coalesced_stability.unstable.
                real_subspace_dimension(),
        label + " real-pair coalescence preserves unstable dimension");

    typename refiner_type::options_type options;
    options.maximum_iterations = 32;
    options.parameter_tolerance = 1.0e-8;
    const auto coalescence_only = refiner.refine(
        lower,
        -1.0,
        lower_stability,
        coalesced,
        0.0,
        coalesced_stability,
        refined,
        options);
    require(
        coalescence_only.status ==
            stability::analysis::stability_transition_status::
                no_transition,
        label + " coalescence alone is not a stability transition");

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
        label + " coalescence followed by Hopf refinement succeeds");
    require(
        std::abs(transition.parameter - 0.25) <= 2.0e-7,
        label + " Hopf transition parameter");
    require(
        transition.before_stability.unstable.as_pair() ==
            std::make_pair(0, 2),
        label + " refined local pre-Hopf signature");
    require(
        transition.after_stability.unstable.as_pair() ==
            std::make_pair(0, 1),
        label + " refined local post-Hopf signature");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(coalesced);
    vector_space.stop_use_vector(lower);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(coalesced);
    vector_space.free_vector(lower);
}

struct quiet_log
{
    template<class... Args>
    void info_f(const char*, Args...)
    {
    }

    template<class... Args>
    void warning_f(const char*, Args...)
    {
    }

    void info(const std::string&)
    {
    }
};

template<class Backend>
void test_multiple_transition_sequence(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using eigensolver_adapter_type =
        two_real_crossings_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using facade_type =
        stability::stability_analysis<
            vector_space_type,
            problem_type,
            quiet_log,
            newton_type,
            eigensolver_adapter_type>;

    vector_space_type vector_space(3);
    problem_type problem(&vector_space);
    newton_type newton(&vector_space);
    eigensolver_adapter_type eigensolver_adapter(&problem);
    quiet_log log;
    facade_type facade(
        &vector_space,
        &log,
        &problem,
        &newton,
        &eigensolver_adapter);
    facade.set_transition_refinement_uses_fixed_parameter_newton(false);
    facade.set_transition_refinement_maximum_iterations(32);
    facade.set_transition_refinement_parameter_tolerance(1.0e-8);
    facade.set_transition_refinement_maximum_subdivisions(4);

    vector_type lower;
    vector_type upper;
    vector_space.init_vector(lower);
    vector_space.init_vector(upper);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(upper);
    vector_space.assign_scalar(-1.0, lower);
    vector_space.assign_scalar(1.0, upper);
    const auto lower_stability =
        facade.analyze_confirmed(lower, -1.0);
    const auto upper_stability =
        facade.analyze_confirmed(upper, 1.0);

    std::vector<double> parameters;
    std::vector<std::pair<
        std::pair<int, int>,
        std::pair<int, int>>> signatures;
    const std::size_t event_count =
        facade.refine_transition_sequence_confirmed(
            lower,
            -1.0,
            lower_stability,
            upper,
            1.0,
            upper_stability,
            [&parameters, &signatures](
                const typename facade_type::transition_result_type&
                    transition,
                const vector_type&)
            {
                parameters.push_back(transition.parameter);
                signatures.emplace_back(
                    transition.before_stability.
                        unstable_dimension_pair(),
                    transition.after_stability.
                        unstable_dimension_pair());
            });

    require(
        event_count == 2 && parameters.size() == 2,
        label + " two transitions are recovered from one interval");
    require(
        parameters.size() == 2 &&
            std::abs(parameters[0] + 0.25) <= 2.0e-7 &&
            std::abs(parameters[1] - 0.25) <= 2.0e-7,
        label + " transition sequence parameters");
    require(
        signatures.size() == 2 &&
            signatures[0] ==
            std::make_pair(
                std::make_pair(0, 0),
                std::make_pair(1, 0)) &&
            signatures[1] ==
                std::make_pair(
                    std::make_pair(1, 0),
                    std::make_pair(2, 0)),
        label + " transition sequence signatures");

    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
}

void test_structured_facade()
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using eigensolver_adapter_type =
        parameter_spectrum_adapter<
            vector_space_type,
            problem_type>;
    using facade_type =
        stability::stability_analysis<
            vector_space_type,
            problem_type,
            quiet_log,
            newton_type,
            eigensolver_adapter_type>;

    vector_space_type vector_space(3);
    problem_type problem(&vector_space);
    newton_type newton(&vector_space);
    eigensolver_adapter_type eigensolver_adapter(&problem);
    quiet_log log;
    facade_type facade(
        &vector_space,
        &log,
        &problem,
        &newton,
        &eigensolver_adapter);
    require(
        eigensolver_adapter.probe_generator_configured(),
        "structured facade configures matrix-free probe generator");
    vector_type configured_probe;
    vector_space.init_vector(configured_probe);
    vector_space.start_use_vector(configured_probe);
    vector_space.assign_scalar(0.0, configured_probe);
    eigensolver_adapter.generate_probe(
        std::size_t(1),
        configured_probe,
        configured_probe);
    double configured_probe_value = 0.0;
    vector_space.get(
        configured_probe,
        &configured_probe_value,
        std::size_t(1));
    require(
        configured_probe_value == 1.0,
        "structured facade reuses nonlinear stability probe policy");
    vector_space.stop_use_vector(configured_probe);
    vector_space.free_vector(configured_probe);
    facade.set_linear_operator_stable_eigenvalues_halfplane(-1.0);

    vector_type lower;
    vector_type upper;
    vector_type refined;
    vector_space.init_vector(lower);
    vector_space.init_vector(upper);
    vector_space.init_vector(refined);
    vector_space.start_use_vector(lower);
    vector_space.start_use_vector(upper);
    vector_space.start_use_vector(refined);
    vector_space.assign_scalar(-1.0, lower);
    vector_space.assign_scalar(1.0, upper);

    require(
        facade.execute(lower, -1.0) == std::make_pair(0, 0),
        "structured facade stable dimensions");
    require(
        facade.execute(upper, 1.0) == std::make_pair(1, 0),
        "structured facade unstable dimensions");

    double refined_parameter = 0.0;
    facade.bisect_bifurcation_point_known(
        lower,
        -1.0,
        {0, 0},
        upper,
        1.0,
        {1, 0},
        refined,
        refined_parameter,
        24);
    require(
        std::abs(refined_parameter) <= 2.0e-7,
        "structured facade transition refinement");

    vector_space.stop_use_vector(refined);
    vector_space.stop_use_vector(upper);
    vector_space.stop_use_vector(lower);
    vector_space.free_vector(refined);
    vector_space.free_vector(upper);
    vector_space.free_vector(lower);
}

template<class Backend>
void test_fold_secant_refinement(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type =
        nonlinear_operators::circle<vector_space_type>;
    using operator_type =
        nonlinear_operators::linear_operator_circle<
            vector_space_type,
            problem_type>;
    using scaled_operator_type =
        stability::eigensolvers::transformations::
            scaled_real_operator<
                vector_space_type,
                operator_type>;
    using eigensolver_type =
        stability::eigensolvers::direct_scalar_eigensolver<
            vector_space_type,
            scaled_operator_type>;
    using classifier_type =
        stability::analysis::spectrum_classifier<double>;
    using initial_policy_type =
        stability::analysis::nonlinear_operator_random_initial_vector<
            problem_type>;
    using evaluator_type =
        stability::analysis::stability_evaluator<
            vector_space_type,
            problem_type,
            eigensolver_type,
            classifier_type,
            initial_policy_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using refiner_type =
        stability::analysis::stability_transition_refiner<
            vector_space_type,
            problem_type,
            newton_type,
            evaluator_type>;

    vector_space_type vector_space(1);
    problem_type problem(1.0, 1, &vector_space);
    problem_type* problem_pointer = &problem;
    operator_type linear_operator(problem_pointer);
    scaled_operator_type stability_operator(
        vector_space,
        linear_operator,
        -1.0);
    eigensolver_type eigensolver(
        vector_space,
        stability_operator);
    evaluator_type evaluator(
        &vector_space,
        &problem,
        &eigensolver,
        classifier_type{},
        initial_policy_type(&problem));
    newton_type newton(&vector_space);
    newton.fail(true);
    refiner_type refiner(
        &vector_space,
        &problem,
        &newton,
        &evaluator);

    vector_type positive_state;
    vector_type negative_state;
    vector_type refined_state;
    vector_space.init_vector(positive_state);
    vector_space.init_vector(negative_state);
    vector_space.init_vector(refined_state);
    vector_space.start_use_vector(positive_state);
    vector_space.start_use_vector(negative_state);
    vector_space.start_use_vector(refined_state);
    const double positive = 0.02;
    const double negative = -positive;
    const double parameter =
        -std::sqrt(1.0 - positive*positive);
    vector_space.set(&positive, positive_state, 1);
    vector_space.set(&negative, negative_state, 1);

    const auto positive_stability =
        evaluator.analyze(positive_state, parameter);
    const auto negative_stability =
        evaluator.analyze(negative_state, parameter);
    require(
        positive_stability.succeeded() &&
            negative_stability.succeeded() &&
            positive_stability.unstable !=
                negative_stability.unstable,
        label + " fold endpoints have different stability");

    typename refiner_type::options_type options;
    options.maximum_iterations = 24;
    options.correct_with_fixed_parameter_newton = false;
    const auto transition = refiner.refine(
        positive_state,
        parameter,
        positive_stability,
        negative_state,
        parameter,
        negative_stability,
        refined_state,
        options);
    double refined = 1.0;
    vector_space.get(refined_state, &refined, 1);
    require(
        transition.succeeded(),
        label + " fold secant refinement succeeds");
    require(
        std::abs(refined) < 2.0e-7,
        label + " fold secant state approaches the singular point: " +
            std::to_string(refined));
    require(
        newton.solve_calls() == 0,
        label + " fold secant refinement does not call Newton");

    vector_space.stop_use_vector(refined_state);
    vector_space.stop_use_vector(negative_state);
    vector_space.stop_use_vector(positive_state);
    vector_space.free_vector(refined_state);
    vector_space.free_vector(negative_state);
    vector_space.free_vector(positive_state);
}

void test_facade_with_separate_linearization_provider()
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::serial_cpu, double>;
    using vector_type = typename vector_space_type::vector_type;
    using problem_type = parameterized_problem<vector_space_type>;
    using provider_type =
        separate_linearization_provider<vector_space_type>;
    using newton_type =
        exact_branch_newton<vector_space_type, problem_type>;
    using eigensolver_adapter_type =
        parameter_spectrum_adapter<
            vector_space_type,
            provider_type>;
    using facade_type =
        stability::stability_analysis<
            vector_space_type,
            problem_type,
            quiet_log,
            newton_type,
            eigensolver_adapter_type,
            provider_type>;

    vector_space_type vector_space(3);
    problem_type problem(&vector_space);
    provider_type provider;
    newton_type newton(&vector_space);
    eigensolver_adapter_type eigensolver_adapter(&provider);
    quiet_log log;
    facade_type facade(
        &vector_space,
        &log,
        &problem,
        &newton,
        &eigensolver_adapter,
        &provider);

    vector_type state;
    vector_space.init_vector(state);
    vector_space.start_use_vector(state);
    vector_space.assign_scalar(0.0, state);

    const auto result = facade.analyze(state, 0.75);
    require(
        result.succeeded(),
        "separate linearization provider classifies spectrum");
    require(
        result.unstable.real == 1,
        "separate linearization provider supplies spectrum parameter");
    require(
        provider.linearization_calls() == 1,
        "separate linearization provider prepares operator");
    require(
        problem.linearization_calls() == 0,
        "nonlinear operator is not used as stability provider");
    require(
        problem.randomize_calls() == 1,
        "nonlinear operator still initializes Arnoldi vector");

    vector_space.stop_use_vector(state);
    vector_space.free_vector(state);
}

} // namespace

int main()
{
    test_spectrum_classifier();
    test_distinct_stability_probe_policy<scfd::backend::serial_cpu>(
        "serial");
    test_distinct_stability_probe_policy<scfd::backend::omp>("OMP");
    test_evaluator_and_refiner<scfd::backend::serial_cpu>("serial");
    test_evaluator_and_refiner<scfd::backend::omp>("OMP");
    test_classification_retry<scfd::backend::serial_cpu>("serial");
    test_classification_retry<scfd::backend::omp>("OMP");
    test_classification_fallback<scfd::backend::serial_cpu>("serial");
    test_classification_fallback<scfd::backend::omp>("OMP");
    test_classification_confirmation<scfd::backend::serial_cpu>(
        "serial");
    test_classification_confirmation<scfd::backend::omp>("OMP");
    test_classification_confirmation_consensus<
        scfd::backend::serial_cpu>("serial");
    test_classification_confirmation_consensus<
        scfd::backend::omp>("OMP");
    test_classification_confirmation_retry_consensus<
        scfd::backend::serial_cpu>("serial");
    test_classification_confirmation_retry_consensus<
        scfd::backend::omp>("OMP");
    test_recycling_transaction<scfd::backend::serial_cpu>("serial");
    test_recycling_transaction<scfd::backend::omp>("OMP");
    test_multiple_transition_sequence<scfd::backend::serial_cpu>(
        "serial");
    test_multiple_transition_sequence<scfd::backend::omp>("OMP");
    test_transition_state_alignment<
        scfd::backend::serial_cpu>("serial");
    test_transition_state_alignment<
        scfd::backend::omp>("OMP");
    test_unexpected_secant_signature_recovery<
        scfd::backend::serial_cpu>("serial");
    test_unexpected_secant_signature_recovery<
        scfd::backend::omp>("OMP");
    test_coalescence_before_hopf_refinement<
        scfd::backend::serial_cpu>("serial");
    test_coalescence_before_hopf_refinement<
        scfd::backend::omp>("OMP");
    test_structured_facade();
    test_fold_secant_refinement<scfd::backend::serial_cpu>(
        "serial");
    test_fold_secant_refinement<scfd::backend::omp>("OMP");
    test_facade_with_separate_linearization_provider();

    std::cout << "Checks: " << checks
              << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
