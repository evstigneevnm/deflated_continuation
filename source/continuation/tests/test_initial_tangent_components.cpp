#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <continuation/initial_tangent_candidates.h>
#include <continuation/initial_tangent_chart_validator.h>
#include <continuation/initial_tangent_secant_builder.h>
#include <continuation/semicurve_tangent_cache.h>
#include <continuation/tangent_normalization.h>

namespace
{

int checks = 0;
int failures = 0;

void require_true(const char* name, const bool condition)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cerr << "FAIL " << name << '\n';
    }
}

void require_close(const char* name, const double value, const double expected)
{
    require_true(name, std::abs(value - expected) <= 1.0e-14);
}

struct mock_vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    explicit mock_vector_operations(const std::size_t size_): size(size_) {}

    void init_vector(vector_type& vector) const { vector.assign(size, 0.0); }
    void start_use_vector(vector_type&) const {}
    void stop_use_vector(vector_type&) const {}
    void free_vector(vector_type& vector) const { vector.clear(); }
    void assign(const vector_type& source, vector_type& destination) const { destination = source; }
    void assign_mul(
        const double source_factor,
        const vector_type& source,
        const double other_factor,
        const vector_type& other,
        vector_type& destination) const
    {
        destination.resize(size);
        for(std::size_t index = 0; index < size; ++index)
        {
            destination[index] =
                source_factor*source[index] + other_factor*other[index];
        }
    }
    void scale(const double factor, vector_type& vector) const
    {
        for(auto& value: vector) value *= factor;
    }
    double norm(const vector_type& vector) const
    {
        double result = 0.0;
        for(const auto value: vector) result += value*value;
        return std::sqrt(result);
    }
    double norm_rank1(const vector_type& vector, const double scalar) const
    {
        const double vector_norm = norm(vector);
        return std::sqrt(vector_norm*vector_norm + scalar*scalar);
    }
    bool check_is_valid_number(const vector_type& vector) const
    {
        for(const double value: vector)
        {
            if(!std::isfinite(value)) return false;
        }
        return true;
    }

    std::size_t size;
};

struct mock_log
{
    template<class... Args>
    void info_f(const char*, Args...) const {}

    template<class... Args>
    void warning_f(const char*, Args...) const {}
};

struct mock_nonlinear_operator
{
};

struct mock_newton
{
    bool solve(
        mock_nonlinear_operator*,
        mock_vector_operations::vector_type& x,
        const double lambda) const
    {
        for(std::size_t index = 0; index < x.size(); ++index)
        {
            x[index] = lambda*static_cast<double>(index + 1);
        }
        return true;
    }
};

void test_tangent_quality_policy()
{
    continuation::tangent_equation_quality_policy<double> equation_policy;
    const auto accurate_equation =
        continuation::make_tangent_equation_quality(
            1.0e-3,
            2.0,
            2.0);
    require_true(
        "small relative tangent residual is accepted",
        continuation::tangent_equation_quality_is_acceptable(
            accurate_equation,
            equation_policy));

    const auto inconsistent_equation =
        continuation::make_tangent_equation_quality(
            30.0,
            0.0,
            30.0);
    require_true(
        "pure-parameter inconsistent tangent is rejected",
        !continuation::tangent_equation_quality_is_acceptable(
            inconsistent_equation,
            equation_policy));
    require_close(
        "inconsistent tangent relative residual",
        inconsistent_equation.relative_residual,
        1.0);

    const auto roundoff_equation =
        continuation::make_tangent_equation_quality(
            1.0e-10,
            0.0,
            0.0);
    require_true(
        "roundoff-scale tangent residual is accepted",
        continuation::tangent_equation_quality_is_acceptable(
            roundoff_equation,
            equation_policy));

    continuation::projected_tangent_quality_policy<double> policy;
    continuation::tangent_candidate_quality<double> quality;
    quality.solved = true;
    quality.tangent_residual = 1.0e-4;
    quality.row_residual_abs = 1.0e-6;
    quality.pre_norm = 2.0;
    quality.lambda_s = 0.5;
    quality.orientation = 0.75;
    quality.residual_tol = 1.0e-3;
    quality.row_residual_tol = 1.0e-5;

    require_true(
        "valid tangent quality is accepted",
        continuation::projected_tangent_quality_is_acceptable(quality, policy));
    const double score = continuation::projected_tangent_candidate_score(quality, policy);
    require_true("valid tangent score is finite", std::isfinite(score));

    quality.pre_norm = 51.0;
    require_true(
        "oversized tangent candidate is rejected",
        !continuation::projected_tangent_quality_is_acceptable(quality, policy));

    policy.residual_tolerance_floor = 2.0;
    policy.residual_tolerance_ceiling = 1.0;
    bool rejected = false;
    try
    {
        policy.validate();
    }
    catch(const std::invalid_argument&)
    {
        rejected = true;
    }
    require_true("invalid tangent policy is rejected", rejected);
}

void test_semicurve_tangent_cache()
{
    mock_vector_operations vec_ops(3);
    continuation::semicurve_tangent_cache<mock_vector_operations> cache(&vec_ops);
    mock_vector_operations::vector_type source{1.0, -2.0, 3.0};
    mock_vector_operations::vector_type restored(3, 0.0);
    double lambda_component = 0.0;

    require_true("empty tangent cache does not restore", !cache.restore(-1, restored, lambda_component));
    cache.store(source, 0.25, -1);
    require_true("stored tangent cache is valid", cache.valid());
    require_true("stored tangent direction", cache.stored_direction() == -1);

    require_true("same direction restores", cache.restore(-1, restored, lambda_component));
    require_close("same direction component 0", restored[0], 1.0);
    require_close("same direction component 1", restored[1], -2.0);
    require_close("same direction lambda", lambda_component, 0.25);

    require_true("opposite direction restores", cache.restore(1, restored, lambda_component));
    require_close("opposite direction component 0", restored[0], -1.0);
    require_close("opposite direction component 1", restored[1], 2.0);
    require_close("opposite direction lambda", lambda_component, -0.25);

    cache.clear();
    require_true("cleared tangent cache is invalid", !cache.valid());
}

void test_tangent_candidate_selector()
{
    continuation::tangent_candidate_selector<double> selector;
    continuation::tangent_candidate_quality<double> first;
    first.valid = true;
    first.score = 3.0;
    first.lambda_s = 0.25;
    require_true("first candidate is selected", selector.consider(first));

    auto worse = first;
    worse.score = 4.0;
    require_true("worse candidate is ignored", !selector.consider(worse));

    auto better = first;
    better.score = 1.0;
    better.lambda_s = -0.5;
    require_true("better candidate replaces selection", selector.consider(better));
    require_close("selector returns best lambda component", selector.best().lambda_s, -0.5);

    auto invalid = better;
    invalid.valid = false;
    invalid.score = 0.0;
    require_true("invalid candidate is ignored", !selector.consider(invalid));
}

void test_tangent_normalization()
{
    mock_vector_operations vec_ops(2);
    mock_vector_operations::vector_type tangent{3.0, 4.0};
    double lambda_tangent = 12.0;
    require_true(
        "finite rank-one tangent normalizes",
        continuation::normalize_rank1_tangent(
            &vec_ops,
            tangent,
            lambda_tangent));
    require_close(
        "normalized rank-one tangent has unit norm",
        vec_ops.norm_rank1(tangent, lambda_tangent),
        1.0);

    tangent = {0.0, 0.0};
    lambda_tangent = 0.0;
    require_true(
        "zero rank-one tangent is rejected",
        !continuation::normalize_rank1_tangent(
            &vec_ops,
            tangent,
            lambda_tangent));

    tangent = {std::numeric_limits<double>::quiet_NaN(), 0.0};
    lambda_tangent = 1.0;
    require_true(
        "non-finite rank-one tangent is rejected",
        !continuation::normalize_rank1_tangent(
            &vec_ops,
            tangent,
            lambda_tangent));
}

void test_secant_builder_and_identity_chart()
{
    mock_vector_operations vec_ops(3);
    mock_log log;
    mock_newton newton;
    mock_nonlinear_operator nonlin_op;
    continuation::initial_tangent_secant_builder<
        mock_vector_operations,
        mock_log,
        mock_newton,
        mock_nonlinear_operator> builder(&vec_ops, &log, &newton);

    const mock_vector_operations::vector_type seed{1.0, 0.0, 0.0};
    const auto shifted = builder.solve_pair(&nonlin_op, 1.0, seed, 2.0);
    require_true("plus shifted Newton converges", shifted.plus_converged);
    require_true("minus shifted Newton converges", shifted.minus_converged);
    require_close("shifted Newton delta lambda", shifted.d_lambda, 1.0);

    mock_vector_operations::vector_type row(3, 0.0);
    double row_lambda = 0.0;
    const char* method = nullptr;
    require_true(
        "two-sided secant row is built",
        builder.build_row(
            1.0,
            seed,
            shifted,
            continuation::secant_candidate_kind::two_sided,
            row,
            row_lambda,
            method));
    require_close("two-sided row component 0", row[0], 2.0);
    require_close("two-sided row component 1", row[1], 4.0);
    require_close("two-sided row lambda", row_lambda, 2.0);
    require_true("two-sided row normalizes", builder.normalize(method, row, row_lambda));
    require_close("normalized secant rank-one norm", vec_ops.norm_rank1(row, row_lambda), 1.0);

    continuation::initial_tangent_chart_validator<
        mock_vector_operations,
        mock_log,
        mock_nonlinear_operator> validator(&vec_ops, &log);
    require_true(
        "ordinary operator uses identity chart validation",
        validator.accepts(
            &nonlin_op,
            seed,
            2.0,
            row,
            row_lambda,
            0.1,
            method,
            nullptr));
}

} // namespace

int main()
{
    test_tangent_quality_policy();
    test_tangent_candidate_selector();
    test_secant_builder_and_identity_chart();
    test_semicurve_tangent_cache();
    test_tangent_normalization();
    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
