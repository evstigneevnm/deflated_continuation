#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>

#include <nonlinear_operators/bratu/bratu.h>
#include <nonlinear_operators/bratu/linear_operator_bratu.h>
#include <nonlinear_operators/tests/linear_nonlinear_decomposition_test.h>

#include <stability/eigensolvers/host_dense_operator_eigensolver.h>

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
    scfd_vector_operations<scfd::backend::omp, double>;
using vector_type = typename vector_space_type::vector_type;
using problem_type =
    nonlinear_operators::bratu<vector_space_type>;
using operator_type =
    nonlinear_operators::linear_operator_bratu<
        vector_space_type,
        problem_type>;
using dense_lapack_type =
    nmfd::operations::linalg::
        host_small_dense_lapack<double>;
using dense_eigensolver_type =
    stability::eigensolvers::
        host_dense_operator_eigensolver<
            vector_space_type,
            operator_type,
            dense_lapack_type>;

class vector_workspace
{
public:
    explicit vector_workspace(vector_space_type& vector_space)
        : vector_space_(vector_space)
    {
        vector_space_.init_vector(vector_);
        vector_space_.start_use_vector(vector_);
    }

    ~vector_workspace()
    {
        vector_space_.stop_use_vector(vector_);
        vector_space_.free_vector(vector_);
    }

    vector_type& get()
    {
        return vector_;
    }

private:
    vector_space_type& vector_space_;
    vector_type vector_;
};

double relative_error(
    vector_space_type& vector_space,
    const vector_type& actual,
    const vector_type& expected,
    vector_type& workspace)
{
    vector_space.assign_mul(
        1.0,
        actual,
        -1.0,
        expected,
        workspace);
    return vector_space.norm_l2(workspace)/
        std::max(1.0, vector_space.norm_l2(expected));
}

void test_residual_jacobian_decomposition(
    const problem_type::spatial_discretization discretization,
    const std::string& label)
{
    constexpr std::size_t n = 31;
    constexpr double parameter = 2.5;
    constexpr double finite_difference_step = 1.0e-6;
    vector_space_type vector_space(n);
    problem_type problem(
        n,
        &vector_space,
        discretization);
    vector_workspace state(vector_space);
    vector_workspace direction(vector_space);
    vector_workspace parameter_jacobian(vector_space);
    vector_workspace value_plus(vector_space);
    vector_workspace value_minus(vector_space);
    vector_workspace finite_difference(vector_space);
    vector_workspace difference(vector_space);

    std::vector<double> host_state(n);
    std::vector<double> host_direction(n);
    for(std::size_t index = 0; index < n; ++index)
    {
        const double x = static_cast<double>(index + 1)/
            static_cast<double>(n + 1);
        host_state[index] = 0.1*std::sin(std::acos(-1.0)*x);
        host_direction[index] = 0.07*std::sin(2.0*std::acos(-1.0)*x);
    }
    vector_space.set(host_state.data(), state.get(), n);
    vector_space.set(host_direction.data(), direction.get(), n);

    nonlinear_operators::tests::check_linear_nonlinear_decomposition(
        vector_space,
        problem,
        state.get(),
        direction.get(),
        parameter,
        finite_difference_step,
        2.0e-8,
        [](const bool condition, const std::string& message)
        {
            require(condition, message);
        },
        label
    );

    problem.F(
        state.get(),
        parameter + finite_difference_step,
        value_plus.get());
    problem.F(
        state.get(),
        parameter - finite_difference_step,
        value_minus.get());
    vector_space.assign_mul(
        1.0/(2.0*finite_difference_step),
        value_plus.get(),
        -1.0/(2.0*finite_difference_step),
        value_minus.get(),
        finite_difference.get());
    problem.jacobian_alpha(
        state.get(),
        parameter,
        parameter_jacobian.get());
    require(
        relative_error(
            vector_space,
            parameter_jacobian.get(),
            finite_difference.get(),
            difference.get()) < 2.0e-8,
        "Bratu parameter Jacobian finite difference");
}

void test_fd3_affine_inverse_and_eigenvectors()
{
    constexpr std::size_t n = 31;
    vector_space_type vector_space(n);
    problem_type problem(
        n,
        &vector_space,
        problem_type::spatial_discretization::fd3);
    operator_type linear_operator(&problem);
    vector_workspace state(vector_space);
    vector_workspace rhs(vector_space);
    vector_workspace original_rhs(vector_space);
    vector_workspace image(vector_space);
    vector_workspace residual(vector_space);
    vector_space.assign_scalar(0.0, state.get());
    problem.set_linearization_point(state.get(), 0.0);

    std::vector<double> host_rhs(n);
    for(std::size_t index = 0; index < n; ++index)
    {
        host_rhs[index] =
            1.0 +
            0.1*static_cast<double>(index) +
            0.03*std::sin(static_cast<double>(index));
    }
    vector_space.set(host_rhs.data(), rhs.get(), n);
    vector_space.assign(rhs.get(), original_rhs.get());
    const double jacobian_scale = 1.3;
    const double identity_shift = 0.7;
    problem.preconditioner_jacobian_affine_u(
        rhs.get(),
        jacobian_scale,
        identity_shift);
    linear_operator.apply(rhs.get(), image.get());
    vector_space.assign_mul(
        jacobian_scale,
        image.get(),
        identity_shift,
        rhs.get(),
        residual.get());
    require(
        relative_error(
            vector_space,
            residual.get(),
            original_rhs.get(),
            image.get()) < 2.0e-12,
        "Bratu affine inverse residual");

    const double pi = std::acos(-1.0);
    const double h = 1.0/static_cast<double>(n + 1);
    for(const std::size_t mode : {std::size_t(1), n/2, n})
    {
        std::vector<double> eigenvector(n);
        for(std::size_t index = 0; index < n; ++index)
        {
            eigenvector[index] = std::sin(
                static_cast<double>(mode)*
                pi*
                static_cast<double>(index + 1)/
                static_cast<double>(n + 1));
        }
        const double eigenvalue =
            -4.0/(h*h)*
            std::pow(
                std::sin(
                    static_cast<double>(mode)*pi/
                    (2.0*static_cast<double>(n + 1))),
                2);
        vector_space.set(
            eigenvector.data(),
            rhs.get(),
            n);
        linear_operator.apply(rhs.get(), image.get());
        vector_space.assign_mul(
            1.0,
            image.get(),
            -eigenvalue,
            rhs.get(),
            residual.get());
        require(
            vector_space.norm_l2(residual.get())/
                vector_space.norm_l2(rhs.get()) < 2.0e-10,
            "FD3 analytical eigenvector mode " +
                std::to_string(mode));
    }
}

std::vector<double> dense_spectrum(
    vector_space_type& vector_space,
    problem_type& problem,
    operator_type& linear_operator,
    vector_type& initial)
{
    dense_lapack_type lapack;
    dense_eigensolver_type eigensolver(
        vector_space,
        linear_operator,
        lapack);
    const auto result = eigensolver.execute(initial);
    require(
        result.succeeded(),
        "Bratu dense oracle succeeds: " + result.diagnostic);
    std::vector<double> spectrum;
    spectrum.reserve(result.eigenpairs.size());
    for(const auto& pair : result.eigenpairs)
    {
        require(
            pair.converged &&
                std::abs(pair.value.imag()) < 1.0e-9,
            "Bratu dense eigenpair residual");
        spectrum.push_back(pair.value.real());
    }
    std::sort(spectrum.begin(), spectrum.end());
    return spectrum;
}

void test_fd3_dense_spectrum()
{
    constexpr std::size_t n = 31;
    vector_space_type vector_space(n);
    problem_type problem(
        n,
        &vector_space,
        problem_type::spatial_discretization::fd3);
    operator_type linear_operator(&problem);
    vector_workspace state(vector_space);
    vector_workspace initial(vector_space);
    vector_space.assign_scalar(0.0, state.get());
    vector_space.assign_scalar(1.0, initial.get());
    problem.set_linearization_point(state.get(), 0.0);
    const auto actual = dense_spectrum(
        vector_space,
        problem,
        linear_operator,
        initial.get());

    const double pi = std::acos(-1.0);
    const double h = 1.0/static_cast<double>(n + 1);
    std::vector<double> expected;
    expected.reserve(n);
    for(std::size_t mode = 1; mode <= n; ++mode)
    {
        expected.push_back(
            -4.0/(h*h)*
            std::pow(
                std::sin(
                    static_cast<double>(mode)*pi/
                    (2.0*static_cast<double>(n + 1))),
                2));
    }
    std::sort(expected.begin(), expected.end());
    require(
        actual.size() == expected.size(),
        "FD3 dense spectrum count");
    for(std::size_t index = 0;
        index < std::min(actual.size(), expected.size());
        ++index)
    {
        const double scale =
            std::max(1.0, std::abs(expected[index]));
        require(
            std::abs(actual[index] - expected[index])/scale <
                2.0e-11,
            "FD3 dense eigenvalue " +
                std::to_string(index));
    }
}

void test_branch_morse_index()
{
    constexpr std::size_t n = 31;
    vector_space_type vector_space(n);
    problem_type problem(
        n,
        &vector_space,
        problem_type::spatial_discretization::fd3);
    operator_type linear_operator(&problem);
    vector_workspace state(vector_space);
    vector_workspace initial(vector_space);
    vector_space.assign_scalar(1.0, initial.get());

    problem.exact_solution_from_theta(1.0, state.get());
    problem.set_linearization_point(
        state.get(),
        problem.lambda_from_theta(1.0));
    const auto lower = dense_spectrum(
        vector_space,
        problem,
        linear_operator,
        initial.get());
    require(
        !lower.empty() && lower.back() < 0.0,
        "Bratu lower branch is linearly stable");

    problem.exact_solution_from_theta(4.0, state.get());
    problem.set_linearization_point(
        state.get(),
        problem.lambda_from_theta(4.0));
    const auto upper = dense_spectrum(
        vector_space,
        problem,
        linear_operator,
        initial.get());
    const std::size_t positive = static_cast<std::size_t>(
        std::count_if(
            upper.begin(),
            upper.end(),
            [](double value)
            {
                return value > 1.0e-8;
            }));
    require(
        positive == 1,
        "Bratu upper branch has Morse index one");
}

} // namespace

int main()
{
    test_residual_jacobian_decomposition(
        problem_type::spatial_discretization::fd3,
        "Bratu FD3");
    test_residual_jacobian_decomposition(
        problem_type::spatial_discretization::chebyshev,
        "Bratu Chebyshev");
    test_fd3_affine_inverse_and_eigenvectors();
    test_fd3_dense_spectrum();
    test_branch_morse_index();
    std::cout
        << "Bratu stability operator checks: "
        << checks << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
