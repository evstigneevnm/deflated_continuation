#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nmfd/operations/linalg/small_dense.h>
#include <symmetry/fourier/frozen_fourier_chart_1d.h>

namespace
{

using real = double;
using complex = std::complex<real>;
using chart_t = symmetry::fourier::frozen_fourier_chart_1d<complex>;

int checks = 0;
int failures = 0;

void require(const bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}

real norm(const std::vector<complex>& values)
{
    real sum = 0;
    for(const auto& value: values)
    {
        sum += std::norm(value);
    }
    return std::sqrt(sum);
}

std::vector<complex> add_scaled(
    const std::vector<complex>& left,
    const real scale,
    const std::vector<complex>& right)
{
    std::vector<complex> result(left.size());
    for(std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = left[i] + scale*right[i];
    }
    return result;
}

std::vector<complex> difference_scaled(
    const std::vector<complex>& plus,
    const std::vector<complex>& minus,
    const real denominator)
{
    std::vector<complex> result(plus.size());
    for(std::size_t i = 0; i < result.size(); ++i)
    {
        result[i] = (plus[i]-minus[i])/denominator;
    }
    return result;
}

void require_close(
    const std::vector<complex>& actual,
    const std::vector<complex>& expected,
    const real tolerance,
    const std::string& label)
{
    std::vector<complex> delta(actual.size());
    for(std::size_t i = 0; i < delta.size(); ++i)
    {
        delta[i] = actual[i]-expected[i];
    }
    const real error = norm(delta);
    const real scale = real(1)+norm(expected);
    require(error <= tolerance*scale, label + " error=" + std::to_string(error));
}

std::vector<complex> diagonal_field(const std::vector<complex>& state)
{
    std::vector<complex> result(state.size());
    for(std::size_t mode = 0; mode < state.size(); ++mode)
    {
        const real multiplier = real(0.3) + real(0.17)*static_cast<real>(mode);
        result[mode] = multiplier*state[mode];
    }
    return result;
}

std::vector<complex> projected_field(chart_t& chart, const std::vector<complex>& source)
{
    const auto at = chart.evaluate(source);
    return chart.project_tangent(at, diagonal_field(at.state_on_slice));
}

real inner(const std::vector<complex>& left, const std::vector<complex>& right)
{
    real result = 0;
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        result += left[i].real()*right[i].real() + left[i].imag()*right[i].imag();
    }
    return result;
}

std::vector<complex> real_to_complex(const std::vector<real>& values)
{
    std::vector<complex> result(values.size()/2 + 1, complex(0, 0));
    for(std::size_t mode = 1; mode < result.size(); ++mode)
    {
        result[mode] = complex(values[2*(mode-1)], values[2*(mode-1)+1]);
    }
    return result;
}

void test_lsq_stabilizer_differential()
{
    const std::vector<complex> reference{
        complex(0, 0),
        complex(0.8, -0.2),
        complex(-0.35, 0.6),
        complex(0.5, 0.15),
        complex(-0.1, 0.08),
        complex(0.2, -0.3)};
    chart_t chart;
    chart_t::lsq_options_type options;
    options.mode_min = 1;
    options.mode_max = 5;
    options.max_active_modes = 4;
    options.grid_points = 128;
    options.newton_iterations = 12;
    chart.freeze_lsq(reference, options, 1.0e-12);

    require(chart.frozen(), "LSQ chart is frozen");
    require(chart.active_modes().size() == 4, "LSQ chart freezes the selected mode set");

    const std::vector<complex> source{
        complex(0, 0),
        complex(0.72, -0.12),
        complex(-0.28, 0.55),
        complex(0.48, 0.21),
        complex(-0.07, 0.12),
        complex(0.18, -0.24)};
    const std::vector<complex> tangent{
        complex(0, 0),
        complex(0.11, -0.07),
        complex(-0.05, 0.09),
        complex(0.04, -0.03),
        complex(0.02, 0.01),
        complex(-0.08, 0.06)};

    const auto at = chart.evaluate(source);
    const auto predicted = chart.stabilizer_differential(at, tangent);
    const real eps = 1.0e-7;
    const auto plus = chart.evaluate(add_scaled(source, eps, tangent)).state_on_slice;
    const auto minus = chart.evaluate(add_scaled(source, -eps, tangent)).state_on_slice;
    const auto finite_difference = difference_scaled(plus, minus, real(2)*eps);
    require_close(predicted, finite_difference, 2.0e-7, "frozen LSQ stabilizer differential");

    const real phase = chart_t::lsq_solver_type::phase_value(
        predicted, chart.reference_on_slice(), chart.active_modes());
    require(std::abs(phase) <= 1.0e-11, "frozen LSQ differential is tangent to the fixed slice");
}

void test_lsq_projected_field_differential()
{
    const std::vector<complex> reference{
        complex(0, 0),
        complex(0.7, 0.1),
        complex(-0.4, 0.5),
        complex(0.3, -0.25),
        complex(0.18, 0.09)};
    chart_t chart;
    chart_t::lsq_options_type options;
    options.max_active_modes = 4;
    options.grid_points = 128;
    options.newton_iterations = 12;
    chart.freeze_lsq(reference, options, 1.0e-12);

    const std::vector<complex> source{
        complex(0, 0),
        complex(0.62, 0.16),
        complex(-0.31, 0.46),
        complex(0.27, -0.18),
        complex(0.12, 0.13)};
    const std::vector<complex> tangent{
        complex(0, 0),
        complex(0.08, -0.04),
        complex(-0.03, 0.07),
        complex(0.05, 0.02),
        complex(-0.02, 0.01)};

    const auto at = chart.evaluate(source);
    const auto tangent_on_slice = chart.stabilizer_differential(at, tangent);
    const auto field = diagonal_field(at.state_on_slice);
    const auto field_derivative = diagonal_field(tangent_on_slice);
    const auto predicted = chart.projected_vector_field_differential(
        at, tangent_on_slice, field, field_derivative);

    const real eps = 1.0e-7;
    const auto plus = projected_field(chart, add_scaled(source, eps, tangent));
    const auto minus = projected_field(chart, add_scaled(source, -eps, tangent));
    const auto finite_difference = difference_scaled(plus, minus, real(2)*eps);
    require_close(predicted, finite_difference, 5.0e-7, "frozen LSQ projected-field Jacobian");
}

void test_mode_set_does_not_change_during_evaluation()
{
    const std::vector<complex> reference{
        complex(0, 0),
        complex(1.0, 0.0),
        complex(0.8, 0.0),
        complex(0.6, 0.0),
        complex(0.01, 0.0)};
    chart_t chart;
    chart_t::lsq_options_type options;
    options.max_active_modes = 2;
    chart.freeze_lsq(reference, options, 1.0e-12);
    const auto frozen_modes = chart.active_modes();

    auto changed = reference;
    changed[4] = complex(100.0, 0.0);
    (void)chart.evaluate(changed);
    require(chart.active_modes() == frozen_modes, "evaluation cannot reselect frozen LSQ modes");
}

void test_lsq_projected_bordered_correction()
{
    const std::vector<complex> reference{
        complex(0, 0),
        complex(0.9, -0.2),
        complex(-0.45, 0.55),
        complex(0.25, 0.35)};
    chart_t chart;
    chart_t::lsq_options_type options;
    options.max_active_modes = 3;
    options.grid_points = 128;
    options.newton_iterations = 12;
    chart.freeze_lsq(reference, options, 1.0e-12);
    const auto at = chart.evaluate(reference);

    const std::size_t real_dimension = 2*(reference.size()-1);
    std::vector<real> phase_covector(real_dimension, real(0));
    for(const auto mode: chart.active_modes())
    {
        const complex r = chart.reference_on_slice()[mode];
        const std::size_t offset = 2*(mode-1);
        phase_covector[offset] = -static_cast<real>(mode)*r.imag();
        phase_covector[offset+1] = static_cast<real>(mode)*r.real();
    }
    real phase_norm_sq = 0;
    for(const auto value: phase_covector)
    {
        phase_norm_sq += value*value;
    }
    require(phase_norm_sq > 0, "LSQ bordered test has a nonzero phase covector");

    std::vector<std::vector<complex>> basis;
    for(std::size_t column = 0; column < real_dimension && basis.size()+1 < real_dimension; ++column)
    {
        std::vector<real> candidate(real_dimension, real(0));
        candidate[column] = real(1);
        const real phase_component = phase_covector[column]/phase_norm_sq;
        for(std::size_t i = 0; i < real_dimension; ++i)
        {
            candidate[i] -= phase_component*phase_covector[i];
        }

        auto candidate_complex = real_to_complex(candidate);
        for(const auto& q: basis)
        {
            const real coefficient = inner(candidate_complex, q);
            for(std::size_t i = 0; i < candidate_complex.size(); ++i)
            {
                candidate_complex[i] -= coefficient*q[i];
            }
        }
        const real candidate_norm = norm(candidate_complex);
        if(candidate_norm <= 1.0e-10)
        {
            continue;
        }
        for(auto& value: candidate_complex)
        {
            value /= candidate_norm;
        }
        basis.push_back(candidate_complex);
    }
    require(basis.size()+1 == real_dimension, "LSQ bordered test constructed a full tangent basis");

    std::vector<complex> jlambda(reference.size(), complex(0, 0));
    for(std::size_t mode = 0; mode < jlambda.size(); ++mode)
    {
        jlambda[mode] = basis[0][mode] + real(0.2)*basis[1][mode];
    }
    jlambda = chart.project_tangent(at, jlambda);

    constexpr std::size_t max_size = 8;
    const std::size_t system_size = basis.size()+1;
    nmfd::operations::linalg::small_matrix<real, max_size> matrix(system_size, system_size);
    std::vector<std::vector<complex>> jacobian_columns;
    jacobian_columns.reserve(basis.size());
    for(const auto& q: basis)
    {
        jacobian_columns.push_back(
            chart.projected_vector_field_differential(at, q, at.state_on_slice, q));
    }
    for(std::size_t row = 0; row < basis.size(); ++row)
    {
        for(std::size_t column = 0; column < basis.size(); ++column)
        {
            matrix(row, column) = inner(basis[row], jacobian_columns[column]);
        }
        matrix(row, basis.size()) = inner(basis[row], jlambda);
    }
    for(std::size_t column = 0; column < basis.size(); ++column)
    {
        matrix(basis.size(), column) = real(0.07)*static_cast<real>(column+1);
    }
    matrix(basis.size(), basis.size()) = real(0.8);

    nmfd::operations::linalg::small_vector<real, max_size> expected(system_size);
    for(std::size_t i = 0; i < basis.size(); ++i)
    {
        expected[i] = real(0.03)*static_cast<real>(i+1);
    }
    expected[basis.size()] = real(-0.17);

    nmfd::operations::linalg::small_vector<real, max_size> rhs(system_size);
    for(std::size_t row = 0; row < system_size; ++row)
    {
        rhs[row] = real(0);
        for(std::size_t column = 0; column < system_size; ++column)
        {
            rhs[row] += matrix(row, column)*expected[column];
        }
    }

    nmfd::operations::linalg::small_vector<real, max_size> correction;
    const auto solve_info = nmfd::operations::linalg::solve(matrix, rhs, correction);
    require(solve_info.ok(), "LSQ reduced bordered corrector solve succeeds");
    if(solve_info.ok())
    {
        real maximum_error = 0;
        for(std::size_t i = 0; i < system_size; ++i)
        {
            maximum_error = std::max(maximum_error, std::abs(correction[i]-expected[i]));
        }
        require(maximum_error <= 1.0e-12, "LSQ bordered correction satisfies vector and arclength rows");
    }
}

} // namespace

int main()
{
    try
    {
        test_lsq_stabilizer_differential();
        test_lsq_projected_field_differential();
        test_mode_set_does_not_change_during_evaluation();
        test_lsq_projected_bordered_correction();
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED with exception: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASSED\n";
    return EXIT_SUCCESS;
}
