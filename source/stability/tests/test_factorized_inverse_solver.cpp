#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <stability/eigensolvers/transformations/euler_polynomial_factorization.h>
#include <stability/eigensolvers/transformations/factorized_inverse_solver.h>

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

bool close(
    const std::complex<double>& actual,
    const std::complex<double>& expected,
    double tolerance = 1.0e-12)
{
    return std::abs(actual - expected) <=
        tolerance *
        std::max({1.0, std::abs(actual), std::abs(expected)});
}

template<class ProductSpace>
class diagonal_complex_affine_solver
{
public:
    using vector_type = typename ProductSpace::vector_type;
    using norm_type = typename ProductSpace::norm_type;

    diagonal_complex_affine_solver(
        const ProductSpace& vector_space,
        std::vector<double> diagonal,
        std::complex<double> operator_scale,
        std::complex<double> diagonal_shift,
        bool succeeds = true)
        : vector_space_(vector_space),
          diagonal_(std::move(diagonal)),
          operator_scale_(operator_scale),
          diagonal_shift_(diagonal_shift),
          succeeds_(succeeds),
          input_real_(diagonal_.size()),
          input_imag_(diagonal_.size()),
          output_real_(diagonal_.size()),
          output_imag_(diagonal_.size())
    {
    }

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        if(!succeeds_)
            return false;
        const std::size_t size = diagonal_.size();
        vector_space_.first_space().get(
            right_hand_side.first,
            input_real_.data(),
            size);
        vector_space_.second_space().get(
            right_hand_side.second,
            input_imag_.data(),
            size);
        for(std::size_t index = 0; index < size; ++index)
        {
            const std::complex<double> coefficient =
                operator_scale_ * diagonal_[index] +
                diagonal_shift_;
            const std::complex<double> value =
                std::complex<double>(
                    input_real_[index],
                    input_imag_[index]) /
                coefficient;
            output_real_[index] = value.real();
            output_imag_[index] = value.imag();
        }
        vector_space_.first_space().set(
            output_real_.data(),
            solution.first,
            size);
        vector_space_.second_space().set(
            output_imag_.data(),
            solution.second,
            size);
        return true;
    }

private:
    const ProductSpace& vector_space_;
    std::vector<double> diagonal_;
    std::complex<double> operator_scale_;
    std::complex<double> diagonal_shift_;
    bool succeeds_;
    mutable std::vector<double> input_real_;
    mutable std::vector<double> input_imag_;
    mutable std::vector<double> output_real_;
    mutable std::vector<double> output_imag_;
};

template<class Backend>
void run_backend(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<
            component_space_type>;
    using vector_type = typename product_space_type::vector_type;
    using factor_solver_type =
        diagonal_complex_affine_solver<product_space_type>;
    using solver_type =
        stability::eigensolvers::transformations::
            factorized_inverse_solver<
                product_space_type,
                factor_solver_type>;

    constexpr double step = 0.1;
    constexpr std::size_t power = 3;
    const std::complex<double> shift =
        std::polar(1.01, 0.47);
    const auto factors =
        stability::eigensolvers::transformations::
            euler_denominator_factors(step, power, shift);

    const std::vector<std::complex<double>> probes{
        {-2.1, 0.7},
        {-0.4, -1.3},
        {0.0, 1.68},
        {0.25, 0.0}};
    for(const auto& probe : probes)
    {
        const auto actual =
            stability::eigensolvers::transformations::
                evaluate_affine_factors(factors, probe);
        const auto expected =
            std::pow(
                std::complex<double>(1.0, 0.0) +
                    step * probe,
                power) -
            shift;
        require(
            close(actual, expected),
            label + " Euler polynomial factor identity");
    }

    const std::vector<double> diagonal{-1.25, 0.2, 2.4};
    const std::vector<std::complex<double>> right_hand_side{
        {1.0, -0.25},
        {-0.3, 0.8},
        {2.0, 1.5}};
    component_space_type component_space(diagonal.size());
    product_space_type product_space(
        component_space,
        component_space);

    std::vector<std::shared_ptr<factor_solver_type>> factor_solvers;
    for(const auto& factor : factors)
    {
        factor_solvers.push_back(
            std::make_shared<factor_solver_type>(
                product_space,
                diagonal,
                factor.operator_scale,
                factor.diagonal_shift));
    }
    solver_type solver(product_space, factor_solvers);

    vector_type right;
    vector_type solution;
    product_space.init_vector(right);
    product_space.init_vector(solution);
    product_space.start_use_vector(right);
    product_space.start_use_vector(solution);
    std::vector<double> packed(2 * diagonal.size());
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        packed[index] = right_hand_side[index].real();
        packed[diagonal.size() + index] =
            right_hand_side[index].imag();
    }
    product_space.set(packed.data(), right, packed.size());

    require(solver.solve(right, solution), label + " factorized solve");
    product_space.get(solution, packed.data(), packed.size());
    for(std::size_t index = 0; index < diagonal.size(); ++index)
    {
        const std::complex<double> actual(
            packed[index],
            packed[diagonal.size() + index]);
        const std::complex<double> expected =
            right_hand_side[index] /
            (std::pow(
                 std::complex<double>(1.0, 0.0) +
                     step * diagonal[index],
                 power) -
             shift);
        require(
            close(actual, expected),
            label + " factorized inverse value");
    }
    require(
        solver.factor_count() == power,
        label + " factor count");
    require(
        solver.factor_solve_calls() == power,
        label + " factor solve accounting");
    require(
        solver.failed_solves() == 0,
        label + " success accounting");

    auto failing_factors = factor_solvers;
    failing_factors[1] =
        std::make_shared<factor_solver_type>(
            product_space,
            diagonal,
            factors[1].operator_scale,
            factors[1].diagonal_shift,
            false);
    solver_type failing_solver(
        product_space,
        std::move(failing_factors));
    require(
        !failing_solver.solve(right, solution),
        label + " factor failure propagation");
    require(
        failing_solver.factor_solve_calls() == 2,
        label + " stops after failed factor");
    require(
        failing_solver.failed_solves() == 1,
        label + " failed solve accounting");

    product_space.stop_use_vector(solution);
    product_space.stop_use_vector(right);
    product_space.free_vector(solution);
    product_space.free_vector(right);
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu>("serial");
    run_backend<scfd::backend::omp>("omp");
    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
        return EXIT_FAILURE;
    std::cout << "Factorized inverse solver tests: PASSED\n";
    return EXIT_SUCCESS;
}
