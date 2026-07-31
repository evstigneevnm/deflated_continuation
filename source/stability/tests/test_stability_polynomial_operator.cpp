#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <stability/eigensolvers/transformations/stability_polynomial_operator.h>

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

bool close(double actual, double expected, double tolerance = 1.0e-13)
{
    return std::abs(actual - expected) <=
        tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template<class VectorSpace>
class diagonal_operator
{
public:
    using vector_type = typename VectorSpace::vector_type;

    diagonal_operator(
        const VectorSpace& vector_space,
        std::vector<double> diagonal)
        : vector_space_(vector_space),
          diagonal_(std::move(diagonal)),
          input_(diagonal_.size()),
          output_(diagonal_.size())
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        vector_space_.get(source, input_.data(), diagonal_.size());
        for(std::size_t index = 0; index < diagonal_.size(); ++index)
            output_[index] = diagonal_[index] * input_[index];
        vector_space_.set(output_.data(), destination, diagonal_.size());
        return true;
    }

private:
    const VectorSpace& vector_space_;
    std::vector<double> diagonal_;
    mutable std::vector<double> input_;
    mutable std::vector<double> output_;
};

template<class Backend>
void run_backend(const std::string& label)
{
    using vector_space_type =
        scfd_vector_operations<Backend, double>;
    using vector_type = typename vector_space_type::vector_type;
    using operator_type = diagonal_operator<vector_space_type>;
    using polynomial_type =
        stability::eigensolvers::transformations::
            stability_polynomial_operator<
                vector_space_type,
                operator_type>;

    const std::vector<double> diagonal{-2.0, -0.25, 0.0, 1.5};
    const std::vector<double> input{0.5, -1.0, 2.0, 0.25};
    constexpr double step = 0.1;
    constexpr std::size_t repetitions = 3;
    vector_space_type vector_space(diagonal.size());
    operator_type diagonal_matrix(vector_space, diagonal);

    vector_type source;
    vector_type destination;
    vector_space.init_vector(source);
    vector_space.init_vector(destination);
    vector_space.start_use_vector(source);
    vector_space.start_use_vector(destination);
    vector_space.set(input.data(), source, input.size());

    const auto euler_coefficients =
        stability::eigensolvers::transformations::
            explicit_euler_stability_polynomial<double>();
    polynomial_type euler(
        vector_space,
        diagonal_matrix,
        step,
        repetitions,
        euler_coefficients);
    require(euler.apply(source, destination), label + " Euler apply");
    std::vector<double> actual(input.size());
    vector_space.get(destination, actual.data(), actual.size());
    for(std::size_t index = 0; index < input.size(); ++index)
    {
        const double expected =
            input[index] *
            std::pow(1.0 + step * diagonal[index], repetitions);
        require(
            close(actual[index], expected),
            label + " Euler value");
    }
    require(
        euler.original_operator_calls() == repetitions,
        label + " Euler operator calls");

    const auto rk4_coefficients =
        stability::eigensolvers::transformations::
            classical_rk4_stability_polynomial<double>();
    polynomial_type rk4(
        vector_space,
        diagonal_matrix,
        step,
        repetitions,
        rk4_coefficients);
    require(rk4.apply(source, destination), label + " RK4 apply");
    vector_space.get(destination, actual.data(), actual.size());
    for(std::size_t index = 0; index < input.size(); ++index)
    {
        const double one_step =
            stability::eigensolvers::transformations::
                evaluate_stability_polynomial(
                    rk4_coefficients,
                    step * diagonal[index]);
        const double expected =
            input[index] * std::pow(one_step, repetitions);
        require(
            close(actual[index], expected),
            label + " RK4 value");
    }
    require(
        rk4.original_operator_calls() == 4 * repetitions,
        label + " RK4 operator calls");

    const std::complex<double> imaginary_argument(0.0, 0.18);
    const auto euler_amplification =
        stability::eigensolvers::transformations::
            evaluate_repeated_stability_polynomial(
                euler_coefficients,
                imaginary_argument,
                repetitions);
    const auto rk4_amplification =
        stability::eigensolvers::transformations::
            evaluate_repeated_stability_polynomial(
                rk4_coefficients,
                imaginary_argument,
                repetitions);
    require(
        std::abs(euler_amplification) > 1.01,
        label + " Euler exposes imaginary-axis amplification");
    require(
        std::abs(rk4_amplification) < 1.000001,
        label + " RK4 keeps imaginary-axis amplification near unity");

    vector_space.stop_use_vector(destination);
    vector_space.stop_use_vector(source);
    vector_space.free_vector(destination);
    vector_space.free_vector(source);
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
    std::cout << "Stability polynomial operator tests: PASSED\n";
    return EXIT_SUCCESS;
}
