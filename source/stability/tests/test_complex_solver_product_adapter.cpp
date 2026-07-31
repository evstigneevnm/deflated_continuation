#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_backend_ext/complex.h>
#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/scfd_complex_vector_bridge.h>
#include <stability/eigensolvers/transformations/complex_solver_product_adapter.h>
#include <stability/eigensolvers/transformations/pointwise_diagonal_preconditioner.h>

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

struct diagonal_operator_tag
{
};

template<class Backend>
void run_backend(const std::string& label)
{
    using real_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<
            real_space_type>;
    using complex_type =
        common::scfd_backend_ext::complex_t<Backend, double>;
    using complex_traits =
        common::scfd_backend_ext::complex_value_traits<
            complex_type>;
    using complex_space_type =
        scfd_vector_operations<Backend, complex_type>;
    using bridge_type =
        nmfd::operations::scfd_complex_vector_bridge<
            product_space_type,
            complex_space_type>;
    using solver_type =
        stability::eigensolvers::transformations::
            pointwise_diagonal_preconditioner<
                complex_space_type,
                diagonal_operator_tag>;
    using adapter_type =
        stability::eigensolvers::transformations::
            complex_solver_product_adapter<
                product_space_type,
                complex_space_type,
                bridge_type,
                solver_type>;

    constexpr std::size_t size = 4;
    real_space_type real_space(size);
    product_space_type product_space(real_space, real_space);
    complex_space_type complex_space(size);
    bridge_type bridge(product_space, complex_space);

    typename product_space_type::vector_type right;
    typename product_space_type::vector_type solution;
    typename complex_space_type::vector_type packed;
    typename complex_space_type::vector_type diagonal;
    product_space.init_vector(right);
    product_space.init_vector(solution);
    complex_space.init_vector(packed);
    complex_space.init_vector(diagonal);
    product_space.start_use_vector(right);
    product_space.start_use_vector(solution);
    complex_space.start_use_vector(packed);
    complex_space.start_use_vector(diagonal);

    const std::vector<double> right_host{
        1.0, -0.5, 2.0, -1.0,
        -0.2, 0.7, 1.5, -0.8};
    const std::vector<std::complex<double>> diagonal_host{
        {2.0, 0.4},
        {-1.0, -0.3},
        {0.5, 1.2},
        {3.0, -2.0}};
    std::vector<complex_type> backend_diagonal(size);
    for(std::size_t index = 0; index < size; ++index)
    {
        backend_diagonal[index] = complex_traits::make(
            diagonal_host[index].real(),
            diagonal_host[index].imag());
    }
    product_space.set(
        right_host.data(),
        right,
        right_host.size());
    complex_space.set(
        backend_diagonal.data(),
        diagonal,
        backend_diagonal.size());

    bridge.pack(right, packed);
    std::vector<complex_type> packed_host(size);
    complex_space.get(
        packed,
        packed_host.data(),
        packed_host.size());
    for(std::size_t index = 0; index < size; ++index)
    {
        require(
            std::abs(
                std::complex<double>(
                    complex_traits::real(packed_host[index]),
                    complex_traits::imag(packed_host[index])) -
                std::complex<double>(
                    right_host[index],
                    right_host[size + index])) <= 1.0e-13,
            label + " bridge pack");
    }

    solver_type solver(complex_space, diagonal);
    adapter_type adapter(
        product_space,
        complex_space,
        bridge,
        solver);
    require(
        adapter.solve(right, solution),
        label + " product adapter solve");
    std::vector<double> solution_host(2 * size);
    product_space.get(
        solution,
        solution_host.data(),
        solution_host.size());
    for(std::size_t index = 0; index < size; ++index)
    {
        const std::complex<double> actual(
            solution_host[index],
            solution_host[size + index]);
        const std::complex<double> expected =
            std::complex<double>(
                right_host[index],
                right_host[size + index]) /
            diagonal_host[index];
        require(
            std::abs(actual - expected) <=
                2.0e-13 * std::max(1.0, std::abs(expected)),
            label + " adapter value");
    }
    require(
        adapter.solve_calls() == 1 &&
        adapter.failed_solves() == 0,
        label + " adapter accounting");

    complex_space.stop_use_vector(diagonal);
    complex_space.stop_use_vector(packed);
    product_space.stop_use_vector(solution);
    product_space.stop_use_vector(right);
    complex_space.free_vector(diagonal);
    complex_space.free_vector(packed);
    product_space.free_vector(solution);
    product_space.free_vector(right);
}

} // namespace

int main()
{
    run_backend<scfd::backend::serial_cpu>("serial");
    run_backend<scfd::backend::omp>("omp");
    std::cout
        << "Checks: " << checks
        << ", failures: " << failures << '\n';
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
