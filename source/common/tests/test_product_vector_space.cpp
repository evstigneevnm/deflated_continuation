#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>

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

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

template<class Backend>
void run_case(const std::string& label)
{
    using component_space_type =
        scfd_vector_operations<Backend, double>;
    using product_space_type =
        nmfd::operations::two_block_vector_space<component_space_type>;
    using vector_type = typename product_space_type::vector_type;
    using multivector_type =
        typename product_space_type::multivector_type;

    component_space_type component_space(3);
    product_space_type space(component_space, component_space);

    vector_type x;
    vector_type y;
    space.init_vector(x);
    space.init_vector(y);
    space.start_use_vector(x);
    space.start_use_vector(y);

    const std::vector<double> host_x{1.0, 2.0, 3.0, -4.0, 5.0, -6.0};
    const std::vector<double> host_y{2.0, -1.0, 0.5, 3.0, 1.0, -2.0};
    space.set(host_x.data(), x, host_x.size());
    space.set(host_y.data(), y, host_y.size());

    require(space.get_default_size() == 6, label + " flattened size");
    require_close(space.norm_sq(x), 91.0, 1.0e-13, label + " norm squared");
    require_close(space.scalar_prod(x, y), 6.5, 1.0e-13, label + " dot");

    space.assign_lin_comb(2.0, x, -1.0, y, y);
    std::vector<double> actual(6);
    space.get(y, actual.data(), actual.size());
    const std::vector<double> expected{
        0.0, 5.0, 5.5, -11.0, 9.0, -10.0};
    for(std::size_t index = 0; index < actual.size(); ++index)
    {
        require_close(
            actual[index],
            expected[index],
            1.0e-13,
            label + " linear combination " + std::to_string(index));
    }

    multivector_type basis;
    space.init_multivector(basis, 2);
    space.start_use_multivector(basis, 2);
    space.assign(x, basis, 2, 0);
    space.assign(y, basis, 2, 1);
    space.assign_scalar(0.0, y);
    space.add_lin_comb(3.0, basis, 2, 0, 0.0, y);
    require_close(
        space.scalar_prod(basis, 2, 0, y),
        273.0,
        1.0e-12,
        label + " multivector column operations");

    space.stop_use_multivector(basis, 2);
    space.free_multivector(basis, 2);
    require(basis.empty(), label + " multivector release");

    space.stop_use_vector(y);
    space.stop_use_vector(x);
    space.free_vector(y);
    space.free_vector(x);
}

} // namespace

int main()
{
    run_case<scfd::backend::serial_cpu>("serial");
    run_case<scfd::backend::omp>("omp");

    std::cout << "Checks: " << checks
              << ", failures: " << failures << '\n';
    if(failures != 0)
    {
        std::cout << "FAILED\n";
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
