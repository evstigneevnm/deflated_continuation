#include <cmath>
#include <algorithm>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include <symmetry/fourier/fourier_slice.h>

namespace
{

int checks = 0;
int failures = 0;

using complex_type = std::complex<double>;
using slice_type = symmetry::fourier::fourier_slice_1d<complex_type>;

void require(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        std::cout << "FAIL " << label << std::endl;
        ++failures;
    }
}

void require_near(const std::string& label, double value, double expected, double tol)
{
    ++checks;
    const double err = std::abs(value - expected);
    if(!(err <= tol))
    {
        std::cout << "FAIL " << label << " value=" << value
                  << " expected=" << expected << " err=" << err
                  << " tol=" << tol << std::endl;
        ++failures;
    }
}

void require_complex_near(const std::string& label, complex_type value, complex_type expected, double tol)
{
    require_near(label + " real", value.real(), expected.real(), tol);
    require_near(label + " imag", value.imag(), expected.imag(), tol);
}

void require_vector_near(
    const std::string& label,
    const std::vector<complex_type>& value,
    const std::vector<complex_type>& expected,
    double tol)
{
    require(value.size() == expected.size(), label + " size");
    const std::size_t n = std::min(value.size(), expected.size());
    for(std::size_t i = 0; i < n; ++i)
        require_complex_near(label + " [" + std::to_string(i) + "]", value[i], expected[i], tol);
}

std::vector<complex_type> sample_spectrum()
{
    return {
        complex_type(0.0, 0.0),
        complex_type(0.0, 0.0),
        complex_type(0.0, 0.0),
        complex_type(2.0, 1.0),
        complex_type(-0.5, 0.25),
        complex_type(0.1, -0.2),
        complex_type(0.0, 0.0),
        complex_type(0.0, 0.0)
    };
}

void test_stabilization()
{
    const slice_type slice(1e-12);
    const auto input = sample_spectrum();

    slice_type::slice_data data;
    data.mode = 3;
    const auto stabilized = slice.stabilize(input, data);

    require(data.active(), "stabilization active");
    require(data.active_rank == 1, "active rank one");
    require(data.mode == 3, "selected preferred mode");
    require(data.residual_group_order() == 3, "residual group order");
    require_near("selected abs", data.selected_abs, std::sqrt(5.0), 1e-14);
    require_near("slice matrix", data.slice_matrix, 3.0*std::sqrt(5.0), 1e-14);
    require_near("stabilized selected imag", stabilized[3].imag(), 0.0, 1e-14);
    require_near("stabilized selected real", stabilized[3].real(), std::sqrt(5.0), 1e-14);
}

void test_shift_invariance_and_idempotence()
{
    const slice_type slice(1e-12);
    const auto input = sample_spectrum();

    slice_type::slice_data data_a;
    data_a.mode = 3;
    const auto stabilized_a = slice.stabilize(input, data_a);

    const auto shifted = slice.apply_shift(input, 0.37);
    slice_type::slice_data data_b;
    data_b.mode = 3;
    const auto stabilized_b = slice.stabilize(shifted, data_b);
    require_vector_near("shifted copies stabilize to same representative", stabilized_b, stabilized_a, 1e-13);

    slice_type::slice_data data_c;
    data_c.mode = 3;
    const auto stabilized_c = slice.stabilize(stabilized_a, data_c);
    require_vector_near("stabilization idempotence", stabilized_c, stabilized_a, 1e-13);
}

void test_projector_identities()
{
    const slice_type slice(1e-12);
    const auto input = sample_spectrum();

    slice_type::slice_data data;
    data.mode = 3;
    const auto state = slice.stabilize(input, data);
    const auto generator = slice.translation_generator(state);

    std::vector<complex_type> v = {
        complex_type(0.0, 0.0),
        complex_type(1.0, -2.0),
        complex_type(-0.25, 0.75),
        complex_type(4.0, 6.0),
        complex_type(0.5, -1.5),
        complex_type(-2.0, 1.0),
        complex_type(0.0, 0.0),
        complex_type(0.0, 0.0)
    };

    slice_type::projection_info info;
    const auto projected = slice.project(data, generator, v, info);
    require(info.ok(), "projector solve status");
    require_near("projector phase removed", slice.phase_value(data, projected.data(), projected.size()), 0.0, 1e-13);
    require_near("projector alpha", info.alpha, 6.0/(3.0*std::sqrt(5.0)), 1e-14);

    slice_type::projection_info info_twice;
    const auto projected_twice = slice.project(data, generator, projected, info_twice);
    require(info_twice.ok(), "projector idempotence solve status");
    require_near("projector idempotence alpha", info_twice.alpha, 0.0, 1e-13);
    require_vector_near("projector idempotence", projected_twice, projected, 1e-13);

    slice_type::projection_info info_generator;
    const auto projected_generator = slice.project(data, generator, generator, info_generator);
    require(info_generator.ok(), "project generator solve status");
    for(std::size_t i = 0; i < projected_generator.size(); ++i)
        require_complex_near("P T = 0 " + std::to_string(i), projected_generator[i], complex_type(0.0, 0.0), 1e-13);
}

void test_inactive_slice()
{
    const slice_type slice(1e-12);
    const std::vector<complex_type> zero(6, complex_type(0.0, 0.0));

    slice_type::slice_data data;
    const auto stabilized = slice.stabilize(zero, data);
    require(!data.active(), "inactive zero spectrum");
    require(data.active_rank == 0, "inactive rank zero");
    require(data.residual_group_order() == 1, "inactive residual group order");
    require_vector_near("inactive stabilization", stabilized, zero, 1e-15);

    const std::vector<complex_type> v = {
        complex_type(0.0, 0.0),
        complex_type(1.0, 2.0),
        complex_type(3.0, -4.0),
        complex_type(-1.0, 0.5),
        complex_type(0.25, 0.75),
        complex_type(0.0, 0.0)
    };
    const std::vector<complex_type> generator(v.size(), complex_type(0.0, 0.0));
    slice_type::projection_info info;
    const auto projected = slice.project(data, generator, v, info);
    require(info.ok(), "inactive projection status");
    require_near("inactive alpha", info.alpha, 0.0, 1e-15);
    require_vector_near("inactive projection identity", projected, v, 1e-15);
}

} // namespace

int main()
{
    test_stabilization();
    test_shift_invariance_and_idempotence();
    test_projector_identities();
    test_inactive_slice();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
