#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nmfd/operations/linalg/small_dense.h>
#include <symmetry/fourier/mode_traits.h>

namespace
{

int checks = 0;
int failures = 0;

void require(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        std::cout << "FAIL " << label << std::endl;
        ++failures;
    }
}

template<class T>
void require_near(const std::string& label, T value, T expected, T tol)
{
    ++checks;
    const T err = std::abs(value - expected);
    if(!(err <= tol))
    {
        std::cout << "FAIL " << label << " value=" << value
                  << " expected=" << expected << " err=" << err
                  << " tol=" << tol << std::endl;
        ++failures;
    }
}

void test_mode_dot()
{
    using symmetry::fourier::mode_index;
    using symmetry::fourier::translation_direction;
    using symmetry::fourier::wave_dot;

    mode_index<3> mode{2, -3, 1};
    translation_direction<double, 3> direction{0.5, 2.0, -4.0};
    require_near("wave_dot 3D", wave_dot(mode, direction), -9.0, 1e-14);

    require_near(
        "slice_matrix_entry",
        symmetry::fourier::slice_matrix_entry(wave_dot(mode, direction), 2.5),
        -22.5,
        1e-14);
}

void test_vector_phase_functional()
{
    using complex_type = std::complex<double>;
    using functional_type = symmetry::fourier::vector_valued_phase_functional<complex_type, 3>;

    functional_type functional{
        complex_type(1.0, 2.0),
        complex_type(0.5, -1.0)
    };

    complex_type coeffs[2] = {
        complex_type(3.0, -4.0),
        complex_type(2.0, 5.0)
    };

    const complex_type expected = std::conj(complex_type(1.0, 2.0))*coeffs[0]
                                + std::conj(complex_type(0.5, -1.0))*coeffs[1];
    const complex_type value = functional.evaluate(coeffs);

    require(functional.size() == 2, "functional size");
    require_near("functional real", value.real(), expected.real(), 1e-14);
    require_near("functional imag", value.imag(), expected.imag(), 1e-14);
    require_near("phase value", functional.phase_value(coeffs), expected.imag(), 1e-14);
    require_near("real value", functional.real_value(coeffs), expected.real(), 1e-14);

    functional.resize(1);
    functional.set_eta(0, complex_type(0.0, 1.0));
    const complex_type coeff1[1] = {complex_type(2.0, 3.0)};
    const complex_type expected1 = std::conj(complex_type(0.0, 1.0))*coeff1[0];
    require_near("reset functional real", functional.real_value(coeff1), expected1.real(), 1e-14);
    require_near("reset functional imag", functional.phase_value(coeff1), expected1.imag(), 1e-14);
}

void test_active_dimension_less_than_group_dimension()
{
    using symmetry::fourier::mode_index;
    using symmetry::fourier::slice_matrix_entry;
    using symmetry::fourier::translation_direction;
    using symmetry::fourier::wave_dot;
    using namespace nmfd::operations::linalg;

    // This mirrors the plane-parallel case from the paper: the full
    // translation group is two-dimensional, but the state depends only on x.
    // Therefore z-translation has zero tangent and the active rank is rho=1.
    mode_index<2> q1{3, 0};
    mode_index<2> q2{6, 0};
    translation_direction<double, 2> r_x{1.0, 0.0};
    translation_direction<double, 2> r_z{0.0, 1.0};

    const double q1_dot_x = wave_dot(q1, r_x);
    const double q1_dot_z = wave_dot(q1, r_z);
    const double q2_dot_x = wave_dot(q2, r_x);
    const double q2_dot_z = wave_dot(q2, r_z);

    require_near("rho<d q1 dot x", q1_dot_x, 3.0, 1e-14);
    require_near("rho<d q1 dot z", q1_dot_z, 0.0, 1e-14);
    require_near("rho<d q2 dot x", q2_dot_x, 6.0, 1e-14);
    require_near("rho<d q2 dot z", q2_dot_z, 0.0, 1e-14);

    small_matrix<double, 2> full_candidate{
        {slice_matrix_entry(q1_dot_x, 2.0), slice_matrix_entry(q1_dot_z, 2.0)},
        {slice_matrix_entry(q2_dot_x, 1.5), slice_matrix_entry(q2_dot_z, 1.5)}
    };
    small_vector<double, 2> full_rhs{1.0, 1.0};
    small_vector<double, 2> full_solution;
    const auto full_info = solve(full_candidate, full_rhs, full_solution);

    require(
        full_info.status == small_solve_status::singular,
        "rho<d full d-dimensional candidate is singular");
    require(full_info.rank == 1, "rho<d full candidate rank is one");

    small_matrix<double, 1> active_slice{
        {slice_matrix_entry(q1_dot_x, 2.0)}
    };
    small_vector<double, 1> active_rhs{12.0};
    small_vector<double, 1> active_alpha;
    const auto active_info = solve(active_slice, active_rhs, active_alpha);

    require(
        active_info.status == small_solve_status::success,
        "rho<d active one-dimensional slice solves");
    require_near("rho<d active alpha", active_alpha[0], 2.0, 1e-14);
}

void test_dimension_errors()
{
    bool mode_threw = false;
    try
    {
        symmetry::fourier::mode_index<2> bad_mode{1, 2, 3};
        (void)bad_mode;
    }
    catch(const std::invalid_argument&)
    {
        mode_threw = true;
    }
    require(mode_threw, "mode_index rejects wrong dimension");

    bool direction_threw = false;
    try
    {
        symmetry::fourier::translation_direction<double, 2> bad_direction{1.0};
        (void)bad_direction;
    }
    catch(const std::invalid_argument&)
    {
        direction_threw = true;
    }
    require(direction_threw, "translation_direction rejects wrong dimension");

    bool functional_threw = false;
    try
    {
        symmetry::fourier::vector_valued_phase_functional<std::complex<double>, 1> bad_functional{
            std::complex<double>(1.0, 0.0),
            std::complex<double>(0.0, 1.0)
        };
        (void)bad_functional;
    }
    catch(const std::out_of_range&)
    {
        functional_threw = true;
    }
    require(functional_threw, "functional rejects too many components");
}

} // namespace

int main()
{
    test_mode_dot();
    test_vector_phase_functional();
    test_active_dimension_less_than_group_dimension();
    test_dimension_errors();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
