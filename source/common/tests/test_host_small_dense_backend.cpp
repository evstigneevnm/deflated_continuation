#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include <nmfd/operations/linalg/host_small_dense_lapack.h>

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

template<class T>
void require_near(
    T value,
    T expected,
    T tolerance,
    const std::string& message)
{
    require(
        std::abs(value - expected) <= tolerance,
        message + " value=" + std::to_string(value) +
            " expected=" + std::to_string(expected));
}

template<class Real>
void test_backend(const std::string& label, Real tolerance)
{
    using backend_type =
        nmfd::operations::linalg::host_small_dense_backend<std::ptrdiff_t, Real>;
    using matrix_type = typename backend_type::matrix_type;
    using vector_type = typename backend_type::vector_type;

    static_assert(
        nmfd::operations::linalg::is_host_small_dense_backend<
            backend_type>::value,
        "backend concept check failed");

    backend_type backend(3, 2);
    matrix_type hessenberg;
    vector_type rhs;
    vector_type solution;
    vector_type cosines;
    vector_type sines;
    backend.init_matrix(hessenberg);
    backend.init_col_vectors(rhs, solution, cosines, sines);

    backend.assign_scalar_matrix(Real{}, hessenberg);
    backend.assign_scalar_col_vector(Real{}, rhs);
    hessenberg(0, 0) = Real(3);
    hessenberg(1, 0) = Real(4);
    rhs(0) = Real(5);
    backend.plane_rotation_col(hessenberg, cosines, sines, rhs, 0);
    require_near(hessenberg(0, 0), Real(5), tolerance, label + " Givens radius");
    require_near(hessenberg(1, 0), Real(0), tolerance, label + " Givens zero");
    require_near(rhs(0), Real(3), tolerance, label + " rotated rhs 0");
    require_near(rhs(1), Real(-4), tolerance, label + " rotated rhs 1");

    matrix_type upper(3, 3);
    upper(0, 0) = Real(2);
    upper(0, 1) = Real(1);
    upper(0, 2) = Real(-1);
    upper(1, 1) = Real(3);
    upper(1, 2) = Real(2);
    upper(2, 2) = Real(4);
    vector_type expected(3);
    expected(0) = Real(1);
    expected(1) = Real(-2);
    expected(2) = Real(3);
    vector_type upper_rhs(3);
    for(std::size_t row = 0; row < 3; ++row)
    {
        upper_rhs(row) = Real{};
        for(std::size_t col = row; col < 3; ++col)
            upper_rhs(row) += upper(row, col)*expected(col);
    }
    vector_type solved;
    backend.solve_upper_triangular_subsystem(upper, upper_rhs, solved, 3);
    for(std::size_t index = 0; index < 3; ++index)
    {
        require_near(
            solved(index),
            expected(index),
            tolerance,
            label + " triangular solve " + std::to_string(index));
    }

    matrix_type rotation(3, 3);
    rotation(0, 0) = Real(0);
    rotation(1, 0) = Real(2);
    rotation(2, 0) = Real(0);
    rotation(0, 1) = Real(-2);
    rotation(1, 1) = Real(0);
    rotation(2, 1) = Real(0);
    rotation(0, 2) = Real(0);
    rotation(1, 2) = Real(0);
    rotation(2, 2) = Real(-3);

    nmfd::operations::linalg::host_small_dense_lapack<Real> lapack;
    auto eigenvalues = lapack.eigenvalues(rotation);
    std::sort(
        eigenvalues.begin(),
        eigenvalues.end(),
        [](const auto& left, const auto& right)
        {
            if(left.real() != right.real())
                return left.real() < right.real();
            return left.imag() < right.imag();
        });
    require(
        std::abs(eigenvalues[0] - std::complex<Real>(Real(-3), Real(0))) <=
            tolerance*Real(10),
        label + " LAPACK real eigenvalue");
    require(
        std::abs(eigenvalues[1] - std::complex<Real>(Real(0), Real(-2))) <=
            tolerance*Real(10),
        label + " LAPACK negative imaginary eigenvalue");
    require(
        std::abs(eigenvalues[2] - std::complex<Real>(Real(0), Real(2))) <=
            tolerance*Real(10),
        label + " LAPACK positive imaginary eigenvalue");

    const auto schur = lapack.hessenberg_schur(rotation);
    require(
        schur.orthogonal_vectors.rows() == 3 &&
            schur.quasi_triangular.cols() == 3,
        label + " Schur dimensions");

    auto general_schur = lapack.schur(rotation);
    auto blocks = lapack.real_schur_blocks(general_schur);
    require(blocks.size() == 2, label + " real Schur block count");
    std::size_t real_block = blocks.size();
    std::size_t complex_block = blocks.size();
    for(std::size_t index = 0; index < blocks.size(); ++index)
    {
        if(blocks[index].size == 1)
            real_block = index;
        if(blocks[index].size == 2)
            complex_block = index;
    }
    require(
        real_block < blocks.size() && complex_block < blocks.size(),
        label + " real Schur block sizes");
    lapack.move_schur_block(
        general_schur,
        blocks[real_block].first,
        0);
    blocks = lapack.real_schur_blocks(general_schur);
    require(
        blocks.front().size == 1 &&
            std::abs(
                blocks.front().eigenvalues.front() -
                std::complex<Real>(Real(-3), Real{})) <= tolerance*Real(10),
        label + " real Schur block reorder");

    Real reconstruction_error_sq = Real{};
    for(std::size_t row = 0; row < 3; ++row)
    {
        for(std::size_t col = 0; col < 3; ++col)
        {
            Real reconstructed = Real{};
            for(std::size_t left = 0; left < 3; ++left)
            {
                for(std::size_t right = 0; right < 3; ++right)
                {
                    reconstructed +=
                        general_schur.orthogonal_vectors(row, left)*
                        general_schur.quasi_triangular(left, right)*
                        general_schur.orthogonal_vectors(col, right);
                }
            }
            const Real difference = reconstructed - rotation(row, col);
            reconstruction_error_sq += difference*difference;
        }
    }
    require(
        std::sqrt(reconstruction_error_sq) <= tolerance*Real(100),
        label + " reordered Schur reconstruction");
}

template<class Real>
void test_complex_backend(const std::string& label, Real tolerance)
{
    using complex_type = std::complex<Real>;
    using backend_type =
        nmfd::operations::linalg::host_small_dense_backend<
            std::ptrdiff_t,
            complex_type>;
    using matrix_type = typename backend_type::matrix_type;
    using vector_type = typename backend_type::vector_type;

    backend_type backend(3, 2);
    matrix_type hessenberg;
    vector_type rhs;
    vector_type cosines;
    vector_type sines;
    backend.init_matrix(hessenberg);
    backend.init_col_vectors(rhs, cosines, sines);
    backend.assign_scalar_matrix(complex_type{}, hessenberg);
    backend.assign_scalar_col_vector(complex_type{}, rhs);

    const complex_type first(Real(3), Real(4));
    const complex_type second(Real(2), Real(-1));
    const complex_type rhs_first(Real(1), Real(2));
    const complex_type rhs_second(Real(-0.5), Real(0.75));
    hessenberg(0, 0) = first;
    hessenberg(1, 0) = second;
    rhs(0) = rhs_first;
    rhs(1) = rhs_second;
    backend.plane_rotation_col(
        hessenberg,
        cosines,
        sines,
        rhs,
        0);

    const Real radius = std::hypot(
        std::abs(first),
        std::abs(second));
    const complex_type phase = first/std::abs(first);
    const complex_type expected_cosine(
        std::abs(first)/radius,
        Real{});
    const complex_type expected_sine =
        phase*std::conj(second)/radius;
    const complex_type expected_first =
        expected_cosine*first + expected_sine*second;
    const complex_type expected_rhs_first =
        expected_cosine*rhs_first +
        expected_sine*rhs_second;
    const complex_type expected_rhs_second =
        -std::conj(expected_sine)*rhs_first +
        expected_cosine*rhs_second;

    require(
        std::abs(hessenberg(1, 0)) <= tolerance,
        label + " complex Givens zero");
    require(
        std::abs(hessenberg(0, 0) - expected_first) <= tolerance,
        label + " complex Givens first value");
    require(
        std::abs(rhs(0) - expected_rhs_first) <= tolerance,
        label + " complex Givens rhs first");
    require(
        std::abs(rhs(1) - expected_rhs_second) <= tolerance,
        label + " complex Givens rhs second");

    matrix_type upper(3, 3);
    upper(0, 0) = complex_type(Real(2), Real(0.5));
    upper(0, 1) = complex_type(Real(1), Real(-1));
    upper(0, 2) = complex_type(Real(-1), Real(0.25));
    upper(1, 1) = complex_type(Real(3), Real(-0.2));
    upper(1, 2) = complex_type(Real(2), Real(0.5));
    upper(2, 2) = complex_type(Real(4), Real(1));
    vector_type expected(3);
    expected(0) = complex_type(Real(1), Real(-0.5));
    expected(1) = complex_type(Real(-2), Real(0.75));
    expected(2) = complex_type(Real(3), Real(0.2));
    vector_type upper_rhs(3);
    for(std::size_t row = 0; row < 3; ++row)
    {
        upper_rhs(row) = complex_type{};
        for(std::size_t col = row; col < 3; ++col)
            upper_rhs(row) += upper(row, col)*expected(col);
    }
    vector_type solved;
    backend.solve_upper_triangular_subsystem(
        upper,
        upper_rhs,
        solved,
        3);
    for(std::size_t index = 0; index < 3; ++index)
    {
        require(
            std::abs(solved(index) - expected(index)) <= tolerance,
            label + " complex triangular solve " +
                std::to_string(index));
    }
}

} // namespace

int main()
{
    test_backend<float>("float", 1e-5f);
    test_backend<double>("double", 1e-12);
    test_complex_backend<float>("complex<float>", 2e-5f);
    test_complex_backend<double>("complex<double>", 2e-12);

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
