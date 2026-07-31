#include <cmath>
#include <complex>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nmfd/operations/io/matrix_market.h>

namespace
{

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

template<class Scalar>
void require_close(
    const Scalar& actual,
    const Scalar& expected,
    double tolerance,
    const std::string& message)
{
    if(std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

template<class Callable>
void require_throws(Callable&& callable, const std::string& message)
{
    bool threw = false;
    try
    {
        callable();
    }
    catch(const std::exception&)
    {
        threw = true;
    }
    require(threw, message);
}

void test_general_real_and_duplicates()
{
    std::istringstream input(
        "%%MatrixMarket matrix coordinate real general\n"
        "% duplicate coordinates must be combined\n"
        "3 3 5\n"
        "1 1 2\n"
        "1 2 3\n"
        "1 2 -1\n"
        "2 3 4\n"
        "3 1 -2\n");

    const auto coordinate =
        nmfd::operations::io::read_matrix_market<double>(input);
    require(coordinate.metadata.rows == 3, "general: wrong row count");
    require(coordinate.metadata.columns == 3, "general: wrong column count");
    require(
        coordinate.metadata.stored_entries == 5,
        "general: wrong stored entry count");

    const auto csr = coordinate.to_host_csr();
    require(csr.nonzeros() == 4, "general: duplicate was not combined");
    const auto result = csr.apply(std::vector<double>{1.0, 2.0, 3.0});
    require_close(result[0], 6.0, 1.0e-14, "general: row 0 mismatch");
    require_close(result[1], 12.0, 1.0e-14, "general: row 1 mismatch");
    require_close(result[2], -2.0, 1.0e-14, "general: row 2 mismatch");
}

void test_symmetric_pattern()
{
    std::istringstream input(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "3 3 3\n"
        "1 1\n"
        "2 1\n"
        "3 2\n");

    const auto coordinate =
        nmfd::operations::io::read_matrix_market<double>(input);
    require(
        coordinate.entries.size() == 5,
        "symmetric pattern: reflected entries are missing");
    const auto result =
        coordinate.to_host_csr().apply(
            std::vector<double>{1.0, 2.0, 3.0});
    require_close(result[0], 3.0, 1.0e-14, "symmetric: row 0 mismatch");
    require_close(result[1], 4.0, 1.0e-14, "symmetric: row 1 mismatch");
    require_close(result[2], 2.0, 1.0e-14, "symmetric: row 2 mismatch");
}

void test_skew_symmetric_integer()
{
    std::istringstream input(
        "%%MatrixMarket matrix coordinate integer skew-symmetric\n"
        "3 3 3\n"
        "1 2 2\n"
        "1 3 -1\n"
        "2 3 4\n");

    const auto result =
        nmfd::operations::io::read_matrix_market<double>(input)
            .to_host_csr()
            .apply(std::vector<double>{1.0, 2.0, 3.0});
    require_close(result[0], 1.0, 1.0e-14, "skew: row 0 mismatch");
    require_close(result[1], 10.0, 1.0e-14, "skew: row 1 mismatch");
    require_close(result[2], -7.0, 1.0e-14, "skew: row 2 mismatch");
}

void test_hermitian_complex()
{
    using complex_type = std::complex<double>;
    std::istringstream input(
        "%%MatrixMarket matrix coordinate complex hermitian\n"
        "2 2 3\n"
        "1 1 2 0\n"
        "2 1 3 -4\n"
        "2 2 5 0\n");

    const auto result =
        nmfd::operations::io::read_matrix_market<complex_type>(input)
            .to_host_csr()
            .apply(
                std::vector<complex_type>{
                    complex_type(1.0, 0.0),
                    complex_type(0.0, 1.0)});
    require_close(
        result[0],
        complex_type(-2.0, 3.0),
        1.0e-14,
        "Hermitian: row 0 mismatch");
    require_close(
        result[1],
        complex_type(3.0, 1.0),
        1.0e-14,
        "Hermitian: row 1 mismatch");
}

void test_rejections()
{
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix array real general\n"
                "1 1\n"
                "1\n");
            (void)nmfd::operations::io::read_matrix_market<double>(input);
        },
        "array format was accepted");
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix coordinate complex general\n"
                "1 1 1\n"
                "1 1 1 2\n");
            (void)nmfd::operations::io::read_matrix_market<double>(input);
        },
        "complex matrix was accepted into real storage");
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix coordinate real general\n"
                "2 2 1\n"
                "3 1 1\n");
            (void)nmfd::operations::io::read_matrix_market<double>(input);
        },
        "out-of-range coordinate was accepted");
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix coordinate integer skew-symmetric\n"
                "2 2 1\n"
                "1 1 1\n");
            (void)nmfd::operations::io::read_matrix_market<double>(input);
        },
        "nonzero skew diagonal was accepted");
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix coordinate complex hermitian\n"
                "1 1 1\n"
                "1 1 1 2\n");
            (void)nmfd::operations::io::read_matrix_market<
                std::complex<double>>(input);
        },
        "non-real Hermitian diagonal was accepted");
    require_throws(
        []
        {
            std::istringstream input(
                "%%MatrixMarket matrix coordinate real general\n"
                "1 1 1\n"
                "1 1 1\n"
                "1 1 2\n");
            (void)nmfd::operations::io::read_matrix_market<double>(input);
        },
        "undeclared trailing coordinate was accepted");
}

} // namespace

int main()
{
    try
    {
        test_general_real_and_duplicates();
        test_symmetric_pattern();
        test_skew_symmetric_integer();
        test_hermitian_complex();
        test_rejections();
        std::cout << "Matrix Market parser/CSR tests: PASSED\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Matrix Market parser/CSR tests: FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
