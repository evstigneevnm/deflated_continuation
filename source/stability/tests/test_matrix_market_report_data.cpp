#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/backend/omp.h>

#include <common/scfd_vector_operations.h>
#include <nmfd/operations/io/matrix_market.h>
#include <nmfd/operations/sparse/host_csr_matrix.h>

#include "common/host_csr_operator.h"

namespace
{

using matrix_type =
    nmfd::operations::sparse::host_csr_matrix<double>;

struct fingerprint
{
    long double sum = 0.0L;
    long double norm = 0.0L;
    long double weighted_sum = 0.0L;
};

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

void require_close(
    long double actual,
    long double expected,
    long double absolute_tolerance,
    long double relative_tolerance,
    const std::string& message)
{
    const long double error = std::abs(actual - expected);
    const long double tolerance =
        absolute_tolerance +
        relative_tolerance * std::max(std::abs(actual), std::abs(expected));
    if(error > tolerance)
    {
        throw std::runtime_error(
            message + ": actual=" +
            std::to_string(static_cast<double>(actual)) +
            ", expected=" +
            std::to_string(static_cast<double>(expected)));
    }
}

std::vector<double> test_vector(std::size_t dimension)
{
    std::vector<double> result(dimension);
    for(std::size_t index = 0; index < dimension; ++index)
    {
        result[index] =
            static_cast<double>(
                static_cast<int>(index % 17) - 8) / 8.0;
    }
    return result;
}

fingerprint calculate_fingerprint(const std::vector<double>& values)
{
    fingerprint result;
    long double norm_squared = 0.0L;
    for(std::size_t index = 0; index < values.size(); ++index)
    {
        const long double value = values[index];
        result.sum += value;
        norm_squared += value * value;
        result.weighted_sum +=
            static_cast<long double>(index + 1) * value;
    }
    result.norm = std::sqrt(norm_squared);
    return result;
}

void check_fingerprint(
    const std::string& name,
    const matrix_type& matrix,
    const fingerprint& expected)
{
    const auto source =
        test_vector(static_cast<std::size_t>(matrix.columns()));
    const auto serial = matrix.apply(
        source,
        nmfd::operations::sparse::host_csr_execution::serial);
    const auto parallel = matrix.apply(
        source,
        nmfd::operations::sparse::host_csr_execution::openmp);
    require(serial == parallel, name + ": serial/OMP matvec mismatch");

    const auto actual = calculate_fingerprint(serial);
    require_close(
        actual.sum,
        expected.sum,
        1.0e-12L,
        1.0e-12L,
        name + ": sum fingerprint");
    require_close(
        actual.norm,
        expected.norm,
        1.0e-12L,
        1.0e-12L,
        name + ": norm fingerprint");
    require_close(
        actual.weighted_sum,
        expected.weighted_sum,
        1.0e-10L,
        1.0e-12L,
        name + ": weighted fingerprint");
}

void check_scfd_adapter(const matrix_type& matrix)
{
    using vector_space_type =
        scfd_vector_operations<scfd::backend::omp, double>;
    using vector_type = typename vector_space_type::vector_type;
    const std::size_t dimension =
        static_cast<std::size_t>(matrix.rows());
    vector_space_type vector_space(dimension);
    stability::tests::host_csr_operator<vector_space_type> matrix_operator(
        vector_space,
        matrix,
        nmfd::operations::sparse::host_csr_execution::openmp);

    vector_type source;
    vector_type destination;
    vector_space.init_vector(source);
    vector_space.init_vector(destination);
    vector_space.start_use_vector(source);
    vector_space.start_use_vector(destination);

    const auto host_source = test_vector(dimension);
    vector_space.set(host_source.data(), source, dimension);
    require(
        matrix_operator.apply(source, destination),
        "SCFD host CSR adapter reported failure");
    std::vector<double> actual(dimension);
    vector_space.get(destination, actual.data(), dimension);
    const auto expected = matrix.apply(
        host_source,
        nmfd::operations::sparse::host_csr_execution::serial);
    require(actual == expected, "SCFD host CSR adapter result mismatch");
    require(
        matrix_operator.operator_calls() == 1,
        "SCFD host CSR adapter call accounting mismatch");

    vector_space.stop_use_vector(destination);
    vector_space.stop_use_vector(source);
    vector_space.free_vector(destination);
    vector_space.free_vector(source);
}

std::string join(const std::string& root, const std::string& relative)
{
    if(root.empty() || root.back() == '/')
        return root + relative;
    return root + "/" + relative;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string root =
            argc > 1 ? argv[1] : "data/external/suitesparse";
        const auto rdb_coordinate =
            nmfd::operations::io::read_matrix_market_file<double>(
                join(root, "Bai/rdb450/rdb450.mtx"));
        const auto s80_coordinate =
            nmfd::operations::io::read_matrix_market_file<double>(
                join(root, "Rommes/S80PI_n1/S80PI_n1.mtx"));
        const auto s80_e_coordinate =
            nmfd::operations::io::read_matrix_market_file<double>(
                join(root, "Rommes/S80PI_n1/S80PI_n1_E.mtx"));

        require(
            rdb_coordinate.metadata.rows == 450 &&
            rdb_coordinate.metadata.columns == 450 &&
            rdb_coordinate.metadata.stored_entries == 2580,
            "rdb450 metadata mismatch");
        require(
            s80_coordinate.metadata.rows == 4028 &&
            s80_coordinate.metadata.columns == 4028 &&
            s80_coordinate.metadata.stored_entries == 9927,
            "S80PI_n1 metadata mismatch");
        require(
            s80_e_coordinate.metadata.rows == 4028 &&
            s80_e_coordinate.metadata.columns == 4028 &&
            s80_e_coordinate.metadata.stored_entries == 4028,
            "S80PI_n1 E metadata mismatch");

        const auto rdb = rdb_coordinate.to_host_csr();
        const auto s80 = s80_coordinate.to_host_csr();
        const auto s80_e = s80_e_coordinate.to_host_csr();
        require(rdb.nonzeros() == 2580, "rdb450 CSR nonzero mismatch");
        require(s80.nonzeros() == 9927, "S80PI_n1 CSR nonzero mismatch");
        require(s80_e.nonzeros() == 4028, "S80PI_n1 E CSR nonzero mismatch");

        check_fingerprint(
            "rdb450",
            rdb,
            fingerprint{
                2.75880000000000649e+01L,
                3.64700784049061212e+02L,
                2.17685000000008586e+02L});
        check_fingerprint(
            "S80PI_n1",
            s80,
            fingerprint{
                1.50785348015400444e-01L,
                4.36550571451883940e+01L,
                -1.22651284845528698e+04L});
        check_fingerprint(
            "S80PI_n1_E",
            s80_e,
            fingerprint{
                1.26805052782027819e-02L,
                1.63226432832003499e-02L,
                3.89595398576324428e+01L});

        for(std::size_t row = 0; row < 4028; ++row)
        {
            require(
                s80_e.row_offsets()[row + 1] -
                    s80_e.row_offsets()[row] == 1,
                "S80PI_n1 E is not diagonal");
            require(
                s80_e.column_indices()[s80_e.row_offsets()[row]] == row,
                "S80PI_n1 E diagonal index mismatch");
            require(
                s80_e.values()[s80_e.row_offsets()[row]] > 0.0,
                "S80PI_n1 E is not positive");
        }

        check_scfd_adapter(rdb);
        check_scfd_adapter(s80);
        std::cout << "Matrix Market report data tests: PASSED\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr << "Matrix Market report data tests: FAILED: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
