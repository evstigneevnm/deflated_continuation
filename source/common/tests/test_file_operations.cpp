#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <common/cpu_file_operations.h>
#include <common/cpu_matrix_file_operations.h>
#include <common/file_operations.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <common/macros.h>
#include <common/scfd_serial_cpu_vector_operations.h>

#include <nmfd/operations/io/matrix_file_operations.h>
#include <nmfd/operations/io/vector_file_operations.h>

namespace
{

struct test_report
{
    std::size_t checks = 0;
    std::size_t failures = 0;

    void require(bool condition, const std::string& label)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cout << "FAIL " << label << std::endl;
        }
    }
};

template<class T>
void require_near(test_report& report, const std::string& label, T got, T expected, T tolerance = T(1.0e-12))
{
    ++report.checks;
    if(std::abs(got - expected) > tolerance)
    {
        ++report.failures;
        std::cout << "FAIL " << label << " got=" << got << " expected=" << expected << std::endl;
    }
}

template<class F>
void expect_throw(test_report& report, const std::string& label, F&& function)
{
    ++report.checks;
    try
    {
        function();
    }
    catch(const std::exception&)
    {
        return;
    }
    ++report.failures;
    std::cout << "FAIL " << label << " did not throw" << std::endl;
}

struct matrix_ops_stub
{
    using scalar_type = double;
    using vector_type = std::vector<double>;
    using matrix_type = std::vector<double>;

    std::size_t get_rows() const
    {
        return 2;
    }

    std::size_t get_cols() const
    {
        return 2;
    }
};

struct matrix_ops_view_stub
{
    using scalar_type = double;
    using vector_type = std::vector<double>;
    using matrix_type = std::vector<double>;

    std::size_t get_rows() const
    {
        return 2;
    }

    std::size_t get_cols() const
    {
        return 2;
    }

    matrix_type& view(matrix_type& matrix) const
    {
        return matrix;
    }

    const matrix_type& view(const matrix_type& matrix) const
    {
        return matrix;
    }

    void set(matrix_type&) const
    {
        ++set_calls;
    }

    mutable std::size_t set_calls = 0;
};

std::filesystem::path make_test_dir()
{
    const auto dir = std::filesystem::temp_directory_path() / "linsolver_file_operations_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void test_generic_vector_io(test_report& report, const std::filesystem::path& dir)
{
    const std::vector<double> input{1.25, -2.5, 3.75};
    std::vector<double> output(input.size(), 0.0);
    const auto file = dir / "vector.dat";

    file_operations::write_vector<double, std::vector<double>>(file.string(), input.size(), input, 17);
    report.require(file_operations::read_vector_size(file.string()) == input.size(), "read_vector_size");
    file_operations::read_vector<double, std::vector<double>>(file.string(), output.size(), output);

    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "generic vector round trip " + std::to_string(i), output[i], input[i]);
    }

    const auto empty_file = dir / "empty_vector.dat";
    file_operations::write_vector<double>(empty_file.string(), 0, input.data(), 17);
    report.require(file_operations::read_vector_size(empty_file.string()) == 0, "zero-length vector write");
}

void test_two_vector_and_matrix_io(test_report& report, const std::filesystem::path& dir)
{
    const std::vector<double> left{1.0, 2.0, 3.0};
    const std::vector<double> right{4.0, 5.0, 6.0};
    const auto side_file = dir / "two_vectors.dat";
    file_operations::write_2_vectors_by_side<double, std::vector<double>>(side_file.string(), left.size(), left, right, 17);
    auto side_size = file_operations::read_matrix_size(side_file.string());
    report.require(side_size.first == 3 && side_size.second == 2, "two-vector matrix size");

    std::vector<double> side_matrix(6, 0.0);
    file_operations::read_matrix<double, std::vector<double>>(side_file.string(), 3, 2, side_matrix);
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        require_near(report, "two-vector left column " + std::to_string(i), side_matrix[I2_R(i, 0, 3)], left[i]);
        require_near(report, "two-vector right column " + std::to_string(i), side_matrix[I2_R(i, 1, 3)], right[i]);
    }

    std::vector<double> square{1.0, 2.0, 3.0, 4.0};
    std::vector<double> square_read(4, 0.0);
    const auto square_file = dir / "square_matrix.dat";
    file_operations::write_matrix<std::vector<double>>(square_file.string(), 2, 2, square, 17);
    report.require(file_operations::read_matrix_size(square_file.string()) == std::pair<std::size_t, std::size_t>{2, 2}, "square matrix size");
    report.require(file_operations::read_matrix_size_square(square_file.string()) == 2, "square matrix square size");
    file_operations::read_matrix<double, std::vector<double>>(square_file.string(), 2, 2, square_read);
    for(std::size_t i = 0; i < square.size(); ++i)
    {
        require_near(report, "square matrix round trip " + std::to_string(i), square_read[i], square[i]);
    }

    std::vector<double> rectangular{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto rectangular_file = dir / "rectangular_matrix.dat";
    file_operations::write_matrix<std::vector<double>>(rectangular_file.string(), 2, 3, rectangular, 17);
    report.require(file_operations::read_matrix_size(rectangular_file.string()) == std::pair<std::size_t, std::size_t>{2, 3}, "rectangular matrix size");
    expect_throw(report, "rectangular matrix square size rejects", [&]()
    {
        (void)file_operations::read_matrix_size_square(rectangular_file.string());
    });
}

void test_match_file_names(test_report& report, const std::filesystem::path& dir)
{
    {
        std::ofstream(dir / "branch_1.dat") << "1\n";
        std::ofstream(dir / "branch_0.dat") << "0\n";
        std::ofstream(dir / "ignore.txt") << "x\n";
    }

    const auto matches = file_operations::match_file_names(dir.string(), "branch_[0-9]+\\.dat");
    report.require(matches.size() == 2, "match_file_names size");
    if(matches.size() == 2)
    {
        report.require(matches[0].find("branch_0.dat") != std::string::npos, "match_file_names sort first");
        report.require(matches[1].find("branch_1.dat") != std::string::npos, "match_file_names sort second");
    }
}

void test_wrappers_with_scfd_serial(test_report& report, const std::filesystem::path& dir)
{
    using vec_ops_t = scfd_serial_cpu_vector_operations<double>;
    using vec_t = typename vec_ops_t::vector_type;

    vec_ops_t vec_ops(3);
    nmfd::operations::io::vector_file_operations<vec_ops_t> vector_files(&vec_ops);
    cpu_file_operations<vec_ops_t> cpu_vector_files(&vec_ops);
    gpu_file_operations<vec_ops_t> gpu_vector_files(&vec_ops);

    vec_t x;
    vec_t y;
    vec_t z;
    vec_ops.init_vector(x);
    vec_ops.init_vector(y);
    vec_ops.init_vector(z);
    vec_ops.start_use_vector(x);
    vec_ops.start_use_vector(y);
    vec_ops.start_use_vector(z);

    const std::vector<double> input{7.0, 8.0, 9.0};
    const std::vector<double> second_input{-1.0, -2.0, -3.0};
    std::vector<double> output(3, 0.0);
    vec_ops.set(input.data(), x);
    vec_ops.set(second_input.data(), z);

    const auto vector_file = dir / "scfd_vector.dat";
    vector_files.write_vector(vector_file.string(), x);
    vector_files.read_vector(vector_file.string(), y);
    vec_ops.get(y, output.data());

    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "vector_file_operations SCFD round trip " + std::to_string(i), output[i], input[i]);
    }

    const auto alias_file = dir / "scfd_vector_alias.dat";
    cpu_vector_files.write_vector(alias_file.string(), x);
    gpu_vector_files.read_vector(alias_file.string(), y);
    vec_ops.get(y, output.data());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "file operation aliases SCFD round trip " + std::to_string(i), output[i], input[i]);
    }

    const auto side_file = dir / "scfd_two_vectors.dat";
    vector_files.write_2_vectors_by_side(side_file.string(), x, z, 17);
    std::vector<double> side_matrix(6, 0.0);
    file_operations::read_matrix<double, std::vector<double>>(side_file.string(), 3, 2, side_matrix);
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "SCFD two-vector first column " + std::to_string(i), side_matrix[I2_R(i, 0, 3)], input[i]);
        require_near(report, "SCFD two-vector second column " + std::to_string(i), side_matrix[I2_R(i, 1, 3)], second_input[i]);
    }

    vec_ops.stop_use_vector(x);
    vec_ops.stop_use_vector(y);
    vec_ops.stop_use_vector(z);
    vec_ops.free_vector(x);
    vec_ops.free_vector(y);
    vec_ops.free_vector(z);

    matrix_ops_stub matrix_ops;
    nmfd::operations::io::matrix_file_operations<matrix_ops_stub> matrix_files(&matrix_ops);
    cpu_matrix_file_operations<matrix_ops_stub> cpu_matrix_files(&matrix_ops);
    gpu_matrix_file_operations<matrix_ops_stub> gpu_matrix_files(&matrix_ops);
    std::vector<double> matrix{11.0, 12.0, 13.0, 14.0};
    std::vector<double> matrix_read(4, 0.0);
    const auto matrix_file = dir / "cpu_matrix_wrapper.dat";
    matrix_files.write_matrix(matrix_file.string(), matrix, 17);
    matrix_files.read_matrix(matrix_file.string(), matrix_read);
    for(std::size_t i = 0; i < matrix.size(); ++i)
    {
        require_near(report, "matrix_file_operations round trip " + std::to_string(i), matrix_read[i], matrix[i]);
    }

    const auto matrix_alias_file = dir / "matrix_alias.dat";
    cpu_matrix_files.write_matrix(matrix_alias_file.string(), matrix, 17);
    gpu_matrix_files.read_matrix(matrix_alias_file.string(), matrix_read);
    for(std::size_t i = 0; i < matrix.size(); ++i)
    {
        require_near(report, "matrix file operation aliases round trip " + std::to_string(i), matrix_read[i], matrix[i]);
    }

    matrix_ops_view_stub matrix_ops_with_view;
    nmfd::operations::io::matrix_file_operations<matrix_ops_view_stub> matrix_files_with_view(&matrix_ops_with_view);
    const auto matrix_view_file = dir / "matrix_view.dat";
    matrix_read.assign(4, 0.0);
    matrix_files_with_view.write_matrix(matrix_view_file.string(), matrix, 17);
    matrix_files_with_view.read_matrix(matrix_view_file.string(), matrix_read);
    report.require(matrix_ops_with_view.set_calls == 1, "matrix_file_operations syncs mutable view");
    for(std::size_t i = 0; i < matrix.size(); ++i)
    {
        require_near(report, "matrix_file_operations view round trip " + std::to_string(i), matrix_read[i], matrix[i]);
    }
}

} // namespace

int main()
{
    test_report report;
    const auto dir = make_test_dir();

    try
    {
        test_generic_vector_io(report, dir);
        test_two_vector_and_matrix_io(report, dir);
        test_match_file_names(report, dir);
        test_wrappers_with_scfd_serial(report, dir);
    }
    catch(const std::exception& e)
    {
        ++report.failures;
        std::cout << "FAIL unexpected exception: " << e.what() << std::endl;
    }

    std::filesystem::remove_all(dir);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
