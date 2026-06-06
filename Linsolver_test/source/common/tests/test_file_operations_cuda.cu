#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scfd/backend/cuda.h>
#include <scfd/utils/device_tag.h>

#include <common/cuda_init_scfd.h>
#include <common/file_operations.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <common/macros.h>
#include <common/scfd_backend_ext/host_view.h>
#include <common/scfd_vector_operations.h>

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
void require_near(test_report& report, const std::string& label, T got, T expected, T tolerance = T(1.0e-10))
{
    ++report.checks;
    if(std::abs(got - expected) > tolerance)
    {
        ++report.failures;
        std::cout << "FAIL " << label << " got=" << got << " expected=" << expected << std::endl;
    }
}

std::filesystem::path make_test_dir()
{
    const auto dir = std::filesystem::temp_directory_path() / "linsolver_file_operations_cuda_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

template<class T>
class cuda_flat_matrix_operations
{
public:
    using backend_type = scfd::backend::cuda;
    using scalar_type = T;
    using ordinal_type = int;
    using matrix_type = scfd::arrays::array<scalar_type, typename backend_type::memory_type>;
    using vector_type = matrix_type;
    using host_view_traits = common::scfd_backend_ext::host_view_traits<matrix_type>;
    using matrix_host_view_type = typename host_view_traits::view_type;
    using copy_type = typename backend_type::copy_type;
    using for_each_type = typename backend_type::template for_each_type<ordinal_type>;

    cuda_flat_matrix_operations(std::size_t rows_, std::size_t cols_):
        rows(rows_),
        cols(cols_)
    {
    }

    ~cuda_flat_matrix_operations()
    {
        release_host_view(false);
    }

    std::size_t get_rows() const
    {
        return rows;
    }

    std::size_t get_cols() const
    {
        return cols;
    }

    void init_matrix(matrix_type&) const
    {
    }

    void start_use_matrix(matrix_type& matrix) const
    {
        if(matrix.is_free())
        {
            matrix.init(static_cast<typename matrix_type::ordinal_type>(rows*cols));
        }
    }

    void stop_use_matrix(matrix_type&) const
    {
    }

    void free_matrix(matrix_type& matrix) const
    {
        if(host_view_active_ && active_host_view_array_ptr_ == matrix.raw_ptr())
        {
            release_host_view(false);
        }
        if(!matrix.is_free())
        {
            if(matrix.is_own())
            {
                matrix.free();
            }
            else
            {
                matrix = matrix_type();
            }
        }
    }

    scalar_type* view(matrix_type& matrix) const
    {
        open_host_view(matrix, true);
        return active_host_view_.raw_ptr();
    }

    const scalar_type* view(const matrix_type& matrix) const
    {
        open_host_view(matrix, true);
        return active_host_view_.raw_ptr();
    }

    void set(matrix_type& matrix) const
    {
        if(host_view_active_ && active_host_view_array_ptr_ == matrix.raw_ptr())
        {
            active_host_view_.sync_to_array();
            release_host_view(false);
            return;
        }
        throw std::logic_error("cuda_flat_matrix_operations::set requires an active host view");
    }

    void set(const scalar_type* host, matrix_type& matrix) const
    {
        copy_type()(static_cast<ordinal_type>(rows*cols), host, matrix.raw_ptr());
    }

    void get(const matrix_type& matrix, scalar_type* host) const
    {
        copy_type()(static_cast<ordinal_type>(rows*cols), matrix.raw_ptr(), host);
    }

    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_matrix, matrix_type& matrix) const
    {
        auto matrix_p = matrix.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type i)
        {
            matrix_p[i] = mul_matrix*matrix_p[i] + scalar;
        }, static_cast<ordinal_type>(rows*cols));
        for_each_.wait();
    }

private:
    void open_host_view(const matrix_type& matrix, bool sync_from_array) const
    {
        release_host_view(false);
        active_host_view_.init(matrix, sync_from_array);
        host_view_active_ = true;
        active_host_view_array_ptr_ = matrix.raw_ptr();
    }

    void release_host_view(bool sync_to_array) const
    {
        if(!host_view_active_)
        {
            return;
        }
        active_host_view_.release(sync_to_array);
        host_view_active_ = false;
        active_host_view_array_ptr_ = nullptr;
    }

    std::size_t rows;
    std::size_t cols;
    mutable for_each_type for_each_;
    mutable matrix_host_view_type active_host_view_;
    mutable bool host_view_active_ = false;
    mutable const scalar_type* active_host_view_array_ptr_ = nullptr;
};

void test_cuda_vector_file_operations(test_report& report, const std::filesystem::path& dir)
{
    using scalar_type = SCALAR_TYPE;
    using vec_ops_t = scfd_vector_operations<scfd::backend::cuda, scalar_type>;
    using vector_type = typename vec_ops_t::vector_type;

    vec_ops_t vec_ops(7);
    nmfd::operations::io::vector_file_operations<vec_ops_t> direct_files(&vec_ops);
    gpu_file_operations<vec_ops_t> alias_files(&vec_ops);

    vector_type x;
    vector_type y;
    vector_type z;
    vec_ops.init_vector(x);
    vec_ops.init_vector(y);
    vec_ops.init_vector(z);
    vec_ops.start_use_vector(x);
    vec_ops.start_use_vector(y);
    vec_ops.start_use_vector(z);

    const std::vector<scalar_type> input{1.0, -2.0, 3.5, -4.25, 5.5, 6.75, -7.0};
    const std::vector<scalar_type> second_input{-3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0};
    std::vector<scalar_type> expected(input.size(), scalar_type(0));
    std::vector<scalar_type> expected_second(second_input.size(), scalar_type(0));
    std::vector<scalar_type> output(input.size(), scalar_type(0));

    vec_ops.set(input.data(), x);
    vec_ops.add_mul_scalar(scalar_type(3.0), scalar_type(2.0), x);
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        expected[i] = scalar_type(2.0)*input[i] + scalar_type(3.0);
    }

    const auto vector_file = dir / "cuda_vector.dat";
    direct_files.write_vector(vector_file.string(), x, 17);
    alias_files.read_vector(vector_file.string(), y);
    vec_ops.add_mul_scalar(scalar_type(1.0), scalar_type(1.0), y);
    vec_ops.get(y, output.data());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "CUDA vector read/write round trip " + std::to_string(i), output[i], expected[i] + scalar_type(1.0));
    }

    vec_ops.set(second_input.data(), z);
    vec_ops.add_mul_scalar(scalar_type(-2.0), scalar_type(-1.0), z);
    for(std::size_t i = 0; i < second_input.size(); ++i)
    {
        expected_second[i] = -second_input[i] - scalar_type(2.0);
    }

    const auto side_file = dir / "cuda_two_vectors.dat";
    alias_files.write_2_vectors_by_side(side_file.string(), x, z, 17);
    report.require(file_operations::read_matrix_size(side_file.string()) == std::pair<std::size_t, std::size_t>{input.size(), 2}, "CUDA two-vector file size");
    std::vector<scalar_type> side_matrix(input.size()*2, scalar_type(0));
    file_operations::read_matrix<scalar_type, std::vector<scalar_type>>(side_file.string(), input.size(), 2, side_matrix);
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "CUDA two-vector first column " + std::to_string(i), side_matrix[I2_R(i, 0, input.size())], expected[i]);
        require_near(report, "CUDA two-vector second column " + std::to_string(i), side_matrix[I2_R(i, 1, input.size())], expected_second[i]);
    }

    vec_ops.stop_use_vector(x);
    vec_ops.stop_use_vector(y);
    vec_ops.stop_use_vector(z);
    vec_ops.free_vector(x);
    vec_ops.free_vector(y);
    vec_ops.free_vector(z);
}

void test_cuda_matrix_file_operations(test_report& report, const std::filesystem::path& dir)
{
    using scalar_type = SCALAR_TYPE;
    using matrix_ops_t = cuda_flat_matrix_operations<scalar_type>;
    using matrix_type = typename matrix_ops_t::matrix_type;

    matrix_ops_t matrix_ops(2, 3);
    gpu_matrix_file_operations<matrix_ops_t> matrix_files(&matrix_ops);

    matrix_type matrix;
    matrix_type matrix_read;
    matrix_ops.init_matrix(matrix);
    matrix_ops.init_matrix(matrix_read);
    matrix_ops.start_use_matrix(matrix);
    matrix_ops.start_use_matrix(matrix_read);

    std::vector<scalar_type> input(6, scalar_type(0));
    input[I2_R(0, 0, 2)] = scalar_type(1.0);
    input[I2_R(1, 0, 2)] = scalar_type(-2.0);
    input[I2_R(0, 1, 2)] = scalar_type(3.0);
    input[I2_R(1, 1, 2)] = scalar_type(-4.0);
    input[I2_R(0, 2, 2)] = scalar_type(5.0);
    input[I2_R(1, 2, 2)] = scalar_type(-6.0);

    std::vector<scalar_type> expected(input.size(), scalar_type(0));
    matrix_ops.set(input.data(), matrix);
    matrix_ops.add_mul_scalar(scalar_type(5.0), scalar_type(3.0), matrix);
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        expected[i] = scalar_type(3.0)*input[i] + scalar_type(5.0);
    }

    const auto matrix_file = dir / "cuda_matrix.dat";
    matrix_files.write_matrix(matrix_file.string(), matrix, 17);
    report.require(matrix_files.read_matrix_size(matrix_file.string()) == std::pair<std::size_t, std::size_t>{2, 3}, "CUDA matrix file size");
    matrix_files.read_matrix(matrix_file.string(), matrix_read);
    matrix_ops.add_mul_scalar(scalar_type(-1.0), scalar_type(1.0), matrix_read);

    std::vector<scalar_type> output(input.size(), scalar_type(0));
    matrix_ops.get(matrix_read, output.data());
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        require_near(report, "CUDA matrix read/write round trip " + std::to_string(i), output[i], expected[i] - scalar_type(1.0));
    }

    matrix_ops.stop_use_matrix(matrix);
    matrix_ops.stop_use_matrix(matrix_read);
    matrix_ops.free_matrix(matrix);
    matrix_ops.free_matrix(matrix_read);
}

} // namespace

int main(int argc, char** argv)
{
    test_report report;
    const std::string device_selector = (argc > 1) ? argv[1] : "auto";

    try
    {
        const int device = common::init_cuda_from_scfd_selector(device_selector);
        std::cout << "CUDA device: " << device << std::endl;

        const auto dir = make_test_dir();
        test_cuda_vector_file_operations(report, dir);
        test_cuda_matrix_file_operations(report, dir);
        std::filesystem::remove_all(dir);
    }
    catch(const std::exception& e)
    {
        ++report.failures;
        std::cout << "FAIL unexpected exception: " << e.what() << std::endl;
    }

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << std::endl;
    if(report.failures == 0)
    {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "FAILED" << std::endl;
    return EXIT_FAILURE;
}
