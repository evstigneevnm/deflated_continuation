#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <common/cuda_init_scfd.h>
#include <external_libraries/lapack_wrap.h>

namespace
{

int checks = 0;
int failures = 0;

void check_cuda(cudaError_t status, const std::string& label)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(label + ": " + cudaGetErrorString(status));
    }
}

template<class T>
class device_buffer
{
public:
    explicit device_buffer(std::size_t count_):
        count(count_)
    {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count), "cudaMalloc");
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    ~device_buffer()
    {
        if(ptr_)
        {
            cudaFree(ptr_);
        }
    }

    T* get()
    {
        return ptr_;
    }

    const T* get() const
    {
        return ptr_;
    }

    void copy_from_host(const std::vector<T>& host)
    {
        if(host.size() != count)
        {
            throw std::runtime_error("device_buffer::copy_from_host size mismatch");
        }
        check_cuda(cudaMemcpy(ptr_, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }

private:
    std::size_t count = 0;
    T* ptr_ = nullptr;
};

template<class T>
std::string type_name()
{
    return std::is_same<T, float>::value ? "float" : "double";
}

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(2e-4) : T(2e-10);
}

std::size_t idx(std::size_t row, std::size_t col, std::size_t rows)
{
    return row + rows * col;
}

void record_failure(const std::string& message)
{
    ++failures;
    std::cerr << "FAIL " << message << std::endl;
}

template<class T>
void check_close(T value, T expected, T tol, const std::string& label)
{
    ++checks;
    const T err = std::abs(value - expected);
    if(!(err <= tol))
    {
        record_failure(label + " value=" + std::to_string(static_cast<double>(value)) +
                       " expected=" + std::to_string(static_cast<double>(expected)) +
                       " err=" + std::to_string(static_cast<double>(err)) +
                       " tol=" + std::to_string(static_cast<double>(tol)));
    }
}

template<class T>
bool has_eigenvalue(const std::vector<std::complex<T>>& eigs, std::complex<T> expected, T tol)
{
    for(const auto& eig : eigs)
    {
        if(std::abs(eig - expected) <= tol)
        {
            return true;
        }
    }
    return false;
}

template<class T>
std::vector<T> triangular_hessenberg_matrix_3()
{
    return {
        T(1.0), T(0.0), T(0.0),
        T(0.5), T(2.0), T(0.0),
        T(-0.25), T(1.0), T(3.0)
    };
}

template<class T>
std::vector<T> matmul(
    const std::vector<T>& A,
    const std::vector<T>& B,
    std::size_t n,
    bool transpose_A = false,
    bool transpose_B = false
)
{
    std::vector<T> C(n * n, T(0));
    for(std::size_t col = 0; col < n; ++col)
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            T sum = T(0);
            for(std::size_t k = 0; k < n; ++k)
            {
                const T a = transpose_A ? A[idx(k, row, n)] : A[idx(row, k, n)];
                const T b = transpose_B ? B[idx(col, k, n)] : B[idx(k, col, n)];
                sum += a * b;
            }
            C[idx(row, col, n)] = sum;
        }
    }
    return C;
}

template<class T>
void check_matrix_close(
    const std::vector<T>& value,
    const std::vector<T>& expected,
    T tol,
    const std::string& label
)
{
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        check_close<T>(value[i], expected[i], tol * (T(1) + std::abs(expected[i])), label + " i=" + std::to_string(i));
    }
}

template<class T>
void check_expected_eigs(const std::vector<std::complex<T>>& eigs, const std::string& label)
{
    const T tol = tolerance<T>() * T(100);
    for(std::size_t i = 0; i < 3; ++i)
    {
        ++checks;
        if(!has_eigenvalue<T>(eigs, std::complex<T>(T(i + 1), T(0)), tol))
        {
            record_failure(label + " missing eigenvalue " + std::to_string(i + 1));
        }
    }
}

template<class T>
void test_device_eigs_and_schur()
{
    const std::size_t n = 3;
    const auto A = triangular_hessenberg_matrix_3<T>();
    device_buffer<T> A_device(n * n);
    A_device.copy_from_host(A);

    lapack_wrap<T> lapack(n);

    std::vector<std::complex<T>> eigs_h(n);
    lapack.hessinberg_eigs_from_device(A_device.get(), n, eigs_h.data());
    check_expected_eigs<T>(eigs_h, type_name<T>() + " hessinberg_eigs_from_device");

    std::vector<std::complex<T>> eigs_h_gpu_alias(n);
    lapack.hessinberg_eigs_from_gpu(A_device.get(), n, eigs_h_gpu_alias.data());
    check_expected_eigs<T>(eigs_h_gpu_alias, type_name<T>() + " hessinberg_eigs_from_gpu");

    std::vector<T> Q(n * n, T(0));
    std::vector<T> R(n * n, T(0));
    std::vector<std::complex<T>> eigs_schur(n);
    lapack.hessinberg_schur_from_device(A_device.get(), n, Q.data(), R.data(), eigs_schur.data());
    const auto QR = matmul(Q, R, n);
    const auto QRQT = matmul(QR, Q, n, false, true);
    check_matrix_close<T>(
        QRQT,
        A,
        tolerance<T>() * T(200),
        type_name<T>() + " hessinberg_schur_from_device reconstruction"
    );
    check_expected_eigs<T>(eigs_schur, type_name<T>() + " hessinberg_schur_from_device eigs");

    std::vector<T> Qs(n * n, T(0));
    std::vector<T> Rs(n * n, T(0));
    std::vector<std::complex<T>> eigs_general_schur(n);
    lapack.eigs_schur_from_device(A_device.get(), n, Qs.data(), Rs.data(), eigs_general_schur.data());
    const auto QsRs = matmul(Qs, Rs, n);
    const auto QsRsQst = matmul(QsRs, Qs, n, false, true);
    check_matrix_close<T>(
        QsRsQst,
        A,
        tolerance<T>() * T(200),
        type_name<T>() + " eigs_schur_from_device reconstruction"
    );
    check_expected_eigs<T>(eigs_general_schur, type_name<T>() + " eigs_schur_from_device eigs");

    std::vector<T> Qsg(n * n, T(0));
    std::vector<T> Rsg(n * n, T(0));
    std::vector<std::complex<T>> eigs_general_schur_gpu_alias(n);
    lapack.eigs_schur_from_gpu(A_device.get(), n, Qsg.data(), Rsg.data(), eigs_general_schur_gpu_alias.data());
    check_expected_eigs<T>(eigs_general_schur_gpu_alias, type_name<T>() + " eigs_schur_from_gpu eigs");
}

template<class T>
void test_device_writers()
{
    const std::size_t n = 3;
    const auto A = triangular_hessenberg_matrix_3<T>();
    device_buffer<T> A_device(n * n);
    A_device.copy_from_host(A);

    const std::vector<T> v{T(1.25), T(-2.5), T(3.75)};
    device_buffer<T> v_device(v.size());
    v_device.copy_from_host(v);

    lapack_wrap<T> lapack(n);
    const auto dir = std::filesystem::temp_directory_path() / "linsolver_lapack_wrap_cuda_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto matrix_file = dir / (type_name<T>() + "_matrix.dat");
    const auto vector_file = dir / (type_name<T>() + "_vector.dat");

    lapack.write_matrix_from_device(matrix_file.string(), n, n, A_device.get(), 17);
    lapack.write_vector_from_device(vector_file.string(), v.size(), v_device.get(), 17);

    std::ifstream matrix_in(matrix_file);
    if(!matrix_in)
    {
        record_failure(type_name<T>() + " write_matrix_from_device file missing");
    }
    else
    {
        for(std::size_t row = 0; row < n; ++row)
        {
            for(std::size_t col = 0; col < n; ++col)
            {
                double value = 0.0;
                matrix_in >> value;
                check_close<T>(
                    static_cast<T>(value),
                    A[idx(row, col, n)],
                    tolerance<T>() * T(10),
                    type_name<T>() + " write_matrix_from_device row=" + std::to_string(row) + " col=" + std::to_string(col)
                );
            }
        }
    }

    std::ifstream vector_in(vector_file);
    if(!vector_in)
    {
        record_failure(type_name<T>() + " write_vector_from_device file missing");
    }
    else
    {
        for(std::size_t i = 0; i < v.size(); ++i)
        {
            double value = 0.0;
            vector_in >> value;
            check_close<T>(
                static_cast<T>(value),
                v[i],
                tolerance<T>() * T(10),
                type_name<T>() + " write_vector_from_device i=" + std::to_string(i)
            );
        }
    }

    std::filesystem::remove_all(dir);
}

template<class T>
void run_type_tests()
{
    std::cout << "Testing device LAPACK wrapper " << type_name<T>() << std::endl;
    test_device_eigs_and_schur<T>();
    test_device_writers<T>();
}

} // namespace

int main(int argc, char** argv)
{
    const std::string device_selector = argc > 1 ? argv[1] : "auto";
    try
    {
        const int device = common::init_cuda_from_scfd_selector(device_selector);
        std::cout << "CUDA device: " << device << std::endl;

        run_type_tests<double>();
        run_type_tests<float>();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    catch(const std::exception& exc)
    {
        record_failure(std::string("unexpected exception: ") + exc.what());
    }

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
