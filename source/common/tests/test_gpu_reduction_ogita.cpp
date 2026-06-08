#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <common/cuda_init_scfd.h>
#include <nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita.h>

namespace
{

struct ld_complex
{
    long double real = 0.0L;
    long double imag = 0.0L;
};

struct reference_values
{
    ld_complex sum;
    long double asum = 0.0L;
    ld_complex dot;
    long double norm = 0.0L;
};

enum class data_pattern
{
    signed_abs,
    cancellation,
    mixed
};

template<class T>
struct value_traits
{
    using real_type = T;
    static constexpr bool is_complex = false;

    static T make(long double real, long double)
    {
        return static_cast<T>(real);
    }

    static ld_complex to_ld(T value)
    {
        return {static_cast<long double>(value), 0.0L};
    }
};

template<class T>
struct value_traits<thrust::complex<T>>
{
    using real_type = T;
    static constexpr bool is_complex = true;

    static thrust::complex<T> make(long double real, long double imag)
    {
        return thrust::complex<T>(static_cast<T>(real), static_cast<T>(imag));
    }

    static ld_complex to_ld(thrust::complex<T> value)
    {
        return {static_cast<long double>(value.real()), static_cast<long double>(value.imag())};
    }
};

template<class T>
using real_type_t = typename value_traits<T>::real_type;

template<class T>
class device_buffer
{
public:
    explicit device_buffer(std::size_t size):
    ptr_(nullptr)
    {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * size), "cudaMalloc");
    }

    ~device_buffer()
    {
        if(ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get()
    {
        return ptr_;
    }

private:
    static void check_cuda(cudaError_t status, const char* expression)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(expression) + " failed: " + cudaGetErrorString(status));
        }
    }

    T* ptr_ = nullptr;
};

template<class T>
void copy_to_device(T* device, const T* host, std::size_t size)
{
    const cudaError_t status = cudaMemcpy(device, host, sizeof(T) * size, cudaMemcpyHostToDevice);
    if(status != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaMemcpy host-to-device failed: ") + cudaGetErrorString(status));
    }
}

struct test_report
{
    int checks = 0;
    int failures = 0;
    int details_printed = 0;
    int max_details = 80;

    void fail(const std::string& message)
    {
        ++failures;
        if(details_printed < max_details)
        {
            std::cout << "FAIL " << message << '\n';
        }
        else if(details_printed == max_details)
        {
            std::cout << "Further failures suppressed.\n";
        }
        ++details_printed;
    }
};

const char* pattern_name(data_pattern pattern)
{
    switch(pattern)
    {
        case data_pattern::signed_abs:
            return "signed_abs";
        case data_pattern::cancellation:
            return "cancellation";
        case data_pattern::mixed:
            return "mixed";
    }
    return "unknown";
}

template<class T>
const char* type_name()
{
    if constexpr(std::is_same<T, float>::value)
    {
        return "float";
    }
    else if constexpr(std::is_same<T, double>::value)
    {
        return "double";
    }
    else if constexpr(std::is_same<T, thrust::complex<float>>::value)
    {
        return "thrust::complex<float>";
    }
    else if constexpr(std::is_same<T, thrust::complex<double>>::value)
    {
        return "thrust::complex<double>";
    }
    else
    {
        return "unknown";
    }
}

long double abs_ld(long double value)
{
    return (value < 0.0L) ? -value : value;
}

std::string format_ld(long double value)
{
    std::ostringstream out;
    out << std::setprecision(21) << value;
    return out.str();
}

std::string format_complex(ld_complex value)
{
    std::ostringstream out;
    out << '(' << std::setprecision(21) << value.real << ", " << value.imag << ')';
    return out.str();
}

template<class T>
bool exact_equal(T lhs, T rhs)
{
    if constexpr(value_traits<T>::is_complex)
    {
        return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
    }
    else
    {
        return lhs == rhs;
    }
}

bool check_real(
    test_report& report,
    const std::string& label,
    long double gpu,
    long double ref,
    long double rel_tol,
    long double abs_tol)
{
    ++report.checks;

    if(!std::isfinite(gpu))
    {
        report.fail(label + " is not finite: gpu=" + format_ld(gpu) + " ref=" + format_ld(ref));
        return false;
    }

    const long double scale = std::max(1.0L, abs_ld(ref));
    const long double tol = abs_tol + rel_tol * scale;
    const long double err = abs_ld(gpu - ref);

    if(err > tol)
    {
        report.fail(
            label + " gpu=" + format_ld(gpu) +
            " ref=" + format_ld(ref) +
            " err=" + format_ld(err) +
            " tol=" + format_ld(tol));
        return false;
    }
    return true;
}

template<class T>
bool check_complex(
    test_report& report,
    const std::string& label,
    T gpu,
    ld_complex ref,
    long double rel_tol,
    long double abs_tol)
{
    const ld_complex gpu_ld = value_traits<T>::to_ld(gpu);
    bool ok = check_real(report, label + ".real", gpu_ld.real, ref.real, rel_tol, abs_tol);
    if constexpr(value_traits<T>::is_complex)
    {
        ok = check_real(report, label + ".imag", gpu_ld.imag, ref.imag, rel_tol, abs_tol) && ok;
    }
    return ok;
}

template<class T>
long double cancellation_large()
{
    using real_type = real_type_t<T>;
    if constexpr(std::is_same<real_type, float>::value)
    {
        return 1.0e10L;
    }
    else
    {
        return 1.0e16L;
    }
}

template<class T>
void make_values(std::size_t i, data_pattern pattern, T& x, T& y)
{
    using traits = value_traits<T>;

    long double xr = 0.0L;
    long double xi = 0.0L;
    long double yr = 0.0L;
    long double yi = 0.0L;

    if(pattern == data_pattern::signed_abs)
    {
        const long double sign = (i % 2 == 0) ? -1.0L : 1.0L;
        xr = sign * (1.0L + static_cast<long double>(i % 13) / 7.0L);
        xi = traits::is_complex ? -sign * (0.25L + static_cast<long double>(i % 5) / 11.0L) : 0.0L;
        yr = ((i % 3 == 0) ? -1.0L : 1.0L) * (0.5L + static_cast<long double>(i % 7) / 9.0L);
        yi = traits::is_complex ? ((i % 4 < 2) ? 0.375L : -0.625L) : 0.0L;
    }
    else if(pattern == data_pattern::cancellation)
    {
        const long double large = cancellation_large<T>();
        const int k = static_cast<int>(i % 3);

        if(k == 0)
        {
            xr = large;
            xi = traits::is_complex ? 0.5L * large : 0.0L;
        }
        else if(k == 1)
        {
            xr = 1.0L;
            xi = traits::is_complex ? -2.0L : 0.0L;
        }
        else
        {
            xr = -large;
            xi = traits::is_complex ? -0.5L * large : 0.0L;
        }

        yr = 1.0L;
        yi = traits::is_complex ? -0.5L : 0.0L;
    }
    else
    {
        xr = (static_cast<long double>((i * 37 + 11) % 97) - 48.0L) / 17.0L;
        xi = traits::is_complex ? (static_cast<long double>((i * 19 + 5) % 83) - 41.0L) / 23.0L : 0.0L;
        yr = (static_cast<long double>((i * 29 + 7) % 89) - 44.0L) / 13.0L;
        yi = traits::is_complex ? (static_cast<long double>((i * 31 + 3) % 79) - 39.0L) / 19.0L : 0.0L;
    }

    x = traits::make(xr, xi);
    y = traits::make(yr, yi);
}

template<class T>
void accumulate_reference(reference_values& ref, long double& norm_sq, T x, T y)
{
    const ld_complex x_ld = value_traits<T>::to_ld(x);
    const ld_complex y_ld = value_traits<T>::to_ld(y);

    ref.sum.real += x_ld.real;
    ref.sum.imag += x_ld.imag;
    ref.asum += abs_ld(x_ld.real) + abs_ld(x_ld.imag);

    ref.dot.real += x_ld.real * y_ld.real + x_ld.imag * y_ld.imag;
    ref.dot.imag += x_ld.real * y_ld.imag - x_ld.imag * y_ld.real;
    norm_sq += x_ld.real * x_ld.real + x_ld.imag * x_ld.imag;
}

template<class T>
reference_values fill_input(std::vector<T>& x, std::vector<T>& y, data_pattern pattern)
{
    reference_values ref;
    long double norm_sq = 0.0L;

    for(std::size_t i = 0; i < x.size(); ++i)
    {
        make_values(i, pattern, x[i], y[i]);
        accumulate_reference(ref, norm_sq, x[i], y[i]);
    }

    ref.norm = std::sqrt(norm_sq);
    return ref;
}

template<class T>
void check_repeat(
    test_report& report,
    const std::string& label,
    T first,
    T current)
{
    ++report.checks;
    if(!exact_equal(first, current))
    {
        report.fail(
            label + " changed between repeats: first=" +
            format_complex(value_traits<T>::to_ld(first)) +
            " current=" + format_complex(value_traits<T>::to_ld(current)));
    }
}

template<class T>
void check_repeat_real(
    test_report& report,
    const std::string& label,
    real_type_t<T> first,
    real_type_t<T> current)
{
    ++report.checks;
    if(first != current)
    {
        report.fail(
            label + " changed between repeats: first=" +
            format_ld(static_cast<long double>(first)) +
            " current=" + format_ld(static_cast<long double>(current)));
    }
}

template<class T>
T zero_value()
{
    return value_traits<T>::make(0.0L, 0.0L);
}

template<class T>
void run_case_impl(std::size_t n, data_pattern pattern, int repeats, test_report& report)
{
    using real_type = real_type_t<T>;

    std::vector<T> x(n);
    std::vector<T> y(n);
    const reference_values ref = fill_input(x, y, pattern);

    device_buffer<T> x_d(n);
    device_buffer<T> y_d(n);

    copy_to_device<T>(x_d.get(), x.data(), n);
    copy_to_device<T>(y_d.get(), y.data(), n);

    gpu_reduction_ogita<T, T*> reducer(n);

    const long double rel_tol = 4096.0L * static_cast<long double>(std::numeric_limits<real_type>::epsilon());
    const long double abs_tol = 4096.0L * static_cast<long double>(std::numeric_limits<real_type>::epsilon());

    T first_sum = zero_value<T>();
    T first_dot = zero_value<T>();
    real_type first_asum = real_type(0);
    real_type first_norm = real_type(0);
    bool have_first = false;

    for(int repeat = 0; repeat < repeats; ++repeat)
    {
        const T gpu_sum = reducer.sum(x_d.get());
        const real_type gpu_asum = reducer.asum(x_d.get());
        const T gpu_dot = reducer.dot(x_d.get(), y_d.get());
        const real_type gpu_norm = reducer.norm(x_d.get());

        const std::string label =
            std::string(type_name<T>()) +
            " n=" + std::to_string(n) +
            " pattern=" + pattern_name(pattern) +
            " repeat=" + std::to_string(repeat);

        check_complex(report, label + " sum", gpu_sum, ref.sum, rel_tol, abs_tol);
        check_real(report, label + " asum", static_cast<long double>(gpu_asum), ref.asum, rel_tol, abs_tol);
        check_complex(report, label + " dot", gpu_dot, ref.dot, rel_tol, abs_tol);
        check_real(report, label + " norm", static_cast<long double>(gpu_norm), ref.norm, rel_tol, abs_tol);

        if(have_first)
        {
            check_repeat(report, label + " sum", first_sum, gpu_sum);
            check_repeat_real<T>(report, label + " asum", first_asum, gpu_asum);
            check_repeat(report, label + " dot", first_dot, gpu_dot);
            check_repeat_real<T>(report, label + " norm", first_norm, gpu_norm);
        }
        else
        {
            first_sum = gpu_sum;
            first_asum = gpu_asum;
            first_dot = gpu_dot;
            first_norm = gpu_norm;
            have_first = true;
        }
    }
}

template<class T>
void run_type(test_report& report)
{
    const std::vector<std::size_t> sizes = {
        1,
        2,
        31,
        32,
        33,
        1023,
        1024,
        1025,
        2049,
        4097
    };
    const std::vector<data_pattern> patterns = {
        data_pattern::signed_abs,
        data_pattern::cancellation,
        data_pattern::mixed
    };

    std::cout << "Testing " << type_name<T>() << '\n';
    for(data_pattern pattern: patterns)
    {
        for(std::size_t n: sizes)
        {
            run_case_impl<T>(n, pattern, 1, report);
        }
    }

    // BLOCK_SIZE_1D is 1024. This gives 2049 first-level partial sums, so the
    // recursive phase needs more than one block and exercises in-place recursion.
    const std::size_t recursive_size = 4ULL * 1024ULL * 1024ULL + 129ULL;
    run_case_impl<T>(recursive_size, data_pattern::cancellation, 3, report);
}

} // namespace

int main(int argc, char const* argv[])
{
    std::cout << std::setprecision(17);
    const std::string cuda_selector = (argc > 1) ? argv[1] : "auto";
    const int cuda_device = common::init_cuda_from_scfd_selector(cuda_selector);
    std::cout << "CUDA device: " << cuda_device << '\n';

    test_report report;

    run_type<float>(report);
    run_type<double>(report);
    run_type<thrust::complex<float>>(report);
    run_type<thrust::complex<double>>(report);

    std::cout << "Checks: " << report.checks << ", failures: " << report.failures << '\n';

    if(report.failures == 0)
    {
        std::cout << "PASS\n";
        return 0;
    }

    std::cout << "FAILED\n";
    return 1;
}
