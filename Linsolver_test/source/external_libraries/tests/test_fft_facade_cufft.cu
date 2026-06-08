#include <external_libraries/fft_facade_cufft.h>

#include <common/cuda_init_scfd.h>
#include <scfd/utils/cuda_safe_call.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

namespace fft = external_libraries::fft;

int checks = 0;
int failures = 0;

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(5e-4) : T(5e-10);
}

template<class T>
std::string type_name()
{
    return std::is_same<T, float>::value ? "float" : "double";
}

template<class T>
T sample_value(std::size_t i)
{
    const T a = static_cast<T>((static_cast<int>(i % 11) - 5) * 0.125);
    const T b = static_cast<T>(std::sin(static_cast<double>(i + 1) * 0.37));
    const T c = static_cast<T>(std::cos(static_cast<double>(i + 3) * 0.11));
    return a + b + T(0.25) * c;
}

template<class T>
struct complex_traits
{
    using transform_type = fft::c2c<fft::cufft_backend, T>;
    using complex_type = typename transform_type::complex_type;

    static complex_type make(T real, T imag)
    {
        complex_type value;
        value.x = real;
        value.y = imag;
        return value;
    }

    static T real(const complex_type& value)
    {
        return value.x;
    }

    static T imag(const complex_type& value)
    {
        return value.y;
    }
};

template<class T>
class device_buffer
{
public:
    explicit device_buffer(std::size_t size):
        size_(size)
    {
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * size_));
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    ~device_buffer()
    {
        if(ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
    }

    T* get()
    {
        return ptr_;
    }

    void copy_from(const std::vector<T>& host)
    {
        CUDA_SAFE_CALL(cudaMemcpy(ptr_, host.data(), sizeof(T) * size_, cudaMemcpyHostToDevice));
    }

    void copy_to(std::vector<T>& host) const
    {
        CUDA_SAFE_CALL(cudaMemcpy(host.data(), ptr_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

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

void check_equal(std::size_t value, std::size_t expected, const std::string& label)
{
    ++checks;
    if(value != expected)
    {
        record_failure(label + " value=" + std::to_string(value) +
                       " expected=" + std::to_string(expected));
    }
}

template<class T>
void test_r2c_roundtrip(const std::string& label, const fft::dimensions& dims)
{
    using transform_type = fft::r2c<fft::cufft_backend, T>;
    using complex_type = typename transform_type::complex_type;
    using ctraits = complex_traits<T>;

    std::vector<T> input(dims.physical_size());
    std::vector<T> back(dims.physical_size(), T(0));
    std::vector<complex_type> spectrum(dims.r2c_complex_size(), ctraits::make(T(0), T(0)));

    T sum = T(0);
    for(std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = sample_value<T>(i);
        sum += input[i];
    }

    try
    {
        transform_type fft_plan(dims);
        check_equal(fft_plan.get_reduced_size(), dims.reduced_size(), label + " get_reduced_size");
        check_equal(fft_plan.reduced_size(), dims.reduced_size(), label + " reduced_size");
        check_equal(fft_plan.physical_size(), dims.physical_size(), label + " physical_size");
        check_equal(fft_plan.complex_size(), dims.r2c_complex_size(), label + " complex_size");
        check_close<T>(fft_plan.normalization_factor(), static_cast<T>(dims.physical_size()), T(0), label + " normalization");

        device_buffer<T> d_input(input.size());
        device_buffer<T> d_back(back.size());
        device_buffer<complex_type> d_spectrum(spectrum.size());
        d_input.copy_from(input);

        fft_plan.forward(d_input.get(), d_spectrum.get());
        d_spectrum.copy_to(spectrum);
        check_close<T>(ctraits::real(spectrum[0]), sum, tolerance<T>() * static_cast<T>(input.size()), label + " DC real");
        check_close<T>(ctraits::imag(spectrum[0]), T(0), tolerance<T>() * static_cast<T>(input.size()), label + " DC imag");

        fft_plan.inverse(d_spectrum.get(), d_back.get());
        d_back.copy_to(back);
        const T scale = fft_plan.normalization_factor();
        for(std::size_t i = 0; i < input.size(); ++i)
        {
            check_close<T>(
                back[i] / scale,
                input[i],
                tolerance<T>() * (T(1) + std::abs(input[i])),
                label + " roundtrip i=" + std::to_string(i)
            );
        }
    }
    catch(const std::exception& e)
    {
        record_failure(label + " threw exception: " + e.what());
    }
}

template<class T>
void test_c2c_roundtrip(const std::string& label, const fft::dimensions& dims)
{
    using transform_type = fft::c2c<fft::cufft_backend, T>;
    using complex_type = typename transform_type::complex_type;
    using ctraits = complex_traits<T>;

    std::vector<complex_type> input(dims.physical_size());
    std::vector<complex_type> spectrum(dims.physical_size(), ctraits::make(T(0), T(0)));
    std::vector<complex_type> back(dims.physical_size(), ctraits::make(T(0), T(0)));

    for(std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = ctraits::make(sample_value<T>(i), T(0.5) * sample_value<T>(i + 7));
    }

    try
    {
        transform_type fft_plan(dims);
        check_equal(fft_plan.physical_size(), dims.physical_size(), label + " physical_size");
        check_equal(fft_plan.complex_size(), dims.physical_size(), label + " complex_size");
        check_close<T>(fft_plan.normalization_factor(), static_cast<T>(dims.physical_size()), T(0), label + " normalization");

        device_buffer<complex_type> d_input(input.size());
        device_buffer<complex_type> d_spectrum(spectrum.size());
        device_buffer<complex_type> d_back(back.size());
        d_input.copy_from(input);

        fft_plan.forward(d_input.get(), d_spectrum.get());
        fft_plan.inverse(d_spectrum.get(), d_back.get());
        d_back.copy_to(back);
        const T scale = fft_plan.normalization_factor();
        for(std::size_t i = 0; i < input.size(); ++i)
        {
            check_close<T>(
                ctraits::real(back[i]) / scale,
                ctraits::real(input[i]),
                tolerance<T>() * (T(1) + std::abs(ctraits::real(input[i]))),
                label + " real roundtrip i=" + std::to_string(i)
            );
            check_close<T>(
                ctraits::imag(back[i]) / scale,
                ctraits::imag(input[i]),
                tolerance<T>() * (T(1) + std::abs(ctraits::imag(input[i]))),
                label + " imag roundtrip i=" + std::to_string(i)
            );
        }
    }
    catch(const std::exception& e)
    {
        record_failure(label + " threw exception: " + e.what());
    }
}

template<class T>
void run_type_tests()
{
    std::cout << "Testing FFT facade cuFFT " << type_name<T>() << std::endl;
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d even", fft::dimensions(8));
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d odd", fft::dimensions(9));
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 2d", fft::dimensions(5, 7));
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 3d", fft::dimensions(4, 5, 6));
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 1d", fft::dimensions(11));
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 2d", fft::dimensions(4, 5));
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 3d", fft::dimensions(3, 4, 5));
}

} // namespace

int main(int argc, char** argv)
{
    const std::string device_selector = argc > 1 ? argv[1] : "auto";
    const int device = common::init_cuda_from_scfd_selector(device_selector);
    std::cout << "CUDA device: " << device << std::endl;

    run_type_tests<double>();
    run_type_tests<float>();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
