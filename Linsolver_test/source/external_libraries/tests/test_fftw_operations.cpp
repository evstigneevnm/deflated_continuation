#include <external_libraries/fftw_wrap.h>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

int checks = 0;
int failures = 0;

template<class T>
T tolerance()
{
    return std::is_same<T, float>::value ? T(2e-4) : T(2e-10);
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
std::complex<T> sample_complex_value(std::size_t i)
{
    return std::complex<T>(
        sample_value<T>(i),
        static_cast<T>(0.5) * sample_value<T>(i + 7)
    );
}

template<class T>
std::size_t reduced_size(std::size_t last_dim)
{
    return last_dim / 2 + 1;
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
std::unique_ptr<fftw_wrap_R2C<T>> make_r2c_wrapper(int rank, int n0, int n1, int n2)
{
    if(rank == 1)
    {
        return std::unique_ptr<fftw_wrap_R2C<T>>(new fftw_wrap_R2C<T>(static_cast<std::size_t>(n0)));
    }
    if(rank == 2)
    {
        return std::unique_ptr<fftw_wrap_R2C<T>>(
            new fftw_wrap_R2C<T>(static_cast<std::size_t>(n0), static_cast<std::size_t>(n1))
        );
    }
    return std::unique_ptr<fftw_wrap_R2C<T>>(
        new fftw_wrap_R2C<T>(
            static_cast<std::size_t>(n0),
            static_cast<std::size_t>(n1),
            static_cast<std::size_t>(n2)
        )
    );
}

template<class T>
std::unique_ptr<fftw_wrap_C2C<T>> make_c2c_wrapper(int rank, int n0, int n1, int n2)
{
    if(rank == 1)
    {
        return std::unique_ptr<fftw_wrap_C2C<T>>(new fftw_wrap_C2C<T>(static_cast<std::size_t>(n0)));
    }
    if(rank == 2)
    {
        return std::unique_ptr<fftw_wrap_C2C<T>>(
            new fftw_wrap_C2C<T>(static_cast<std::size_t>(n0), static_cast<std::size_t>(n1))
        );
    }
    return std::unique_ptr<fftw_wrap_C2C<T>>(
        new fftw_wrap_C2C<T>(
            static_cast<std::size_t>(n0),
            static_cast<std::size_t>(n1),
            static_cast<std::size_t>(n2)
        )
    );
}

template<class T>
void test_r2c_roundtrip(const std::string& label, int n0, int n1, int n2)
{
    const int rank = n2 > 0 ? 3 : (n1 > 0 ? 2 : 1);
    const std::size_t real_size =
        static_cast<std::size_t>(n0) *
        static_cast<std::size_t>(rank >= 2 ? n1 : 1) *
        static_cast<std::size_t>(rank >= 3 ? n2 : 1);
    const std::size_t last_dim = static_cast<std::size_t>(rank == 1 ? n0 : (rank == 2 ? n1 : n2));
    const std::size_t prefix_size =
        rank == 1 ? std::size_t(1) :
        (rank == 2 ? static_cast<std::size_t>(n0) : static_cast<std::size_t>(n0) * n1);
    const std::size_t expected_reduced = reduced_size<T>(last_dim);
    const std::size_t complex_size = prefix_size * expected_reduced;

    std::vector<T> in(real_size);
    std::vector<T> back(real_size, T(0));
    std::vector<std::complex<T>> out(complex_size, std::complex<T>(0, 0));
    std::vector<T> in_rebind(real_size);
    std::vector<T> back_rebind(real_size, T(0));
    std::vector<std::complex<T>> out_rebind(complex_size, std::complex<T>(0, 0));

    T sum = T(0);
    T sum_rebind = T(0);
    for(std::size_t i = 0; i < real_size; ++i)
    {
        in[i] = sample_value<T>(i);
        in_rebind[i] = sample_value<T>(i + 17) * T(0.75);
        sum += in[i];
        sum_rebind += in_rebind[i];
    }

    try
    {
        const auto wrapper = make_r2c_wrapper<T>(rank, n0, n1, n2);
        check_equal(wrapper->get_reduced_size(), expected_reduced, label + " get_reduced_size");
        check_equal(wrapper->reduced_size(), expected_reduced, label + " reduced_size");
        check_equal(wrapper->physical_size(), real_size, label + " physical_size");
        check_equal(wrapper->complex_size(), complex_size, label + " complex_size");
        check_close<T>(
            wrapper->normalization_factor(),
            static_cast<T>(real_size),
            T(0),
            label + " normalization_factor"
        );

        wrapper->forward(in.data(), out.data());
        check_close<T>(
            out[0].real(),
            sum,
            tolerance<T>() * static_cast<T>(real_size),
            label + " DC real coefficient"
        );
        check_close<T>(
            out[0].imag(),
            T(0),
            tolerance<T>() * static_cast<T>(real_size),
            label + " DC imaginary coefficient"
        );

        wrapper->inverse(out.data(), back.data());
        const T scale = wrapper->normalization_factor();
        for(std::size_t i = 0; i < real_size; ++i)
        {
            check_close<T>(
                back[i] / scale,
                in[i],
                tolerance<T>() * (T(1) + std::abs(in[i])),
                label + " unnormalized inverse roundtrip i=" + std::to_string(i)
            );
        }

        wrapper->forward(in_rebind.data(), out_rebind.data());
        check_close<T>(
            out_rebind[0].real(),
            sum_rebind,
            tolerance<T>() * static_cast<T>(real_size),
            label + " rebind DC real coefficient"
        );
        wrapper->inverse(out_rebind.data(), back_rebind.data());
        for(std::size_t i = 0; i < real_size; ++i)
        {
            check_close<T>(
                back_rebind[i] / scale,
                in_rebind[i],
                tolerance<T>() * (T(1) + std::abs(in_rebind[i])),
                label + " rebind unnormalized inverse roundtrip i=" + std::to_string(i)
            );
        }
    }
    catch(const std::exception& e)
    {
        record_failure(label + " threw exception: " + e.what());
    }
}

template<class T>
void test_c2c_roundtrip(const std::string& label, int n0, int n1, int n2)
{
    const int rank = n2 > 0 ? 3 : (n1 > 0 ? 2 : 1);
    const std::size_t n =
        static_cast<std::size_t>(n0) *
        static_cast<std::size_t>(rank >= 2 ? n1 : 1) *
        static_cast<std::size_t>(rank >= 3 ? n2 : 1);
    std::vector<std::complex<T>> in(n);
    std::vector<std::complex<T>> forward_out(n, std::complex<T>(0, 0));
    std::vector<std::complex<T>> back(n, std::complex<T>(0, 0));
    std::vector<std::complex<T>> in_rebind(n);
    std::vector<std::complex<T>> forward_out_rebind(n, std::complex<T>(0, 0));
    std::vector<std::complex<T>> back_rebind(n, std::complex<T>(0, 0));

    for(std::size_t i = 0; i < n; ++i)
    {
        in[i] = sample_complex_value<T>(i);
        in_rebind[i] = sample_complex_value<T>(i + 23) * std::complex<T>(T(0.5), T(-0.25));
    }

    try
    {
        const auto wrapper = make_c2c_wrapper<T>(rank, n0, n1, n2);
        check_equal(wrapper->physical_size(), n, label + " physical_size");
        check_equal(wrapper->complex_size(), n, label + " complex_size");
        check_close<T>(
            wrapper->normalization_factor(),
            static_cast<T>(n),
            T(0),
            label + " normalization_factor"
        );

        wrapper->forward(in.data(), forward_out.data());
        wrapper->inverse(forward_out.data(), back.data());
        const T scale = wrapper->normalization_factor();
        for(std::size_t i = 0; i < n; ++i)
        {
            const std::complex<T> value = back[i] / scale;
            check_close<T>(
                value.real(),
                in[i].real(),
                tolerance<T>() * (T(1) + std::abs(in[i].real())),
                label + " real roundtrip i=" + std::to_string(i)
            );
            check_close<T>(
                value.imag(),
                in[i].imag(),
                tolerance<T>() * (T(1) + std::abs(in[i].imag())),
                label + " imag roundtrip i=" + std::to_string(i)
            );
        }

        wrapper->forward(in_rebind.data(), forward_out_rebind.data());
        wrapper->inverse(forward_out_rebind.data(), back_rebind.data());
        for(std::size_t i = 0; i < n; ++i)
        {
            const std::complex<T> value = back_rebind[i] / scale;
            check_close<T>(
                value.real(),
                in_rebind[i].real(),
                tolerance<T>() * (T(1) + std::abs(in_rebind[i].real())),
                label + " rebind real roundtrip i=" + std::to_string(i)
            );
            check_close<T>(
                value.imag(),
                in_rebind[i].imag(),
                tolerance<T>() * (T(1) + std::abs(in_rebind[i].imag())),
                label + " rebind imag roundtrip i=" + std::to_string(i)
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
    std::cout << "Testing FFTW wrapper " << type_name<T>() << std::endl;
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d even", 8, 0, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d odd", 9, 0, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 2d", 5, 7, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 3d", 4, 5, 6);
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 1d", 11, 0, 0);
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 2d", 4, 5, 0);
    test_c2c_roundtrip<T>(type_name<T>() + " c2c 3d", 3, 4, 5);
}

} // namespace

int main()
{
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
