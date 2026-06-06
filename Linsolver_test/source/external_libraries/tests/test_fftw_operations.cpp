#include <cmath>
#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

int checks = 0;
int failures = 0;

template<class T>
struct fftw_traits;

template<>
struct fftw_traits<double>
{
    using real_type = double;
    using complex_type = fftw_complex;
    using plan_type = fftw_plan;

    static complex_type* alloc_complex(std::size_t n)
    {
        return reinterpret_cast<complex_type*>(fftw_malloc(sizeof(fftw_complex) * n));
    }

    static real_type* alloc_real(std::size_t n)
    {
        return reinterpret_cast<real_type*>(fftw_malloc(sizeof(real_type) * n));
    }

    static plan_type plan_r2c_1d(int n0, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_1d(n0, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_1d(int n0, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_1d(n0, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_2d(int n0, int n1, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_2d(n0, n1, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_2d(int n0, int n1, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_2d(n0, n1, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_3d(int n0, int n1, int n2, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_3d(int n0, int n1, int n2, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2c_1d(int n0, complex_type* in, complex_type* out, int sign)
    {
        return fftw_plan_dft_1d(n0, in, out, sign, FFTW_ESTIMATE);
    }

    static void execute(plan_type p)
    {
        fftw_execute(p);
    }

    static void destroy(plan_type p)
    {
        fftw_destroy_plan(p);
    }

    static void cleanup()
    {
        fftw_cleanup();
    }
};

template<>
struct fftw_traits<float>
{
    using real_type = float;
    using complex_type = fftwf_complex;
    using plan_type = fftwf_plan;

    static complex_type* alloc_complex(std::size_t n)
    {
        return reinterpret_cast<complex_type*>(fftwf_malloc(sizeof(fftwf_complex) * n));
    }

    static real_type* alloc_real(std::size_t n)
    {
        return reinterpret_cast<real_type*>(fftwf_malloc(sizeof(real_type) * n));
    }

    static plan_type plan_r2c_1d(int n0, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_1d(n0, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_1d(int n0, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_1d(n0, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_2d(int n0, int n1, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_2d(n0, n1, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_2d(int n0, int n1, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_2d(n0, n1, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_3d(int n0, int n1, int n2, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_3d(int n0, int n1, int n2, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2c_1d(int n0, complex_type* in, complex_type* out, int sign)
    {
        return fftwf_plan_dft_1d(n0, in, out, sign, FFTW_ESTIMATE);
    }

    static void execute(plan_type p)
    {
        fftwf_execute(p);
    }

    static void destroy(plan_type p)
    {
        fftwf_destroy_plan(p);
    }

    static void cleanup()
    {
        fftwf_cleanup();
    }
};

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

template<class T>
T complex_real(const typename fftw_traits<T>::complex_type& z)
{
    return z[0];
}

template<class T>
T complex_imag(const typename fftw_traits<T>::complex_type& z)
{
    return z[1];
}

template<class T>
void set_complex(typename fftw_traits<T>::complex_type& z, const std::complex<T>& value)
{
    z[0] = value.real();
    z[1] = value.imag();
}

template<class T>
std::complex<T> get_complex(const typename fftw_traits<T>::complex_type& z)
{
    return std::complex<T>(z[0], z[1]);
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
void check_plan(typename fftw_traits<T>::plan_type plan, const std::string& label)
{
    ++checks;
    if(!plan)
    {
        record_failure(label + " plan creation returned null");
    }
}

template<class T>
void test_r2c_roundtrip(const std::string& label, int n0, int n1, int n2)
{
    using traits = fftw_traits<T>;
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

    ++checks;
    if(complex_size != prefix_size * expected_reduced)
    {
        record_failure(label + " reduced-size layout mismatch");
    }

    T* in = traits::alloc_real(real_size);
    T* back = traits::alloc_real(real_size);
    typename traits::complex_type* out = traits::alloc_complex(complex_size);
    if(!in || !back || !out)
    {
        record_failure(label + " fftw_malloc failed");
        fftw_free(in);
        fftw_free(back);
        fftw_free(out);
        return;
    }

    T sum = T(0);
    for(std::size_t i = 0; i < real_size; ++i)
    {
        in[i] = sample_value<T>(i);
        back[i] = T(0);
        sum += in[i];
    }
    for(std::size_t i = 0; i < complex_size; ++i)
    {
        out[i][0] = T(0);
        out[i][1] = T(0);
    }

    typename traits::plan_type forward = nullptr;
    typename traits::plan_type inverse = nullptr;
    if(rank == 1)
    {
        forward = traits::plan_r2c_1d(n0, in, out);
        inverse = traits::plan_c2r_1d(n0, out, back);
    }
    else if(rank == 2)
    {
        forward = traits::plan_r2c_2d(n0, n1, in, out);
        inverse = traits::plan_c2r_2d(n0, n1, out, back);
    }
    else
    {
        forward = traits::plan_r2c_3d(n0, n1, n2, in, out);
        inverse = traits::plan_c2r_3d(n0, n1, n2, out, back);
    }

    check_plan<T>(forward, label + " forward");
    check_plan<T>(inverse, label + " inverse");
    if(forward && inverse)
    {
        traits::execute(forward);
        check_close<T>(
            complex_real<T>(out[0]),
            sum,
            tolerance<T>() * static_cast<T>(real_size),
            label + " DC real coefficient"
        );
        check_close<T>(
            complex_imag<T>(out[0]),
            T(0),
            tolerance<T>() * static_cast<T>(real_size),
            label + " DC imaginary coefficient"
        );

        traits::execute(inverse);
        const T scale = static_cast<T>(real_size);
        for(std::size_t i = 0; i < real_size; ++i)
        {
            check_close<T>(
                back[i] / scale,
                in[i],
                tolerance<T>() * (T(1) + std::abs(in[i])),
                label + " unnormalized inverse roundtrip i=" + std::to_string(i)
            );
        }
    }

    if(forward)
    {
        traits::destroy(forward);
    }
    if(inverse)
    {
        traits::destroy(inverse);
    }
    fftw_free(in);
    fftw_free(back);
    fftw_free(out);
}

template<class T>
void test_c2c_roundtrip(int n0)
{
    using traits = fftw_traits<T>;
    const std::string label = type_name<T>() + " c2c n=" + std::to_string(n0);
    const std::size_t n = static_cast<std::size_t>(n0);
    typename traits::complex_type* in = traits::alloc_complex(n);
    typename traits::complex_type* forward_out = traits::alloc_complex(n);
    typename traits::complex_type* back = traits::alloc_complex(n);
    if(!in || !forward_out || !back)
    {
        record_failure(label + " fftw_malloc failed");
        fftw_free(in);
        fftw_free(forward_out);
        fftw_free(back);
        return;
    }

    for(std::size_t i = 0; i < n; ++i)
    {
        set_complex<T>(in[i], sample_complex_value<T>(i));
        set_complex<T>(forward_out[i], std::complex<T>(0, 0));
        set_complex<T>(back[i], std::complex<T>(0, 0));
    }

    typename traits::plan_type forward = traits::plan_c2c_1d(n0, in, forward_out, FFTW_FORWARD);
    typename traits::plan_type inverse = traits::plan_c2c_1d(n0, forward_out, back, FFTW_BACKWARD);
    check_plan<T>(forward, label + " forward");
    check_plan<T>(inverse, label + " inverse");
    if(forward && inverse)
    {
        traits::execute(forward);
        traits::execute(inverse);
        const T scale = static_cast<T>(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            const std::complex<T> expected = get_complex<T>(in[i]);
            const std::complex<T> value = get_complex<T>(back[i]) / scale;
            check_close<T>(
                value.real(),
                expected.real(),
                tolerance<T>() * (T(1) + std::abs(expected.real())),
                label + " real roundtrip i=" + std::to_string(i)
            );
            check_close<T>(
                value.imag(),
                expected.imag(),
                tolerance<T>() * (T(1) + std::abs(expected.imag())),
                label + " imag roundtrip i=" + std::to_string(i)
            );
        }
    }

    if(forward)
    {
        traits::destroy(forward);
    }
    if(inverse)
    {
        traits::destroy(inverse);
    }
    fftw_free(in);
    fftw_free(forward_out);
    fftw_free(back);
}

template<class T>
void run_type_tests()
{
    std::cout << "Testing FFTW " << type_name<T>() << std::endl;
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d even", 8, 0, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 1d odd", 9, 0, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 2d", 5, 7, 0);
    test_r2c_roundtrip<T>(type_name<T>() + " r2c 3d", 4, 5, 6);
    test_c2c_roundtrip<T>(11);
    fftw_traits<T>::cleanup();
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
