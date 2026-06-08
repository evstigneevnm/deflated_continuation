#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <type_traits>

#include <common/scfd_backend_ext/complex.h>

namespace
{

int checks = 0;
int failures = 0;

void require(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        std::cout << "FAIL " << label << std::endl;
        ++failures;
    }
}

template<class T>
void require_near(const std::string& label, T value, T expected, T tol)
{
    ++checks;
    const T err = std::abs(value - expected);
    if(!(err <= tol))
    {
        std::cout << "FAIL " << label << " value=" << value
                  << " expected=" << expected << " err=" << err
                  << " tol=" << tol << std::endl;
        ++failures;
    }
}

template<class Complex>
void test_complex_type(const std::string& name)
{
    using traits = common::scfd_backend_ext::complex_value_traits<Complex>;
    using real_type = typename traits::real_type;

    const real_type tol = std::is_same<real_type, float>::value ? real_type(1e-6) : real_type(1e-13);

    const Complex z = traits::make(real_type(3), real_type(-4));
    require_near(name + " real", traits::real(z), real_type(3), tol);
    require_near(name + " imag", traits::imag(z), real_type(-4), tol);
    require_near(name + " abs_sq", traits::abs_sq(z), real_type(25), tol);

    const Complex z_conj = traits::conj(z);
    require_near(name + " conj real", traits::real(z_conj), real_type(3), tol);
    require_near(name + " conj imag", traits::imag(z_conj), real_type(4), tol);

    const Complex product = traits::mul(z, z_conj);
    require_near(name + " product real", traits::real(product), real_type(25), tol*real_type(10));
    require_near(name + " product imag", traits::imag(product), real_type(0), tol*real_type(10));

    const Complex added = traits::add(z, traits::from_real(real_type(2)));
    require_near(name + " add real", traits::real(added), real_type(5), tol);
    require_near(name + " add imag", traits::imag(added), real_type(-4), tol);

    require(common::scfd_backend_ext::is_complex_type<Complex>::value, name + " is_complex_type");
}

} // namespace

int main()
{
    using common::scfd_backend_ext::complex_t;

    require(
        (std::is_same<complex_t<scfd::backend::serial_cpu, double>, std::complex<double>>::value),
        "serial_cpu complex_t<double>");
    require(
        (std::is_same<complex_t<scfd::backend::omp, float>, std::complex<float>>::value),
        "omp complex_t<float>");

    test_complex_type<std::complex<float>>("std::complex<float>");
    test_complex_type<std::complex<double>>("std::complex<double>");

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
    require(
        (std::is_same<complex_t<scfd::backend::cuda, float>, thrust::complex<float>>::value),
        "cuda complex_t<float>");
    require(
        (std::is_same<complex_t<scfd::backend::hip, double>, thrust::complex<double>>::value),
        "hip complex_t<double>");
    test_complex_type<thrust::complex<float>>("thrust::complex<float>");
    test_complex_type<thrust::complex<double>>("thrust::complex<double>");
#endif

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_CUFFT_COMPLEX
    test_complex_type<cufftComplex>("cufftComplex");
    test_complex_type<cufftDoubleComplex>("cufftDoubleComplex");
#endif

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
