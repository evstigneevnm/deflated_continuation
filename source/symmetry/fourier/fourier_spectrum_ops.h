#ifndef __SYMMETRY_FOURIER_FOURIER_SPECTRUM_OPS_H__
#define __SYMMETRY_FOURIER_FOURIER_SPECTRUM_OPS_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace symmetry
{
namespace fourier
{

template<class Complex>
typename Complex::value_type fourier_spectrum_distance_sq(
    const std::vector<Complex>& left,
    const std::vector<Complex>& right)
{
    if(left.size() != right.size())
    {
        throw std::runtime_error("Fourier spectrum sizes do not match");
    }
    using scalar_type = typename Complex::value_type;
    scalar_type result = scalar_type(0);
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        result += std::norm(left[i] - right[i]);
    }
    return result;
}

template<class Complex>
typename Complex::value_type fourier_spectrum_real_inner(
    const std::vector<Complex>& left,
    const std::vector<Complex>& right)
{
    if(left.size() != right.size())
    {
        throw std::runtime_error("Fourier spectrum sizes do not match");
    }
    using scalar_type = typename Complex::value_type;
    scalar_type result = scalar_type(0);
    for(std::size_t i = 0; i < left.size(); ++i)
    {
        result += left[i].real()*right[i].real() + left[i].imag()*right[i].imag();
    }
    return result;
}

template<class Complex>
typename Complex::value_type fourier_spectrum_delta_inner(
    const std::vector<Complex>& value,
    const std::vector<Complex>& reference,
    const std::vector<Complex>& tangent)
{
    if(value.size() != reference.size() || value.size() != tangent.size())
    {
        throw std::runtime_error("Fourier spectrum sizes do not match");
    }
    using scalar_type = typename Complex::value_type;
    scalar_type result = scalar_type(0);
    for(std::size_t i = 0; i < value.size(); ++i)
    {
        result += (value[i].real() - reference[i].real())*tangent[i].real() +
                  (value[i].imag() - reference[i].imag())*tangent[i].imag();
    }
    return result;
}

template<class Complex>
void fourier_zero_small_components(
    std::vector<Complex>& values,
    const typename Complex::value_type tolerance)
{
    using scalar_type = typename Complex::value_type;
    for(auto& value: values)
    {
        const scalar_type real_part =
            std::abs(value.real()) <= tolerance ? scalar_type(0) : value.real();
        const scalar_type imag_part =
            std::abs(value.imag()) <= tolerance ? scalar_type(0) : value.imag();
        value = Complex(real_part, imag_part);
    }
}

template<class Complex>
bool fourier_spectrum_lexicographically_greater(
    const std::vector<Complex>& left,
    const std::vector<Complex>& right,
    const typename Complex::value_type tolerance)
{
    const std::size_t size = std::min(left.size(), right.size());
    for(std::size_t mode = 1; mode < size; ++mode)
    {
        const auto real_delta = left[mode].real() - right[mode].real();
        if(real_delta > tolerance) return true;
        if(real_delta < -tolerance) return false;

        const auto imag_delta = left[mode].imag() - right[mode].imag();
        if(imag_delta > tolerance) return true;
        if(imag_delta < -tolerance) return false;
    }
    return false;
}

} // namespace fourier
} // namespace symmetry

#endif
