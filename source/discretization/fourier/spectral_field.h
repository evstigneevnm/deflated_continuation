#ifndef __DISCRETIZATION_FOURIER_SPECTRAL_FIELD_H__
#define __DISCRETIZATION_FOURIER_SPECTRAL_FIELD_H__

#include <discretization/common/component_field.h>

namespace discretization
{
namespace fourier
{

template<class Backend, class T, std::size_t Dimension>
using physical_field = discretization::common::component_field<Backend, T, Dimension>;

template<class Backend, class Complex, std::size_t Dimension>
using spectral_field = discretization::common::component_field<Backend, Complex, Dimension>;

} // namespace fourier
} // namespace discretization

#endif
