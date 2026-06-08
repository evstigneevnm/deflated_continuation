#ifndef __SYMMETRY_FOURIER_FOURIER_SLICE_H__
#define __SYMMETRY_FOURIER_FOURIER_SLICE_H__

#include <symmetry/fourier/fourier_slice_1d.h>

namespace symmetry
{
namespace fourier
{

template <class Complex>
using fourier_slice = fourier_slice_1d<Complex>;

} // namespace fourier
} // namespace symmetry

#endif
