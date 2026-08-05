#ifndef __DISCRETIZATION_FOURIER_CODECS_CODEC_DESCRIPTOR_2D_H__
#define __DISCRETIZATION_FOURIER_CODECS_CODEC_DESCRIPTOR_2D_H__

#include <cstddef>

namespace discretization
{
namespace fourier
{
namespace codecs
{
namespace detail
{

enum class coefficient_kind_2d : int
{
    self_real = 0,
    complex = 1,
    pure_imaginary = 2
};

struct coefficient_orbit_2d
{
    std::ptrdiff_t spectrum_index = 0;
    std::ptrdiff_t partner_index = -1;
    std::ptrdiff_t state_offset = 0;
    coefficient_kind_2d kind = coefficient_kind_2d::self_real;
};

} // namespace detail
} // namespace codecs
} // namespace fourier
} // namespace discretization

#endif
