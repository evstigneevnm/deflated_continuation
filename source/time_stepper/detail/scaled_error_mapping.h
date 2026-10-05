#ifndef NMFD_TIME_STEPPERS_SCALED_ERROR_MAPPING_H
#define NMFD_TIME_STEPPERS_SCALED_ERROR_MAPPING_H

#include <cmath>
#include <limits>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace detail
{

// Matlab scaling: the controller divides the reduced maximum by rtol.
template<class T>
struct scaled_error_mapping
{
    using scalar_type = T;
    scalar_type scale_floor;
    scalar_type invalid = std::numeric_limits<scalar_type>::infinity();

    __DEVICE_TAG__ scalar_type operator()(scalar_type error, scalar_type previous, scalar_type candidate) const
    {
        const auto x = std::abs(previous);
        const auto y = std::abs(candidate);
        auto scale = x > y ? x : y;
        scale = scale_floor > scale ? scale_floor : scale;
        const auto value = std::abs(error / scale);
        return std::isfinite(error) && std::isfinite(previous) && std::isfinite(candidate) &&
            std::isfinite(scale) && scale > 0 && std::isfinite(value) ? value : invalid;
    }
};

}
}
}
#endif
