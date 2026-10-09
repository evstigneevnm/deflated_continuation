#ifndef NMFD_TIME_STEPPERS_SCIPY_ERROR_MAPPING_H
#define NMFD_TIME_STEPPERS_SCIPY_ERROR_MAPPING_H

#include <cmath>
#include <limits>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace detail
{
template <class T>
struct scipy_error_square_mapping
{
    using scalar_type = T;
    scalar_type absolute_tolerance;
    scalar_type relative_tolerance;

    __DEVICE_TAG__ scalar_type operator()( scalar_type error, scalar_type previous, scalar_type candidate ) const
    {
        const auto x      = std::abs( previous );
        const auto y      = std::abs( candidate );
        const auto scale  = absolute_tolerance + relative_tolerance * ( x > y ? x : y );
        const auto value  = error / scale;
        const auto square = value * value;
        return std::isfinite( error ) && std::isfinite( previous ) && std::isfinite( candidate ) &&
                       std::isfinite( scale ) && scale > 0 && std::isfinite( square )
                   ? square
                   : std::numeric_limits<scalar_type>::infinity();
    }
};
}
}
}

#endif
