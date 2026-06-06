#ifndef __COMMON_SCALAR_MATH_H__
#define __COMMON_SCALAR_MATH_H__

#include <cmath>

namespace common
{
namespace scalar_math
{

template<class T>
T abs(const T& value)
{
    using std::abs;
    return abs(value);
}

template<class T>
T sqrt(const T& value)
{
    using std::sqrt;
    return sqrt(value);
}

template<class T>
bool isfinite(const T& value)
{
    using std::isfinite;
    return isfinite(value);
}

template<class T>
bool isnan(const T& value)
{
    using std::isnan;
    return isnan(value);
}

template<class T>
bool isinf(const T& value)
{
    using std::isinf;
    return isinf(value);
}

} // namespace scalar_math
} // namespace common

#endif
