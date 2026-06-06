#ifndef __COMMON_SCFD_BACKEND_EXT_ARG_REDUCE_H__
#define __COMMON_SCFD_BACKEND_EXT_ARG_REDUCE_H__

#include <cstddef>
#include <limits>
#include <utility>

#include <scfd/utils/device_tag.h>

#if __has_include(<thrust/execution_policy.h>) && __has_include(<thrust/iterator/counting_iterator.h>) && __has_include(<thrust/iterator/transform_iterator.h>) && __has_include(<thrust/reduce.h>)
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#define COMMON_SCFD_BACKEND_EXT_HAS_THRUST_REDUCE 1
#endif

namespace scfd
{
namespace backend
{
struct cuda;
struct hip;
}
}

namespace common
{
namespace scfd_backend_ext
{

template<class T, class Ordinal>
struct indexed_value
{
    T value;
    Ordinal index;
};

template<class T, class Ordinal>
struct indexed_max_op
{
    __DEVICE_TAG__ indexed_value<T, Ordinal> operator()(
        const indexed_value<T, Ordinal>& a,
        const indexed_value<T, Ordinal>& b) const
    {
        if(b.value > a.value)
        {
            return b;
        }
        if(a.value > b.value)
        {
            return a;
        }
        return b.index < a.index ? b : a;
    }
};

template<class Backend, class T, class Ordinal>
struct arg_reduce
{
    static std::pair<T, std::size_t> max_argmax(Ordinal size, const T* input)
    {
        indexed_value<T, Ordinal> result{input[0], Ordinal(0)};
        for(Ordinal i = 1; i < size; ++i)
        {
            if(input[i] > result.value)
            {
                result = indexed_value<T, Ordinal>{input[i], i};
            }
        }
        return {result.value, static_cast<std::size_t>(result.index)};
    }
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_REDUCE
template<class T, class Ordinal>
struct thrust_indexed_value_functor
{
    const T* input;

    __DEVICE_TAG__ indexed_value<T, Ordinal> operator()(Ordinal i) const
    {
        return indexed_value<T, Ordinal>{input[i], i};
    }
};

template<class T, class Ordinal>
std::pair<T, std::size_t> thrust_max_argmax(Ordinal size, const T* input)
{
    auto begin = thrust::make_transform_iterator(
        thrust::counting_iterator<Ordinal>(Ordinal(0)),
        thrust_indexed_value_functor<T, Ordinal>{input});
    const indexed_value<T, Ordinal> init{std::numeric_limits<T>::lowest(), Ordinal(0)};
    const auto result = thrust::reduce(
        thrust::device,
        begin,
        begin + size,
        init,
        indexed_max_op<T, Ordinal>());
    return {result.value, static_cast<std::size_t>(result.index)};
}

template<class T, class Ordinal>
struct arg_reduce<scfd::backend::cuda, T, Ordinal>
{
    static std::pair<T, std::size_t> max_argmax(Ordinal size, const T* input)
    {
        return thrust_max_argmax(size, input);
    }
};

template<class T, class Ordinal>
struct arg_reduce<scfd::backend::hip, T, Ordinal>
{
    static std::pair<T, std::size_t> max_argmax(Ordinal size, const T* input)
    {
        return thrust_max_argmax(size, input);
    }
};
#endif

} // namespace scfd_backend_ext
} // namespace common

#endif
