#ifndef __DISCRETIZATION_COMMON_COMPONENT_FIELD_H__
#define __DISCRETIZATION_COMMON_COMPONENT_FIELD_H__

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <discretization/common/structured_extent.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

namespace discretization
{
namespace common
{

namespace detail
{

template<std::size_t Dimension>
struct component_field_initializer;

template<>
struct component_field_initializer<1>
{
    template<class Array, class Extent>
    static void init(Array& array, const Extent& extent)
    {
        array.init(static_cast<int>(extent[0]));
    }
};

template<>
struct component_field_initializer<2>
{
    template<class Array, class Extent>
    static void init(Array& array, const Extent& extent)
    {
        array.init(static_cast<int>(extent[0]), static_cast<int>(extent[1]));
    }
};

template<>
struct component_field_initializer<3>
{
    template<class Array, class Extent>
    static void init(Array& array, const Extent& extent)
    {
        array.init(
            static_cast<int>(extent[0]),
            static_cast<int>(extent[1]),
            static_cast<int>(extent[2])
        );
    }
};

} // namespace detail

template<class Backend, class T, std::size_t Dimension>
class component_field
{
public:
    using backend_type = Backend;
    using value_type = T;
    using memory_type = typename backend_type::memory_type;
    using extent_type = structured_extent<Dimension>;
    using array_type = scfd::arrays::array_nd<
        value_type,
        static_cast<scfd::arrays::ordinal_type>(Dimension),
        memory_type,
        scfd::arrays::last_index_fast_arranger
    >;

    component_field() = default;

    explicit component_field(const extent_type& extent):
        extent_(extent)
    {
        detail::component_field_initializer<Dimension>::init(values_, extent_);
    }

    component_field(const component_field&) = delete;
    component_field& operator=(const component_field&) = delete;
    component_field(component_field&&) noexcept = default;
    component_field& operator=(component_field&&) noexcept = default;

    const extent_type& extent() const
    {
        return extent_;
    }

    std::size_t size() const
    {
        return extent_.size();
    }

    value_type* data()
    {
        return values_.raw_ptr();
    }

    const value_type* data() const
    {
        return values_.raw_ptr();
    }

    array_type& array()
    {
        return values_;
    }

    const array_type& array() const
    {
        return values_;
    }

private:
    extent_type extent_;
    array_type values_;
};

} // namespace common
} // namespace discretization

#endif
