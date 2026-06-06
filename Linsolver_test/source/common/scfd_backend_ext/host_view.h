#ifndef __COMMON_SCFD_BACKEND_EXT_HOST_VIEW_H__
#define __COMMON_SCFD_BACKEND_EXT_HOST_VIEW_H__

#include <cstddef>
#include <utility>

#include <scfd/arrays/array.h>
#include <scfd/arrays/tensor_array_nd_view.h>

namespace common
{
namespace scfd_backend_ext
{

template<class View>
class scoped_host_view
{
public:
    scoped_host_view() = default;

    scoped_host_view(const typename View::array_type& array, bool sync_from_array, bool sync_on_destroy):
        sync_on_destroy_(sync_on_destroy),
        active_(true),
        view_(array, sync_from_array)
    {
    }

    scoped_host_view(const scoped_host_view&) = delete;
    scoped_host_view& operator=(const scoped_host_view&) = delete;

    scoped_host_view(scoped_host_view&& other):
        sync_on_destroy_(other.sync_on_destroy_),
        active_(other.active_),
        view_(std::move(other.view_))
    {
        other.sync_on_destroy_ = false;
        other.active_ = false;
    }

    scoped_host_view& operator=(scoped_host_view&& other)
    {
        if(this != &other)
        {
            release(sync_on_destroy_);
            sync_on_destroy_ = other.sync_on_destroy_;
            active_ = other.active_;
            view_ = std::move(other.view_);
            other.sync_on_destroy_ = false;
            other.active_ = false;
        }
        return *this;
    }

    ~scoped_host_view()
    {
        release(sync_on_destroy_);
    }

    void release(bool sync_to_array)
    {
        if(active_)
        {
            view_.release(sync_to_array);
            active_ = false;
            sync_on_destroy_ = false;
        }
    }

    void sync_to_array() const
    {
        view_.sync_to_array();
    }

    void sync_from_array() const
    {
        view_.sync_from_array();
    }

    auto raw_ptr() -> decltype(std::declval<View&>().raw_ptr())
    {
        return view_.raw_ptr();
    }

    auto raw_ptr() const -> decltype(std::declval<const View&>().raw_ptr())
    {
        return view_.raw_ptr();
    }

    auto size() const -> decltype(std::declval<const View&>().size())
    {
        return view_.size();
    }

    auto operator[](std::size_t index) -> decltype(std::declval<View&>().raw_ptr()[index])
    {
        return view_.raw_ptr()[index];
    }

    auto operator[](std::size_t index) const -> decltype(std::declval<const View&>().raw_ptr()[index])
    {
        return view_.raw_ptr()[index];
    }

    template<class... Args>
    auto operator()(Args&&... args) -> decltype(std::declval<View&>()(std::forward<Args>(args)...))
    {
        return view_(std::forward<Args>(args)...);
    }

    template<class... Args>
    auto operator()(Args&&... args) const -> decltype(std::declval<const View&>()(std::forward<Args>(args)...))
    {
        return view_(std::forward<Args>(args)...);
    }

    View& get()
    {
        return view_;
    }

    const View& get() const
    {
        return view_;
    }

private:
    bool sync_on_destroy_ = false;
    bool active_ = false;
    View view_;
};

template<class Array>
struct host_view_traits
{
    using array_type = Array;
    using view_type = typename array_type::view_type;
    using scoped_view_type = scoped_host_view<view_type>;
    using memory_type = typename array_type::memory_type;
    using host_memory_type = typename memory_type::host_memory_type;
    using value_type = typename array_type::value_type;
    using host_array_type = scfd::arrays::array<value_type, host_memory_type>;
};

template<class HostArray>
void ensure_host_array(HostArray& array, std::size_t size)
{
    if(!array.is_free() && static_cast<std::size_t>(array.size()) >= size)
    {
        return;
    }
    if(!array.is_free())
    {
        array.free();
    }
    array.init(static_cast<typename HostArray::ordinal_type>(size));
}

} // namespace scfd_backend_ext
} // namespace common

#endif
