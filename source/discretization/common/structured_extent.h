#ifndef __DISCRETIZATION_COMMON_STRUCTURED_EXTENT_H__
#define __DISCRETIZATION_COMMON_STRUCTURED_EXTENT_H__

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace discretization
{
namespace common
{

template<std::size_t Dimension>
class structured_extent
{
public:
    using size_type = std::size_t;
    using dimensions_type = std::array<size_type, Dimension>;

    structured_extent() = default;

    explicit structured_extent(const dimensions_type& dimensions):
        dimensions_(dimensions)
    {
        validate();
    }

    template<class... Sizes>
    explicit structured_extent(Sizes... sizes):
        dimensions_{static_cast<size_type>(sizes)...}
    {
        static_assert(sizeof...(Sizes) == Dimension, "structured_extent dimension mismatch");
        validate();
    }

    size_type operator[](const size_type dimension) const
    {
        return dimensions_.at(dimension);
    }

    const dimensions_type& dimensions() const
    {
        return dimensions_;
    }

    size_type size() const
    {
        size_type result = 1;
        for(const size_type value: dimensions_)
        {
            if(value > std::numeric_limits<size_type>::max()/result)
            {
                throw std::overflow_error("structured_extent total size overflows size_t");
            }
            result *= value;
        }
        return result;
    }

    bool operator==(const structured_extent& other) const
    {
        return dimensions_ == other.dimensions_;
    }

    bool operator!=(const structured_extent& other) const
    {
        return !(*this == other);
    }

private:
    void validate() const
    {
        for(const size_type value: dimensions_)
        {
            if(value == 0)
            {
                throw std::invalid_argument("structured_extent dimensions must be positive");
            }
        }
    }

    dimensions_type dimensions_{};
};

} // namespace common
} // namespace discretization

#endif
