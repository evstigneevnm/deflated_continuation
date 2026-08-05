#ifndef __DISCRETIZATION_FOURIER_CODECS_COMPONENT_PRODUCT_H__
#define __DISCRETIZATION_FOURIER_CODECS_COMPONENT_PRODUCT_H__

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace discretization
{
namespace fourier
{
namespace codecs
{

template<class ComponentCodec, std::size_t ComponentCount>
class component_product
{
public:
    explicit component_product(ComponentCodec codec):
        codec_(std::move(codec))
    {
    }

    std::size_t component_size() const
    {
        return codec_.state_size();
    }

    std::size_t state_size() const
    {
        return ComponentCount*component_size();
    }

    const ComponentCodec& component_codec() const
    {
        return codec_;
    }

private:
    ComponentCodec codec_;
};

} // namespace codecs
} // namespace fourier
} // namespace discretization

#endif
