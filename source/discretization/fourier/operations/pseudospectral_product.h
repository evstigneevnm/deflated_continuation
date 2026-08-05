#ifndef __DISCRETIZATION_FOURIER_OPERATIONS_PSEUDOSPECTRAL_PRODUCT_H__
#define __DISCRETIZATION_FOURIER_OPERATIONS_PSEUDOSPECTRAL_PRODUCT_H__

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <discretization/fourier/normalized_fft.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{
namespace operations
{

template<class Backend, class FFTBackend, class T, class DealiasingPolicy>
class pseudospectral_product_2d
{
public:
    using transform_type = normalized_r2c_transform<Backend, FFTBackend, T, 2>;
    using physical_field_type = typename transform_type::physical_field_type;
    using spectral_field_type = typename transform_type::spectral_field_type;
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    using copy_type = typename Backend::copy_type;

    pseudospectral_product_2d(transform_type* transform, DealiasingPolicy dealiasing):
        transform_(require_transform(transform)),
        dealiasing_(std::move(dealiasing)),
        filtered_left_(transform_->spectral_extent()),
        filtered_right_(transform_->spectral_extent()),
        physical_left_(transform_->physical_extent()),
        physical_right_(transform_->physical_extent()),
        physical_product_(transform_->physical_extent())
    {
    }

    void apply(
        const spectral_field_type& left,
        const spectral_field_type& right,
        spectral_field_type& product
    )
    {
        if(left.size() != right.size() || left.size() != product.size() ||
           left.size() != transform_->complex_size())
        {
            throw std::invalid_argument("pseudospectral_product_2d spectrum size mismatch");
        }
        copy_type()(
            static_cast<std::ptrdiff_t>(left.size()),
            left.data(),
            filtered_left_.data()
        );
        copy_type()(
            static_cast<std::ptrdiff_t>(right.size()),
            right.data(),
            filtered_right_.data()
        );
        dealiasing_.template apply<Backend>(filtered_left_);
        dealiasing_.template apply<Backend>(filtered_right_);
        transform_->inverse(filtered_left_, physical_left_);
        transform_->inverse(filtered_right_, physical_right_);
        const T* left_values = physical_left_.data();
        const T* right_values = physical_right_.data();
        T* product_values = physical_product_.data();
        for_each_type for_each;
        for_each(
            [=] __DEVICE_TAG__ (const std::ptrdiff_t index)
            {
                product_values[index] = left_values[index]*right_values[index];
            },
            static_cast<std::ptrdiff_t>(physical_product_.size())
        );
        for_each.wait();
        transform_->forward(physical_product_, product);
        dealiasing_.template apply<Backend>(product);
    }

private:
    static transform_type* require_transform(transform_type* transform)
    {
        if(transform == nullptr)
        {
            throw std::invalid_argument("pseudospectral_product_2d requires an FFT transform");
        }
        return transform;
    }

    transform_type* transform_;
    DealiasingPolicy dealiasing_;
    spectral_field_type filtered_left_;
    spectral_field_type filtered_right_;
    physical_field_type physical_left_;
    physical_field_type physical_right_;
    physical_field_type physical_product_;
};

} // namespace operations
} // namespace fourier
} // namespace discretization

#endif
