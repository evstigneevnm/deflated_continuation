#ifndef __DISCRETIZATION_FOURIER_INITIALIZATION_LOW_MODE_ODD_FIELD_H__
#define __DISCRETIZATION_FOURIER_INITIALIZATION_LOW_MODE_ODD_FIELD_H__

#include <cstddef>
#include <cstdint>

#include <common/scfd_backend_ext/math.h>
#include <scfd/utils/device_tag.h>

namespace discretization
{
namespace fourier
{
namespace initialization
{

template<class Backend, class Scalar, class PhysicalField>
void fill_low_mode_odd_physical_field_2d(
    PhysicalField& field,
    const std::size_t nx,
    const std::size_t ny,
    const Scalar parameter,
    const Scalar linear_high_order_coefficient,
    const std::uint64_t seed_id
)
{
    using for_each_type = typename Backend::template for_each_type<std::ptrdiff_t>;
    using math_type = ::common::scfd_backend_ext::math<Backend, Scalar>;
    using scalar_traits = ::common::scfd_backend_ext::scalar_traits<Backend, Scalar>;

    Scalar* values = field.data();
    const unsigned int profile = static_cast<unsigned int>(seed_id%8);
    const Scalar two_pi = Scalar(2)*Scalar(3.141592653589793238462643383279502884L);
    for_each_type for_each;
    for_each(
        [=] __DEVICE_TAG__ (const std::ptrdiff_t flat_index)
        {
            const std::size_t ix = static_cast<std::size_t>(flat_index)/ny;
            const std::size_t iy = static_cast<std::size_t>(flat_index)%ny;
            const Scalar x = two_pi*static_cast<Scalar>(ix)/static_cast<Scalar>(nx);
            const Scalar y = two_pi*static_cast<Scalar>(iy)/static_cast<Scalar>(ny);

            if(profile == 0)
            {
                values[flat_index] = math_type::sin(x);
                return;
            }
            if(profile == 1)
            {
                values[flat_index] = math_type::sin(y);
                return;
            }
            if(profile == 2)
            {
                values[flat_index] = Scalar(0.7071067811865475244L)*
                    (math_type::sin(x) + math_type::sin(y));
                return;
            }
            if(profile == 3)
            {
                values[flat_index] = Scalar(0.7071067811865475244L)*
                    (math_type::sin(x) - math_type::sin(y));
                return;
            }

            const Scalar profile_scale = profile == 4 ? Scalar(0.5) :
                (profile == 5 ? Scalar(1) : (profile == 6 ? Scalar(2) : Scalar(4)));
            Scalar value = Scalar(0);
            std::size_t mode_index = 0;
            for(int mode_x = 0; mode_x <= 3; ++mode_x)
            {
                for(int mode_y = -3; mode_y <= 3; ++mode_y)
                {
                    if(mode_x == 0 && mode_y <= 0)
                    {
                        continue;
                    }
                    const Scalar mode_squared = static_cast<Scalar>(
                        mode_x*mode_x + mode_y*mode_y);
                    if(mode_squared == Scalar(0))
                    {
                        continue;
                    }
                    const Scalar random_coefficient = Scalar(2)*
                        scalar_traits::random_scalar(mode_index++, static_cast<std::size_t>(seed_id)) -
                        Scalar(1);
                    const Scalar resonance = Scalar(1)/(
                        Scalar(1) + math_type::abs(
                            linear_high_order_coefficient*mode_squared - parameter));
                    value += random_coefficient*resonance/mode_squared*
                        math_type::sin(static_cast<Scalar>(mode_x)*x +
                            static_cast<Scalar>(mode_y)*y);
                }
            }
            values[flat_index] = profile_scale*value;
        },
        static_cast<std::ptrdiff_t>(field.size())
    );
    for_each.wait();
}

} // namespace initialization
} // namespace fourier
} // namespace discretization

#endif
