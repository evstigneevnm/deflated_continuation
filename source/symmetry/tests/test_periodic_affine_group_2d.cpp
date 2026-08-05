#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <symmetry/fourier/periodic_affine_element_2d.h>
#include <symmetry/generated_finite_group.h>

namespace
{

using element_type = symmetry::fourier::periodic_affine_element_2d;
using group_type = symmetry::generated_finite_group<element_type>;

void require(const bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

group_type make_square_group(const bool reverse_generator_order = false)
{
    group_type group(element_type::identity(), 16);
    if(reverse_generator_order)
    {
        group.add_generator("axis_swap", element_type::swap_axes());
        group.add_generator("half_shift_y", element_type::half_shift(1));
        group.add_generator("half_shift_x", element_type::half_shift(0));
    }
    else
    {
        group.add_generator("half_shift_x", element_type::half_shift(0));
        group.add_generator("half_shift_y", element_type::half_shift(1));
        group.add_generator("axis_swap", element_type::swap_axes());
    }
    group.finalize();
    return group;
}

} // namespace

int main()
{
    try
    {
        const auto identity = element_type::identity();
        const auto shift_x = element_type::half_shift(0);
        const auto shift_y = element_type::half_shift(1);
        const auto swap = element_type::swap_axes();

        require(
            swap.compose(shift_x).compose(swap) == shift_y,
            "axis swap must conjugate the x half-shift to the y half-shift");
        require(
            shift_x.compose(shift_y) == shift_y.compose(shift_x),
            "orthogonal half-period shifts must commute");

        auto group = make_square_group();
        require(group.size() == 8, "square periodic affine group must have order eight");
        for(const auto& entry: group.entries())
        {
            const auto& inverse = group.entry(entry.inverse_index).element;
            require(
                entry.element.compose(inverse) == identity &&
                    inverse.compose(entry.element) == identity,
                "generated inverse is inconsistent");
            for(const auto& right: group.entries())
            {
                (void)group.find_index(entry.element.compose(right.element));
            }
        }

        auto reordered_group = make_square_group(true);
        require(
            group.fingerprint() == reordered_group.fingerprint(),
            "group fingerprint must be independent of generator order");

        group_type rectangular_group(identity, 8);
        rectangular_group.add_generator("half_shift_x", shift_x);
        rectangular_group.add_generator("half_shift_y", shift_y);
        rectangular_group.finalize();
        require(
            rectangular_group.size() == 4,
            "rectangular inversion-odd residual group must have order four");
        require(
            rectangular_group.fingerprint() != group.fingerprint(),
            "different group closures must have different fingerprints");

        bool maximum_order_was_enforced = false;
        try
        {
            group_type undersized(identity, 4);
            undersized.add_generator("half_shift_x", shift_x);
            undersized.add_generator("half_shift_y", shift_y);
            undersized.add_generator("axis_swap", swap);
            undersized.finalize();
        }
        catch(const std::runtime_error&)
        {
            maximum_order_was_enforced = true;
        }
        require(
            maximum_order_was_enforced,
            "group closure must enforce the configured maximum order");

        std::cout << "periodic affine group 2D: PASS (order="
                  << group.size() << ", fingerprint="
                  << group.fingerprint() << ")\n";
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr << "periodic affine group 2D: FAIL: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
