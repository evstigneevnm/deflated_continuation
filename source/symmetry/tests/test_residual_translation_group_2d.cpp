#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <symmetry/fourier/residual_translation_group_2d.h>

namespace
{

class report
{
public:
    void check(const bool condition, const std::string& message)
    {
        ++checks_;
        if(!condition)
        {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    int finish() const
    {
        std::cout << "2D residual translation group: " << checks_
                  << " checks, " << failures_ << " failures\n";
        return failures_ == 0 ? 0 : 1;
    }

private:
    std::size_t checks_ = 0;
    std::size_t failures_ = 0;
};

using mode_type = symmetry::fourier::mode_index<2>;
using character_type =
    symmetry::fourier::residual_translation_character_2d;

bool has_character(
    const std::vector<character_type>& characters,
    const std::vector<mode_type>& modes,
    const std::vector<int>& signs)
{
    for(const auto& character: characters)
    {
        bool matches = modes.size() == signs.size();
        for(std::size_t index = 0; matches && index < modes.size(); ++index)
        {
            matches = character.compatible(modes[index]) &&
                character.sign(modes[index]) == signs[index];
        }
        if(matches)
        {
            return true;
        }
    }
    return false;
}

bool has_character_with_half_shift(
    const std::vector<character_type>& characters,
    const std::vector<mode_type>& modes,
    const std::vector<int>& signs)
{
    for(const auto& character: characters)
    {
        for(const int shift_x: {0, 1})
        {
            for(const int shift_y: {0, 1})
            {
                bool matches = modes.size() == signs.size();
                for(std::size_t index = 0;
                    matches && index < modes.size();
                    ++index)
                {
                    const int parity =
                        shift_x*modes[index][0] +
                        shift_y*modes[index][1];
                    const int half_shift_sign = parity%2 == 0 ? 1 : -1;
                    matches = character.compatible(modes[index]) &&
                        half_shift_sign*character.sign(modes[index]) ==
                            signs[index];
                }
                if(matches)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

int main()
{
    report result;

    {
        const std::vector<mode_type> modes{{1, 0}, {0, 1}, {1, 1}};
        const std::vector<double> amplitudes{1.0, 0.8, 0.4};
        const auto group =
            symmetry::fourier::build_residual_translation_group_2d(
                modes,
                amplitudes);
        result.check(group.active_rank == 2, "generic support has rank two");
        result.check(
            group.characters.size() == 1 &&
                group.characters.front().is_identity(),
            "coordinate half shifts exhaust a generic residual group");
    }

    {
        const std::vector<mode_type> modes{
            {1, 1}, {2, 0}, {0, 2}, {3, 1}, {1, 3}};
        const std::vector<double> amplitudes{3.0, 2.0, 1.5, 0.8, 0.6};
        const auto group =
            symmetry::fourier::build_residual_translation_group_2d(
                modes,
                amplitudes);
        result.check(group.active_rank == 2, "same-parity support has rank two");
        result.check(
            group.characters.size() == 2,
            "same-parity support adds one residual quotient character");
        result.check(
            has_character_with_half_shift(
                group.characters,
                modes,
                {1, -1, -1, -1, -1}),
            "quarter anti-diagonal translation is represented");
    }

    {
        const std::vector<mode_type> modes{{2, 0}, {4, 0}, {6, 0}};
        const std::vector<double> amplitudes{2.0, 1.0, 0.5};
        const auto group =
            symmetry::fourier::build_residual_translation_group_2d(
                modes,
                amplitudes);
        result.check(
            group.active_rank == 1,
            "one-dimensional support reports reduced active rank");
        result.check(
            group.characters.size() == 2,
            "rank-one even modes retain a nontrivial residual character");
        result.check(
            has_character(group.characters, modes, {-1, 1, -1}),
            "rank-one residual character alternates along its lattice basis");
    }

    {
        const std::vector<mode_type> modes{
            {2, 0}, {0, 2}, {2, 2}, {1, 0}};
        const std::vector<double> amplitudes{2.0, 1.0, 0.5, 1.0e-15};
        const auto group =
            symmetry::fourier::build_residual_translation_group_2d(
                modes,
                amplitudes);
        result.check(
            group.characters.size() == 4,
            "even two-dimensional lattice has four residual quotient characters");
        result.check(
            group.active_tolerance > amplitudes.back(),
            "roundoff-size incompatible modes do not destroy isotropy");
    }

    {
        const std::vector<mode_type> modes{{1, 0}, {0, 1}};
        const std::vector<double> amplitudes{0.0, 0.0};
        const auto group =
            symmetry::fourier::build_residual_translation_group_2d(
                modes,
                amplitudes);
        result.check(group.active_rank == 0, "zero state has active rank zero");
        result.check(
            group.characters.size() == 1 &&
                group.characters.front().is_identity(),
            "zero state uses the identity representative only");
    }

    return result.finish();
}
