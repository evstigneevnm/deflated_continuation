#ifndef __SYMMETRY_FOURIER_PERIODIC_AFFINE_ELEMENT_2D_H__
#define __SYMMETRY_FOURIER_PERIODIC_AFFINE_ELEMENT_2D_H__

#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace symmetry
{
namespace fourier
{

class periodic_fraction
{
public:
    periodic_fraction() = default;

    periodic_fraction(std::int64_t numerator, std::int64_t denominator):
        numerator_(numerator),
        denominator_(denominator)
    {
        normalize();
    }

    std::int64_t numerator() const { return numerator_; }
    std::int64_t denominator() const { return denominator_; }

    template<class Scalar>
    Scalar value() const
    {
        return static_cast<Scalar>(numerator_)/
            static_cast<Scalar>(denominator_);
    }

    periodic_fraction operator-() const
    {
        return periodic_fraction(-numerator_, denominator_);
    }

    periodic_fraction operator+(const periodic_fraction& right) const
    {
        const std::int64_t denominator =
            std::lcm(denominator_, right.denominator_);
        return periodic_fraction(
            numerator_*(denominator/denominator_) +
                right.numerator_*(denominator/right.denominator_),
            denominator);
    }

    periodic_fraction operator*(const int multiplier) const
    {
        return periodic_fraction(
            numerator_*static_cast<std::int64_t>(multiplier),
            denominator_);
    }

    bool operator==(const periodic_fraction& right) const
    {
        return numerator_ == right.numerator_ &&
            denominator_ == right.denominator_;
    }

    bool operator!=(const periodic_fraction& right) const
    {
        return !(*this == right);
    }

    std::string key() const
    {
        return std::to_string(numerator_) + "/" +
            std::to_string(denominator_);
    }

private:
    void normalize()
    {
        if(denominator_ == 0)
        {
            throw std::invalid_argument(
                "periodic_fraction denominator must not be zero");
        }
        if(denominator_ < 0)
        {
            denominator_ = -denominator_;
            numerator_ = -numerator_;
        }
        numerator_ %= denominator_;
        if(numerator_ < 0)
        {
            numerator_ += denominator_;
        }
        if(numerator_ == 0)
        {
            denominator_ = 1;
            return;
        }
        const std::int64_t divisor =
            std::gcd(numerator_, denominator_);
        numerator_ /= divisor;
        denominator_ /= divisor;
    }

    std::int64_t numerator_ = 0;
    std::int64_t denominator_ = 1;
};

struct signed_permutation_2d
{
    std::array<int, 4> values{{1, 0, 0, 1}};

    static signed_permutation_2d identity()
    {
        return signed_permutation_2d{};
    }

    static signed_permutation_2d swap_axes()
    {
        return signed_permutation_2d{{0, 1, 1, 0}};
    }

    void validate() const
    {
        for(const int value: values)
        {
            if(value < -1 || value > 1)
            {
                throw std::invalid_argument(
                    "signed_permutation_2d entries must be -1, 0, or 1");
            }
        }
        for(int row = 0; row < 2; ++row)
        {
            int count = 0;
            for(int column = 0; column < 2; ++column)
            {
                count += values[2*row + column] == 0 ? 0 : 1;
            }
            if(count != 1)
            {
                throw std::invalid_argument(
                    "signed_permutation_2d needs one nonzero entry per row");
            }
        }
        for(int column = 0; column < 2; ++column)
        {
            int count = 0;
            for(int row = 0; row < 2; ++row)
            {
                count += values[2*row + column] == 0 ? 0 : 1;
            }
            if(count != 1)
            {
                throw std::invalid_argument(
                    "signed_permutation_2d needs one nonzero entry per column");
            }
        }
    }

    signed_permutation_2d compose(
        const signed_permutation_2d& right) const
    {
        signed_permutation_2d result{{0, 0, 0, 0}};
        for(int row = 0; row < 2; ++row)
        {
            for(int column = 0; column < 2; ++column)
            {
                for(int inner = 0; inner < 2; ++inner)
                {
                    result.values[2*row + column] +=
                        values[2*row + inner]*
                        right.values[2*inner + column];
                }
            }
        }
        result.validate();
        return result;
    }

    signed_permutation_2d inverse() const
    {
        return signed_permutation_2d{{
            values[0], values[2], values[1], values[3]}};
    }

    std::array<periodic_fraction, 2> apply(
        const std::array<periodic_fraction, 2>& vector) const
    {
        return {{
            vector[0]*values[0] + vector[1]*values[1],
            vector[0]*values[2] + vector[1]*values[3]}};
    }

    bool operator==(const signed_permutation_2d& right) const
    {
        return values == right.values;
    }

    bool operator!=(const signed_permutation_2d& right) const
    {
        return !(*this == right);
    }

    std::string key() const
    {
        std::ostringstream result;
        result << values[0] << ',' << values[1] << ','
               << values[2] << ',' << values[3];
        return result.str();
    }
};

class periodic_affine_element_2d
{
public:
    periodic_affine_element_2d() = default;

    periodic_affine_element_2d(
        signed_permutation_2d linear,
        std::array<periodic_fraction, 2> translation,
        const int scalar_sign = 1):
        linear_(std::move(linear)),
        translation_(std::move(translation)),
        scalar_sign_(scalar_sign)
    {
        linear_.validate();
        if(scalar_sign_ != -1 && scalar_sign_ != 1)
        {
            throw std::invalid_argument(
                "periodic_affine_element_2d scalar sign must be -1 or 1");
        }
    }

    static periodic_affine_element_2d identity()
    {
        return periodic_affine_element_2d{};
    }

    static periodic_affine_element_2d half_shift(const std::size_t axis)
    {
        if(axis >= 2)
        {
            throw std::out_of_range(
                "periodic_affine_element_2d half-shift axis is out of range");
        }
        std::array<periodic_fraction, 2> translation{};
        translation[axis] = periodic_fraction(1, 2);
        return periodic_affine_element_2d(
            signed_permutation_2d::identity(),
            translation);
    }

    static periodic_affine_element_2d swap_axes()
    {
        return periodic_affine_element_2d(
            signed_permutation_2d::swap_axes(),
            {});
    }

    periodic_affine_element_2d compose(
        const periodic_affine_element_2d& right) const
    {
        const signed_permutation_2d linear =
            linear_.compose(right.linear_);
        const auto transformed_translation =
            linear_.apply(right.translation_);
        return periodic_affine_element_2d(
            linear,
            {{
                transformed_translation[0] + translation_[0],
                transformed_translation[1] + translation_[1]}},
            scalar_sign_*right.scalar_sign_);
    }

    periodic_affine_element_2d inverse() const
    {
        const signed_permutation_2d inverse_linear = linear_.inverse();
        const auto inverse_translation =
            inverse_linear.apply({{-translation_[0], -translation_[1]}});
        return periodic_affine_element_2d(
            inverse_linear,
            inverse_translation,
            scalar_sign_);
    }

    const signed_permutation_2d& linear() const { return linear_; }
    const std::array<periodic_fraction, 2>& translation() const
    {
        return translation_;
    }
    int scalar_sign() const { return scalar_sign_; }

    bool operator==(const periodic_affine_element_2d& right) const
    {
        return linear_ == right.linear_ &&
            translation_ == right.translation_ &&
            scalar_sign_ == right.scalar_sign_;
    }

    bool operator!=(const periodic_affine_element_2d& right) const
    {
        return !(*this == right);
    }

    std::string key() const
    {
        return linear_.key() + ";" + translation_[0].key() + "," +
            translation_[1].key() + ";" +
            std::to_string(scalar_sign_);
    }

private:
    signed_permutation_2d linear_;
    std::array<periodic_fraction, 2> translation_{};
    int scalar_sign_ = 1;
};

} // namespace fourier
} // namespace symmetry

#endif
