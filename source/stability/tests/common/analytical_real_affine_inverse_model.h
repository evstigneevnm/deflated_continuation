#ifndef __STABILITY_TESTS_COMMON_ANALYTICAL_REAL_AFFINE_INVERSE_MODEL_H__
#define __STABILITY_TESTS_COMMON_ANALYTICAL_REAL_AFFINE_INVERSE_MODEL_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace stability
{
namespace tests
{

template<class RealVectorSpace>
class analytical_real_affine_inverse_model
{
public:
    using scalar_type = typename RealVectorSpace::scalar_type;
    using vector_type = typename RealVectorSpace::vector_type;

    static_assert(
        std::is_floating_point<scalar_type>::value,
        "the analytical model intentionally accepts real scalars only");

    analytical_real_affine_inverse_model(
        const RealVectorSpace& vector_space,
        std::vector<scalar_type> diagonal)
        : vector_space_(vector_space),
          diagonal_(std::move(diagonal))
    {
        if(diagonal_.size() != vector_space_.get_default_size())
            throw std::invalid_argument(
                "real affine model diagonal size mismatch");
    }

    bool preconditioner_jacobian_affine_u(
        vector_type& right_hand_side_to_solution,
        scalar_type jacobian_scale,
        scalar_type identity_shift) const
    {
        ++apply_calls_;
        if(apply_calls_ > fail_after_)
            return false;

        std::vector<scalar_type> host(diagonal_.size());
        vector_space_.get(
            right_hand_side_to_solution,
            host.data(),
            host.size());
        for(std::size_t index = 0; index < host.size(); ++index)
        {
            const scalar_type denominator =
                jacobian_scale*diagonal_[index] +
                identity_shift;
            if(
                !(std::abs(denominator) >
                  std::numeric_limits<scalar_type>::min()))
            {
                return false;
            }
            host[index] /= denominator;
        }
        vector_space_.set(
            host.data(),
            right_hand_side_to_solution,
            host.size());
        return true;
    }

    void fail_after(std::size_t successful_applications)
    {
        fail_after_ = successful_applications;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

private:
    const RealVectorSpace& vector_space_;
    std::vector<scalar_type> diagonal_;
    mutable std::size_t apply_calls_ = 0;
    std::size_t fail_after_ =
        std::numeric_limits<std::size_t>::max();
};

} // namespace tests
} // namespace stability

#endif
