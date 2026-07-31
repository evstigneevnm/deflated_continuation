#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_AFFINE_INVERSE_PROVIDER_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_AFFINE_INVERSE_PROVIDER_H__

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class RealAffineInverseProvider>
class scaled_real_affine_inverse_provider
{
public:
    using provider_type = RealAffineInverseProvider;
    using scalar_type = typename provider_type::scalar_type;
    using vector_type = typename provider_type::vector_type;
    using health_type = typename provider_type::health_type;

    scaled_real_affine_inverse_provider(
        std::shared_ptr<const provider_type> provider,
        scalar_type scale)
        : provider_(std::move(provider)),
          scale_(scale)
    {
        if(!provider_)
            throw std::invalid_argument(
                "scaled affine inverse requires a provider");
    }

    bool apply(
        scalar_type operator_scale,
        scalar_type identity_shift,
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++apply_calls_;
        const bool succeeded = provider_->apply(
            operator_scale*scale_,
            identity_shift,
            right_hand_side,
            solution);
        if(!succeeded)
            ++failed_applications_;
        return succeeded;
    }

    scalar_type scale() const
    {
        return scale_;
    }

    const provider_type& provider() const
    {
        return *provider_;
    }

    std::size_t apply_calls() const
    {
        return apply_calls_;
    }

    std::size_t failed_applications() const
    {
        return failed_applications_;
    }

    health_type health(
        scalar_type operator_scale,
        scalar_type identity_shift) const
    {
        return provider_->health(
            operator_scale*scale_,
            identity_shift);
    }

private:
    std::shared_ptr<const provider_type> provider_;
    scalar_type scale_;
    mutable std::size_t apply_calls_ = 0;
    mutable std::size_t failed_applications_ = 0;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
