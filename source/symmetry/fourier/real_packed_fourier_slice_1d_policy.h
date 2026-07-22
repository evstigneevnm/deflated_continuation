#ifndef __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_POLICY_H__
#define __SYMMETRY_FOURIER_REAL_PACKED_FOURIER_SLICE_1D_POLICY_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <symmetry/fourier/lsq_fourier_slice_1d_policy.h>

namespace symmetry
{
namespace fourier
{

enum class real_packed_fourier_1d_stabilizer_policy
{
    single_mode,
    lsq_multimode
};

inline const char* real_packed_fourier_1d_stabilizer_policy_name(
    const real_packed_fourier_1d_stabilizer_policy policy)
{
    switch(policy)
    {
        case real_packed_fourier_1d_stabilizer_policy::single_mode:
            return "single_mode";
        case real_packed_fourier_1d_stabilizer_policy::lsq_multimode:
            return "lsq_multimode";
    }
    return "unknown";
}

template<class T>
struct real_packed_fourier_slice_1d_policy
{
    using scalar_type = T;

    real_packed_fourier_1d_stabilizer_policy stabilizer =
        real_packed_fourier_1d_stabilizer_policy::single_mode;

    T relative_active_mode_tolerance = T(0);
    T continuation_mode_switch_ratio = T(0.25);
    T tangent_continuity_weight = T(0.25);
    T tangent_backward_penalty = T(4);

    lsq_fourier_slice_1d_policy<T> lsq;
    T local_representative_relative_tolerance = T(0.05);

    void validate() const
    {
        if(!finite(relative_active_mode_tolerance) || relative_active_mode_tolerance < T(0))
        {
            throw std::invalid_argument("relative active mode tolerance must be finite and non-negative");
        }
        if(!finite(continuation_mode_switch_ratio) ||
           continuation_mode_switch_ratio < T(0) || continuation_mode_switch_ratio > T(1))
        {
            throw std::invalid_argument("continuation mode switch ratio must be finite and in [0,1]");
        }
        if(!finite(tangent_continuity_weight) || tangent_continuity_weight < T(0))
        {
            throw std::invalid_argument("tangent continuity weight must be finite and non-negative");
        }
        if(!finite(tangent_backward_penalty) || tangent_backward_penalty < T(0))
        {
            throw std::invalid_argument("tangent backward penalty must be finite and non-negative");
        }
        lsq.validate();
        if(!finite(local_representative_relative_tolerance) ||
           local_representative_relative_tolerance < T(0))
        {
            throw std::invalid_argument(
                "continuation locality relative tolerance must be finite and non-negative");
        }
    }

    bool operator==(const real_packed_fourier_slice_1d_policy& other) const
    {
        return stabilizer == other.stabilizer &&
               relative_active_mode_tolerance == other.relative_active_mode_tolerance &&
               continuation_mode_switch_ratio == other.continuation_mode_switch_ratio &&
               tangent_continuity_weight == other.tangent_continuity_weight &&
               tangent_backward_penalty == other.tangent_backward_penalty &&
               lsq == other.lsq &&
               local_representative_relative_tolerance == other.local_representative_relative_tolerance;
    }

    bool operator!=(const real_packed_fourier_slice_1d_policy& other) const
    {
        return !(*this == other);
    }

private:
    static bool finite(const T value)
    {
        using std::isfinite;
        return isfinite(value);
    }
};

} // namespace fourier
} // namespace symmetry

#endif
