#ifndef __CONTINUATION_INITIAL_TANGENT_CANDIDATES_H__
#define __CONTINUATION_INITIAL_TANGENT_CANDIDATES_H__

#include <limits>
#include <stdexcept>

#include <common/scalar_math.h>

namespace continuation
{

template<class T>
struct shifted_newton_pair
{
    T d_lambda = T(0);
    T lambda_plus = T(0);
    T lambda_minus = T(0);
    bool plus_converged = false;
    bool minus_converged = false;
};

enum class secant_candidate_kind
{
    two_sided,
    plus_one_sided,
    minus_one_sided
};

template<class T>
struct tangent_candidate_quality
{
    const char* method = "unknown";
    bool solved = false;
    bool valid = false;
    T tangent_residual = T(0);
    T row_residual_abs = T(0);
    T pre_norm = T(0);
    T lambda_s = T(0);
    T orientation = T(0);
    T residual_tol = T(0);
    T row_residual_tol = T(0);
    T score = T(0);
    T chart_progress_ratio = T(1);
    T chart_displacement_ratio = T(0);
};

template<class T>
class tangent_candidate_selector
{
public:
    bool consider(const tangent_candidate_quality<T>& candidate)
    {
        if(!candidate.valid ||
           (have_best_ && !(candidate.score < best_.score)))
        {
            return false;
        }
        best_ = candidate;
        have_best_ = true;
        return true;
    }

    bool has_candidate() const
    {
        return have_best_;
    }

    const tangent_candidate_quality<T>& best() const
    {
        if(!have_best_)
        {
            throw std::logic_error("no tangent candidate has been selected");
        }
        return best_;
    }

private:
    bool have_best_ = false;
    tangent_candidate_quality<T> best_;
};

template<class T>
struct projected_tangent_quality_policy
{
    T maximum_pre_normalization_norm = T(50);
    T minimum_orientation = T(5.0e-2);
    T residual_tolerance_floor = T(1.0e-3);
    T residual_tolerance_ceiling = T(5.0e-3);
    T direct_tangent_residual_tolerance = T(5.0e-3);
    T pre_norm_score_weight = T(1.0e-3);

    void validate() const
    {
        const bool finite =
            common::scalar_math::isfinite(maximum_pre_normalization_norm) &&
            common::scalar_math::isfinite(minimum_orientation) &&
            common::scalar_math::isfinite(residual_tolerance_floor) &&
            common::scalar_math::isfinite(residual_tolerance_ceiling) &&
            common::scalar_math::isfinite(direct_tangent_residual_tolerance) &&
            common::scalar_math::isfinite(pre_norm_score_weight);
        if(!finite || maximum_pre_normalization_norm <= T(0) ||
           minimum_orientation < T(0) || residual_tolerance_floor < T(0) ||
           residual_tolerance_ceiling < residual_tolerance_floor ||
           direct_tangent_residual_tolerance < T(0) || pre_norm_score_weight < T(0))
        {
            throw std::invalid_argument("invalid projected tangent quality policy");
        }
    }
};

template<class T>
bool projected_tangent_quality_is_acceptable(
    const tangent_candidate_quality<T>& quality,
    const projected_tangent_quality_policy<T>& policy)
{
    const T orientation_abs = common::scalar_math::abs(quality.orientation);
    const bool finite =
        common::scalar_math::isfinite(quality.tangent_residual) &&
        common::scalar_math::isfinite(quality.row_residual_abs) &&
        common::scalar_math::isfinite(quality.pre_norm) &&
        common::scalar_math::isfinite(quality.lambda_s) &&
        common::scalar_math::isfinite(quality.orientation);

    return quality.solved && finite && quality.pre_norm > T(0) &&
           quality.pre_norm <= policy.maximum_pre_normalization_norm &&
           quality.tangent_residual <= quality.residual_tol &&
           quality.row_residual_abs <= quality.row_residual_tol &&
           orientation_abs >= policy.minimum_orientation;
}

template<class T>
T projected_tangent_candidate_score(
    const tangent_candidate_quality<T>& quality,
    const projected_tangent_quality_policy<T>& policy)
{
    if(!quality.solved || !(quality.pre_norm > T(0)))
    {
        return std::numeric_limits<T>::max();
    }
    const T orientation_abs = common::scalar_math::abs(quality.orientation);
    const T orientation_penalty = orientation_abs < T(1) ? T(1) - orientation_abs : T(0);
    return quality.tangent_residual/(quality.residual_tol + T(1.0e-30)) +
           quality.row_residual_abs/(quality.row_residual_tol + T(1.0e-30)) +
           policy.pre_norm_score_weight*quality.pre_norm + orientation_penalty;
}

} // namespace continuation

#endif
