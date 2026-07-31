#ifndef __STABILITY_ANALYSIS_SPECTRUM_CLASSIFIER_H__
#define __STABILITY_ANALYSIS_SPECTRUM_CLASSIFIER_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "stability_point_result.h"

namespace stability
{
namespace analysis
{

template<class Real>
struct spectrum_classifier_options
{
    stable_halfplane stable = stable_halfplane::left;
    Real stability_boundary_tolerance = Real(1.0e-7);
    Real real_eigenvalue_tolerance = Real(1.0e-7);
    Real conjugate_pair_tolerance = Real(1.0e-6);
    bool require_converged_eigenpairs = true;
    bool require_nonempty_spectrum = true;
    bool require_complete_scan_coverage = true;
};

namespace detail
{

enum class spectral_region
{
    stable,
    neutral,
    unstable
};

template<class Real>
bool finite_complex(const std::complex<Real>& value)
{
    using std::isfinite;
    return isfinite(value.real()) && isfinite(value.imag());
}

template<class Real>
bool close_with_scale(Real left, Real right, Real tolerance)
{
    using std::abs;
    const Real scale = std::max(
        Real(1),
        std::max(abs(left), abs(right)));
    return abs(left - right) <= tolerance*scale;
}

template<class Real>
bool conjugate_match(
    const std::complex<Real>& left,
    const std::complex<Real>& right,
    Real tolerance)
{
    if(left.imag()*right.imag() >= Real(0))
        return false;
    return
        close_with_scale(left.real(), right.real(), tolerance) &&
        close_with_scale(left.imag(), -right.imag(), tolerance);
}

} // namespace detail

template<class Real>
class spectrum_classifier
{
public:
    using real_type = Real;
    using options_type = spectrum_classifier_options<Real>;
    using eigensolver_result_type =
        eigensolvers::eigensolver_result<Real>;
    using result_type = stability_point_result<Real>;

    explicit spectrum_classifier(options_type options = {})
        : options_(std::move(options))
    {
    }

    void set_options(const options_type& options)
    {
        options_ = options;
    }

    const options_type& options() const
    {
        return options_;
    }

    result_type classify(eigensolver_result_type solver_result) const
    {
        result_type result;
        result.eigensolver_status = solver_result.status;
        result.eigenpairs = std::move(solver_result.eigenpairs);

        if(solver_result.status !=
           eigensolvers::eigensolver_status::success)
        {
            result.classification_status =
                spectrum_classification_status::eigensolver_failure;
            result.diagnostic = solver_result.diagnostic.empty()
                ? eigensolvers::eigensolver_status_name(
                      solver_result.status)
                : std::move(solver_result.diagnostic);
            return result;
        }

        if(options_.require_nonempty_spectrum &&
           result.eigenpairs.empty())
        {
            result.classification_status =
                spectrum_classification_status::incomplete;
            result.diagnostic =
                "eigensolver returned an empty spectrum";
            return result;
        }

        if(
            options_.require_complete_scan_coverage &&
            !solver_result.coverage_complete)
        {
            result.classification_status =
                spectrum_classification_status::incomplete;
            std::ostringstream message;
            message
                << "spectral scan coverage is incomplete: "
                << solver_result.scans_succeeded << "/"
                << solver_result.scans_requested
                << " scans succeeded";
            result.diagnostic = message.str();
            return result;
        }

        std::vector<std::complex<Real>> stable_complex;
        std::vector<std::complex<Real>> neutral_complex;
        std::vector<std::complex<Real>> unstable_complex;
        std::vector<std::complex<Real>> unmatched_complex;

        for(const auto& estimate : result.eigenpairs)
        {
            if(options_.require_converged_eigenpairs &&
               !estimate.converged)
            {
                ++result.unclassified_eigenvalues;
                continue;
            }
            if(!detail::finite_complex(estimate.value))
            {
                ++result.unclassified_eigenvalues;
                continue;
            }

            const detail::spectral_region region =
                classify_region(estimate.value.real());
            using std::abs;
            if(abs(estimate.value.imag()) <=
               options_.real_eigenvalue_tolerance)
            {
                increment_real(region, result);
                continue;
            }

            switch(region)
            {
            case detail::spectral_region::stable:
                stable_complex.push_back(estimate.value);
                break;
            case detail::spectral_region::neutral:
                neutral_complex.push_back(estimate.value);
                break;
            case detail::spectral_region::unstable:
                unstable_complex.push_back(estimate.value);
                break;
            }
        }

        result.stable_complex_pairs =
            count_pairs(stable_complex, result, unmatched_complex);
        result.neutral_complex_pairs =
            count_pairs(neutral_complex, result, unmatched_complex);
        result.unstable.complex_pairs =
            count_pairs(unstable_complex, result, unmatched_complex);

        const bool complete =
            result.unclassified_eigenvalues == 0 &&
            result.unmatched_complex_eigenvalues == 0;
        result.classification_status = complete
            ? spectrum_classification_status::complete
            : spectrum_classification_status::incomplete;

        if(!complete)
        {
            std::ostringstream message;
            message
                << "spectrum classification is incomplete: "
                << result.unclassified_eigenvalues
                << " unclassified eigenvalues, "
                << result.unmatched_complex_eigenvalues
                << " unmatched complex eigenvalues";
            if(!unmatched_complex.empty())
            {
                message << " [";
                for(std::size_t index = 0;
                    index < unmatched_complex.size();
                    ++index)
                {
                    if(index != 0)
                        message << ", ";
                    message
                        << "(" << unmatched_complex[index].real()
                        << "," << unmatched_complex[index].imag()
                        << ")";
                }
                message << "]";
            }
            result.diagnostic = message.str();
        }
        else
        {
            result.diagnostic = std::move(solver_result.diagnostic);
        }
        return result;
    }

private:
    options_type options_;

    detail::spectral_region classify_region(Real real_part) const
    {
        const Real unstable_coordinate =
            options_.stable == stable_halfplane::left
                ? real_part
                : -real_part;
        if(unstable_coordinate >
           options_.stability_boundary_tolerance)
            return detail::spectral_region::unstable;
        if(unstable_coordinate <
           -options_.stability_boundary_tolerance)
            return detail::spectral_region::stable;
        return detail::spectral_region::neutral;
    }

    static void increment_real(
        detail::spectral_region region,
        result_type& result)
    {
        switch(region)
        {
        case detail::spectral_region::stable:
            ++result.stable_real;
            break;
        case detail::spectral_region::neutral:
            ++result.neutral_real;
            break;
        case detail::spectral_region::unstable:
            ++result.unstable.real;
            break;
        }
    }

    int count_pairs(
        const std::vector<std::complex<Real>>& values,
        result_type& result,
        std::vector<std::complex<Real>>& unmatched) const
    {
        std::vector<bool> paired(values.size(), false);
        int pair_count = 0;
        for(std::size_t left = 0; left < values.size(); ++left)
        {
            if(paired[left])
                continue;

            std::size_t best = values.size();
            Real best_distance =
                std::numeric_limits<Real>::infinity();
            for(std::size_t right = left + 1;
                right < values.size();
                ++right)
            {
                if(paired[right] ||
                   !detail::conjugate_match(
                       values[left],
                       values[right],
                       options_.conjugate_pair_tolerance))
                    continue;
                using std::abs;
                const Real distance =
                    abs(values[left].real() -
                        values[right].real()) +
                    abs(values[left].imag() +
                        values[right].imag());
                if(distance < best_distance)
                {
                    best = right;
                    best_distance = distance;
                }
            }

            if(best == values.size())
            {
                ++result.unmatched_complex_eigenvalues;
                unmatched.push_back(values[left]);
                paired[left] = true;
                continue;
            }

            paired[left] = true;
            paired[best] = true;
            ++pair_count;
        }
        return pair_count;
    }
};

} // namespace analysis
} // namespace stability

#endif
