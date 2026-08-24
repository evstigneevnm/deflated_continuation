#ifndef __STABILITY_ANALYSIS_VALIDATED_SPECTRUM_UNION_H__
#define __STABILITY_ANALYSIS_VALIDATED_SPECTRUM_UNION_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <stability/eigensolvers/eigensolver_result.h>

namespace stability
{
namespace analysis
{

template<class Real>
class validated_spectrum_union
{
public:
    using result_type = eigensolvers::eigensolver_result<Real>;
    using estimate_type = eigensolvers::eigenpair_estimate<Real>;

    struct options_type
    {
        Real absolute_tolerance = Real(1.0e-8);
        Real relative_tolerance = Real(1.0e-7);
    };

    explicit validated_spectrum_union(options_type options = {})
        : options_(std::move(options))
    {
    }

    void set_options(options_type options)
    {
        options_ = std::move(options);
        reset();
    }

    void reset()
    {
        estimates_.clear();
        usable_results_ = 0;
        complete_results_ = 0;
    }

    bool add(const result_type& result)
    {
        const bool complete =
            result.succeeded() && result.coverage_complete;
        const bool aggregate_undercoverage =
            result.status == eigensolvers::eigensolver_status::
                                 no_convergence &&
            result.scans_requested != 0 &&
            result.scans_succeeded == result.scans_requested;
        if(!complete && !aggregate_undercoverage)
            return false;

        std::vector<estimate_type> usable;
        usable.reserve(result.eigenpairs.size());
        for(const auto& estimate : result.eigenpairs)
        {
            if(estimate.converged && finite(estimate.value.real()) &&
               finite(estimate.value.imag()))
            {
                usable.push_back(estimate);
            }
        }
        if(usable.empty())
            return false;

        merge_result(usable);
        ++usable_results_;
        if(complete)
            ++complete_results_;
        return true;
    }

    result_type finish(
        std::size_t minimum_results,
        std::size_t minimum_eigenpairs) const
    {
        result_type result;
        result.scans_requested = usable_results_;
        result.scans_succeeded = complete_results_;
        result.eigenpairs = estimates_;
        std::sort(
            result.eigenpairs.begin(),
            result.eigenpairs.end(),
            [](const estimate_type& left, const estimate_type& right)
            {
                if(left.value.real() != right.value.real())
                    return left.value.real() > right.value.real();
                return left.value.imag() > right.value.imag();
            });

        const bool enough_results =
            usable_results_ >= minimum_results;
        const bool has_complete_result = complete_results_ != 0;
        const bool enough_eigenpairs =
            result.eigenpairs.size() >= minimum_eigenpairs;
        result.status =
            enough_results && has_complete_result && enough_eigenpairs
            ? eigensolvers::eigensolver_status::success
            : eigensolvers::eigensolver_status::no_convergence;
        result.coverage_complete = result.succeeded();

        std::ostringstream diagnostic;
        diagnostic
            << "validated independent-spectrum union: "
            << usable_results_ << " usable result(s), "
            << complete_results_ << " complete result(s), "
            << result.eigenpairs.size() << " distinct physical "
               "eigenpair(s); required results = "
            << minimum_results << ", required eigenpairs = "
            << minimum_eigenpairs;
        result.diagnostic = diagnostic.str();
        return result;
    }

    std::size_t usable_results() const
    {
        return usable_results_;
    }

private:
    static bool finite(Real value)
    {
        using std::isfinite;
        return isfinite(value);
    }

    bool equivalent(
        const estimate_type& left,
        const estimate_type& right) const
    {
        using std::abs;
        const Real scale = std::max(
            Real(1),
            std::max(abs(left.value), abs(right.value)));
        return
            abs(left.value - right.value) <=
            options_.absolute_tolerance +
                options_.relative_tolerance*scale;
    }

    static bool prefer(
        const estimate_type& candidate,
        const estimate_type& current)
    {
        if(finite(candidate.relative_residual) !=
           finite(current.relative_residual))
        {
            return finite(candidate.relative_residual);
        }
        if(finite(candidate.relative_residual) &&
           candidate.relative_residual != current.relative_residual)
        {
            return candidate.relative_residual <
                current.relative_residual;
        }
        if(finite(candidate.residual) != finite(current.residual))
            return finite(candidate.residual);
        return
            finite(candidate.residual) &&
            candidate.residual < current.residual;
    }

    void merge_result(const std::vector<estimate_type>& estimates)
    {
        std::vector<bool> matched(estimates_.size(), false);
        for(const auto& estimate : estimates)
        {
            std::size_t found = estimates_.size();
            for(std::size_t index = 0; index < estimates_.size(); ++index)
            {
                if(!matched[index] &&
                   equivalent(estimates_[index], estimate))
                {
                    found = index;
                    break;
                }
            }
            if(found == estimates_.size())
            {
                estimates_.push_back(estimate);
                matched.push_back(true);
            }
            else
            {
                matched[found] = true;
                if(prefer(estimate, estimates_[found]))
                    estimates_[found] = estimate;
            }
        }
    }

    options_type options_;
    std::vector<estimate_type> estimates_;
    std::size_t usable_results_ = 0;
    std::size_t complete_results_ = 0;
};

} // namespace analysis
} // namespace stability

#endif // __STABILITY_ANALYSIS_VALIDATED_SPECTRUM_UNION_H__
