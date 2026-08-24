#ifndef __STABILITY_ANALYSIS_SPECTRUM_SCAN_AGGREGATOR_H__
#define __STABILITY_ANALYSIS_SPECTRUM_SCAN_AGGREGATOR_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
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
struct spectrum_scan_aggregation_options
{
    Real absolute_tolerance = Real(1.0e-8);
    Real relative_tolerance = Real(1.0e-7);
    std::size_t minimum_successful_scans = 1;
    std::size_t minimum_eigenpairs = 1;
    bool require_all_scans = true;
    std::size_t probe_count = 1;
    std::size_t minimum_successful_probes = 1;
    bool require_all_probes = true;
    Real eigenvector_independence_tolerance = Real(1.0e-6);
    std::size_t eigenvector_orthogonalization_passes = 2;
};

template<class Real>
class spectrum_scan_aggregator
{
public:
    using options_type =
        spectrum_scan_aggregation_options<Real>;
    using result_type =
        eigensolvers::eigensolver_result<Real>;
    using estimate_type =
        eigensolvers::eigenpair_estimate<Real>;

    spectrum_scan_aggregator(
        std::size_t scan_count,
        options_type options = {})
        : scan_count_(scan_count),
          options_(std::move(options))
    {
    }

    void add(
        result_type scan_result,
        std::string label = {})
    {
        result_.iterations += scan_result.iterations;
        result_.restarts += scan_result.restarts;
        result_.operator_calls += scan_result.operator_calls;
        result_.inner_solver_calls +=
            scan_result.inner_solver_calls;
        result_.coverage_recoveries +=
            scan_result.coverage_recoveries;
        result_.effective_subspace_dimension = std::max(
            result_.effective_subspace_dimension,
            scan_result.effective_subspace_dimension);

        if(
            scan_result.succeeded() &&
            scan_result.coverage_complete)
        {
            ++successful_scans_;
            append_success(
                label,
                scan_result.diagnostic);
            merge_scan(std::move(scan_result.eigenpairs));
            return;
        }

        if(first_failure_status_ ==
           eigensolvers::eigensolver_status::success)
        {
            first_failure_status_ =
                scan_result.status ==
                    eigensolvers::eigensolver_status::success
                ? eigensolvers::eigensolver_status::no_convergence
                : scan_result.status;
        }
        append_failure(
            std::move(label),
            scan_result.status ==
                eigensolvers::eigensolver_status::success
                ? eigensolvers::eigensolver_status::no_convergence
                : scan_result.status,
            std::move(scan_result.diagnostic));
    }

    result_type finish()
    {
        result_.scans_requested = scan_count_;
        result_.scans_succeeded = successful_scans_;
        result_.coverage_complete =
            successful_scans_ == scan_count_;

        std::sort(
            result_.eigenpairs.begin(),
            result_.eigenpairs.end(),
            [](const estimate_type& left, const estimate_type& right)
            {
                if(left.value.real() != right.value.real())
                    return left.value.real() > right.value.real();
                return left.value.imag() > right.value.imag();
            });

        const bool enough_scans =
            successful_scans_ >=
                options_.minimum_successful_scans;
        const bool accepted_coverage =
            !options_.require_all_scans ||
            result_.coverage_complete;
        const bool enough_eigenpairs =
            result_.eigenpairs.size() >=
                options_.minimum_eigenpairs;
        if(
            scan_count_ == 0 ||
            !enough_scans ||
            !accepted_coverage ||
            !enough_eigenpairs)
        {
            result_.status =
                first_failure_status_ !=
                    eigensolvers::eigensolver_status::success
                ? first_failure_status_
                : eigensolvers::eigensolver_status::no_convergence;
        }
        else
        {
            result_.status =
                eigensolvers::eigensolver_status::success;
        }

        std::ostringstream diagnostic;
        diagnostic
            << "spectral scans: " << successful_scans_
            << "/" << scan_count_ << " succeeded, "
            << result_.eigenpairs.size()
            << " physical eigenpairs after cross-scan merge "
            << "(minimum " << options_.minimum_eigenpairs << ")";
        if(!enough_eigenpairs && !result_.eigenpairs.empty())
        {
            diagnostic << "; recovered values = [";
            for(std::size_t index = 0;
                index < result_.eigenpairs.size();
                ++index)
            {
                if(index != 0)
                    diagnostic << ", ";
                diagnostic << result_.eigenpairs[index].value;
            }
            diagnostic << ']';
        }
        if(!failures_.empty())
            diagnostic << "; failures: " << failures_;
        if(!successes_.empty())
            diagnostic << "; scan details: " << successes_;
        result_.diagnostic = diagnostic.str();
        return std::move(result_);
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
        if(candidate.converged != current.converged)
            return candidate.converged;
        if(
            finite(candidate.relative_residual) !=
            finite(current.relative_residual))
        {
            return finite(candidate.relative_residual);
        }
        if(
            finite(candidate.relative_residual) &&
            candidate.relative_residual !=
                current.relative_residual)
        {
            return
                candidate.relative_residual <
                current.relative_residual;
        }
        if(
            finite(candidate.residual) !=
            finite(current.residual))
        {
            return finite(candidate.residual);
        }
        return
            finite(candidate.residual) &&
            candidate.residual < current.residual;
    }

    void merge_scan(std::vector<estimate_type> estimates)
    {
        std::vector<bool> matched(
            result_.eigenpairs.size(),
            false);
        for(auto& estimate : estimates)
        {
            std::size_t found = result_.eigenpairs.size();
            for(std::size_t index = 0;
                index < result_.eigenpairs.size();
                ++index)
            {
                if(
                    !matched[index] &&
                    equivalent(
                        result_.eigenpairs[index],
                        estimate))
                {
                    found = index;
                    break;
                }
            }
            if(found == result_.eigenpairs.size())
            {
                result_.eigenpairs.emplace_back(
                    std::move(estimate));
                matched.push_back(true);
            }
            else
            {
                matched[found] = true;
                if(prefer(
                       estimate,
                       result_.eigenpairs[found]))
                {
                    result_.eigenpairs[found] =
                        std::move(estimate);
                }
            }
        }
    }

    void append_failure(
        std::string label,
        eigensolvers::eigensolver_status status,
        std::string diagnostic)
    {
        if(!failures_.empty())
            failures_ += " | ";
        if(!label.empty())
            failures_ += label + ": ";
        failures_ +=
            eigensolvers::eigensolver_status_name(status);
        if(!diagnostic.empty())
            failures_ += " (" + diagnostic + ")";
    }

    void append_success(
        const std::string& label,
        const std::string& diagnostic)
    {
        if(diagnostic.empty())
            return;
        if(!successes_.empty())
            successes_ += " | ";
        if(!label.empty())
            successes_ += label + ": ";
        successes_ += diagnostic;
    }

    std::size_t scan_count_;
    options_type options_;
    result_type result_;
    std::size_t successful_scans_ = 0;
    eigensolvers::eigensolver_status first_failure_status_ =
        eigensolvers::eigensolver_status::success;
    std::string failures_;
    std::string successes_;
};

} // namespace analysis
} // namespace stability

#endif
