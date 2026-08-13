#ifndef __STABILITY_ANALYSIS_EIGENVECTOR_RANK_AGGREGATOR_H__
#define __STABILITY_ANALYSIS_EIGENVECTOR_RANK_AGGREGATOR_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <stability/eigensolvers/eigensolver_result.h>
#include <stability/eigensolvers/ritz_recovery.h>

#include "detail/vector_workspace.h"

namespace stability
{
namespace analysis
{

/**
 * Merges physical Ritz vectors from independent Krylov probes.
 *
 * Equal eigenvalues are retained once per numerically independent
 * eigenvector. This preserves geometric multiplicity while rejecting the
 * same Ritz vector rediscovered by another probe or spectral shift.
 */
template<class VectorSpace>
class eigenvector_rank_aggregator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using real_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;
    using result_type =
        eigensolvers::eigensolver_result<real_type>;
    using estimate_type =
        eigensolvers::eigenpair_estimate<real_type>;
    using storage_type =
        eigensolvers::ritz_vector_storage<vector_space_type>;

    struct options_type
    {
        real_type eigenvalue_absolute_tolerance =
            real_type(1.0e-8);
        real_type eigenvalue_relative_tolerance =
            real_type(1.0e-7);
        real_type independence_tolerance =
            real_type(1.0e-6);
        std::size_t orthogonalization_passes = 2;
    };

    eigenvector_rank_aggregator(
        vector_space_type& vector_space,
        options_type options = {})
        : vector_space_(vector_space),
          options_(std::move(options)),
          candidate_real_(&vector_space_),
          candidate_imaginary_(&vector_space_)
    {
        if(
            options_.eigenvalue_absolute_tolerance < real_type{} ||
            options_.eigenvalue_relative_tolerance < real_type{} ||
            !(options_.independence_tolerance > real_type{}) ||
            options_.orthogonalization_passes == 0)
        {
            throw std::invalid_argument(
                "invalid eigenvector-rank aggregation options");
        }
    }

    void add(
        const result_type& result,
        const storage_type& vectors)
    {
        for(const auto& estimate : result.eigenpairs)
        {
            if(!estimate.converged)
                continue;
            if(estimate.projected_index >= vectors.size())
            {
                throw std::out_of_range(
                    "physical Ritz-vector index is out of range");
            }

            load_candidate(
                vectors,
                estimate.projected_index);
            const real_type input_norm = candidate_norm();
            if(
                !(input_norm > real_type{}) ||
                !std::isfinite(input_norm))
            {
                continue;
            }

            for(std::size_t pass = 0;
                pass < options_.orthogonalization_passes;
                ++pass)
            {
                for(const auto& entry : entries_)
                {
                    if(equivalent(
                           entry.estimate.value,
                           estimate.value))
                    {
                        remove_projection(entry);
                    }
                }
            }

            const real_type independent_norm = candidate_norm();
            if(
                !std::isfinite(independent_norm) ||
                independent_norm <=
                    options_.independence_tolerance*input_norm)
            {
                improve_dependent_estimate(estimate);
                continue;
            }

            vector_space_.scale(
                scalar_type(real_type(1)/independent_norm),
                candidate_real_.get());
            vector_space_.scale(
                scalar_type(real_type(1)/independent_norm),
                candidate_imaginary_.get());
            append(estimate);
        }
    }

    std::vector<estimate_type> estimates() const
    {
        std::vector<estimate_type> result;
        result.reserve(entries_.size());
        for(const auto& entry : entries_)
            result.push_back(entry.estimate);
        return result;
    }

    std::size_t size() const
    {
        return entries_.size();
    }

    template<class Function>
    void for_each_entry(Function&& function) const
    {
        for(const auto& entry : entries_)
        {
            function(
                entry.estimate,
                entry.real->get(),
                entry.imaginary->get());
        }
    }

private:
    struct entry_type
    {
        estimate_type estimate;
        std::unique_ptr<
            detail::vector_workspace<vector_space_type>> real;
        std::unique_ptr<
            detail::vector_workspace<vector_space_type>> imaginary;
    };

    vector_space_type& vector_space_;
    options_type options_;
    detail::vector_workspace<vector_space_type> candidate_real_;
    detail::vector_workspace<vector_space_type> candidate_imaginary_;
    std::vector<entry_type> entries_;

    static ordinal_type ordinal(std::size_t value)
    {
        return static_cast<ordinal_type>(value);
    }

    void load_candidate(
        const storage_type& vectors,
        std::size_t index)
    {
        vector_space_.assign(
            vectors.real(),
            ordinal(vectors.capacity()),
            ordinal(index),
            candidate_real_.get());
        vector_space_.assign(
            vectors.imaginary(),
            ordinal(vectors.capacity()),
            ordinal(index),
            candidate_imaginary_.get());
    }

    real_type candidate_norm() const
    {
        using std::sqrt;
        return sqrt(
            vector_space_.norm_sq(candidate_real_.get()) +
            vector_space_.norm_sq(candidate_imaginary_.get()));
    }

    bool equivalent(
        const std::complex<real_type>& left,
        const std::complex<real_type>& right) const
    {
        using std::abs;
        const real_type scale = std::max(
            real_type(1),
            std::max(abs(left), abs(right)));
        return
            abs(left - right) <=
                options_.eigenvalue_absolute_tolerance +
                options_.eigenvalue_relative_tolerance*scale;
    }

    void remove_projection(const entry_type& entry)
    {
        const scalar_type coefficient_real =
            vector_space_.scalar_prod(
                entry.real->get(),
                candidate_real_.get()) +
            vector_space_.scalar_prod(
                entry.imaginary->get(),
                candidate_imaginary_.get());
        const scalar_type coefficient_imaginary =
            vector_space_.scalar_prod(
                entry.real->get(),
                candidate_imaginary_.get()) -
            vector_space_.scalar_prod(
                entry.imaginary->get(),
                candidate_real_.get());

        vector_space_.add_lin_comb(
            -coefficient_real,
            entry.real->get(),
            scalar_type(1),
            candidate_real_.get());
        vector_space_.add_lin_comb(
            coefficient_imaginary,
            entry.imaginary->get(),
            scalar_type(1),
            candidate_real_.get());
        vector_space_.add_lin_comb(
            -coefficient_imaginary,
            entry.real->get(),
            scalar_type(1),
            candidate_imaginary_.get());
        vector_space_.add_lin_comb(
            -coefficient_real,
            entry.imaginary->get(),
            scalar_type(1),
            candidate_imaginary_.get());
    }

    static bool prefer(
        const estimate_type& candidate,
        const estimate_type& current)
    {
        using std::isfinite;
        if(
            isfinite(candidate.relative_residual) !=
            isfinite(current.relative_residual))
        {
            return isfinite(candidate.relative_residual);
        }
        if(
            isfinite(candidate.relative_residual) &&
            candidate.relative_residual !=
                current.relative_residual)
        {
            return
                candidate.relative_residual <
                current.relative_residual;
        }
        return
            isfinite(candidate.residual) &&
            (!isfinite(current.residual) ||
             candidate.residual < current.residual);
    }

    void improve_dependent_estimate(
        const estimate_type& estimate)
    {
        const auto found = std::find_if(
            entries_.begin(),
            entries_.end(),
            [this, &estimate](const entry_type& entry)
            {
                return equivalent(
                    entry.estimate.value,
                    estimate.value);
            });
        if(
            found != entries_.end() &&
            prefer(estimate, found->estimate))
        {
            const std::size_t index =
                static_cast<std::size_t>(
                    std::distance(entries_.begin(), found));
            found->estimate = estimate;
            found->estimate.projected_index = index;
        }
    }

    void append(const estimate_type& estimate)
    {
        entry_type entry;
        entry.estimate = estimate;
        entry.estimate.projected_index = entries_.size();
        entry.real =
            std::make_unique<
                detail::vector_workspace<vector_space_type>>(
                &vector_space_);
        entry.imaginary =
            std::make_unique<
                detail::vector_workspace<vector_space_type>>(
                &vector_space_);
        vector_space_.assign(
            candidate_real_.get(),
            entry.real->get());
        vector_space_.assign(
            candidate_imaginary_.get(),
            entry.imaginary->get());
        entries_.emplace_back(std::move(entry));
    }
};

} // namespace analysis
} // namespace stability

#endif
