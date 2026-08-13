#ifndef __STABILITY_ANALYSIS_RECYCLED_RITZ_SUBSPACE_H__
#define __STABILITY_ANALYSIS_RECYCLED_RITZ_SUBSPACE_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nmfd/solvers/krylov/operator_apply.h>

#include <stability/eigensolvers/eigensolver_result.h>

#include "detail/vector_workspace.h"

namespace stability
{
namespace analysis
{

template<class Real>
struct recycled_ritz_subspace_options
{
    bool enabled = false;
    std::size_t maximum_vectors = 16;
    Real innovation_weight = Real(0.1);
    Real absolute_residual_tolerance = Real(1.0e-8);
    Real relative_residual_tolerance = Real(0.25);
};

/**
 * Stores converged physical Ritz vectors between nearby linearization points.
 *
 * The previous Arnoldi relation is deliberately not reused after the operator
 * changes. Instead, cached physical Ritz vectors are checked against the new
 * operator and mixed with a fresh probe. This preserves global convergence
 * while supplying a continuation-quality initial subspace.
 */
template<class VectorSpace>
class recycled_ritz_subspace
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using real_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using estimate_type =
        eigensolvers::eigenpair_estimate<real_type>;
    using options_type =
        recycled_ritz_subspace_options<real_type>;

    struct validation_result
    {
        std::vector<std::size_t> accepted_indices;
        std::size_t rejected_vectors = 0;
        std::size_t operator_calls = 0;
    };

    recycled_ritz_subspace(
        vector_space_type& vector_space,
        options_type options = {})
        : vector_space_(vector_space),
          options_(std::move(options)),
          applied_real_(&vector_space_),
          applied_imaginary_(&vector_space_),
          residual_real_(&vector_space_),
          residual_imaginary_(&vector_space_),
          combination_(&vector_space_)
    {
        validate_options(options_);
    }

    void set_options(options_type options)
    {
        validate_options(options);
        options_ = std::move(options);
        trim(committed_);
        trim(staged_);
        if(!options_.enabled)
            clear();
    }

    const options_type& options() const
    {
        return options_;
    }

    bool enabled() const
    {
        return options_.enabled;
    }

    std::size_t size() const
    {
        return committed_.size();
    }

    void clear() const
    {
        committed_.clear();
        staged_.clear();
        transaction_active_ = false;
    }

    void begin_transaction() const
    {
        if(!options_.enabled)
            return;
        if(transaction_active_)
        {
            throw std::logic_error(
                "recycled Ritz-subspace transaction is already active");
        }
        staged_.clear();
        transaction_active_ = true;
    }

    void commit_transaction() const
    {
        if(!options_.enabled)
            return;
        if(!transaction_active_)
            return;
        if(!staged_.empty())
            committed_ = std::move(staged_);
        staged_.clear();
        transaction_active_ = false;
    }

    void rollback_transaction() const
    {
        staged_.clear();
        transaction_active_ = false;
    }

    template<class Aggregator>
    void stage(const Aggregator& source) const
    {
        if(!options_.enabled || source.size() == 0)
            return;

        std::vector<entry_type> candidate;
        candidate.reserve(
            std::min(source.size(), options_.maximum_vectors));
        source.for_each_entry(
            [this, &candidate](
                const estimate_type& estimate,
                const vector_type& real,
                const vector_type& imaginary)
            {
                if(candidate.size() >= options_.maximum_vectors)
                    return;

                // A conjugate pair spans the same real invariant plane.
                const real_type conjugate_tolerance =
                    real_type(64)*
                    std::numeric_limits<real_type>::epsilon()*
                    std::max(real_type(1), std::abs(estimate.value));
                if(estimate.value.imag() < -conjugate_tolerance)
                    return;

                candidate.emplace_back(
                    vector_space_,
                    estimate,
                    real,
                    imaginary);
            });

        if(candidate.empty())
            return;
        if(transaction_active_)
        {
            if(candidate.size() >= staged_.size())
                staged_ = std::move(candidate);
        }
        else
        {
            committed_ = std::move(candidate);
        }
    }

    template<class RealOperator>
    validation_result validate(
        const RealOperator& real_operator) const
    {
        validation_result result;
        if(!options_.enabled)
            return result;
        result.accepted_indices.reserve(committed_.size());

        for(std::size_t index = 0;
            index < committed_.size();
            ++index)
        {
            const entry_type& entry = committed_[index];
            if(!nmfd::solvers::krylov::apply_operator(
                   real_operator,
                   entry.real->get(),
                   applied_real_.get()))
            {
                ++result.rejected_vectors;
                continue;
            }
            ++result.operator_calls;

            const real_type real_norm_sq =
                vector_space_.norm_sq(entry.real->get());
            const real_type imaginary_norm_sq =
                vector_space_.norm_sq(entry.imaginary->get());
            if(imaginary_norm_sq > real_type{})
            {
                if(!nmfd::solvers::krylov::apply_operator(
                       real_operator,
                       entry.imaginary->get(),
                       applied_imaginary_.get()))
                {
                    ++result.rejected_vectors;
                    continue;
                }
                ++result.operator_calls;
            }
            else
            {
                vector_space_.assign_scalar(
                    scalar_type{},
                    applied_imaginary_.get());
            }

            const real_type vector_norm_sq =
                real_norm_sq + imaginary_norm_sq;
            if(
                !std::isfinite(vector_norm_sq) ||
                !(vector_norm_sq > real_type{}))
            {
                ++result.rejected_vectors;
                continue;
            }

            // Eigenvalues move along a continuation curve. Recompute the
            // complex Rayleigh quotient for the current operator before
            // deciding whether the old invariant vector is still useful.
            const real_type real_part =
                (vector_space_.scalar_prod(
                     entry.real->get(),
                     applied_real_.get()) +
                 vector_space_.scalar_prod(
                     entry.imaginary->get(),
                     applied_imaginary_.get()))/
                vector_norm_sq;
            const real_type imaginary_part =
                (vector_space_.scalar_prod(
                     entry.real->get(),
                     applied_imaginary_.get()) -
                 vector_space_.scalar_prod(
                     entry.imaginary->get(),
                     applied_real_.get()))/
                vector_norm_sq;
            vector_space_.assign(
                applied_real_.get(),
                residual_real_.get());
            vector_space_.add_lin_comb(
                scalar_type(-real_part),
                entry.real->get(),
                scalar_type(1),
                residual_real_.get());
            vector_space_.add_lin_comb(
                scalar_type(imaginary_part),
                entry.imaginary->get(),
                scalar_type(1),
                residual_real_.get());

            vector_space_.assign(
                applied_imaginary_.get(),
                residual_imaginary_.get());
            vector_space_.add_lin_comb(
                scalar_type(-imaginary_part),
                entry.real->get(),
                scalar_type(1),
                residual_imaginary_.get());
            vector_space_.add_lin_comb(
                scalar_type(-real_part),
                entry.imaginary->get(),
                scalar_type(1),
                residual_imaginary_.get());

            using std::sqrt;
            const real_type residual = sqrt(
                vector_space_.norm_sq(residual_real_.get()) +
                vector_space_.norm_sq(residual_imaginary_.get()));
            const real_type vector_norm = sqrt(vector_norm_sq);
            const real_type operator_norm = sqrt(
                vector_space_.norm_sq(applied_real_.get()) +
                vector_space_.norm_sq(applied_imaginary_.get()));
            const real_type current_value_norm =
                std::hypot(real_part, imaginary_part);
            const real_type scale = std::max(
                real_type(1),
                operator_norm +
                    current_value_norm*vector_norm);
            if(
                std::isfinite(residual) &&
                residual <=
                    options_.absolute_residual_tolerance +
                    options_.relative_residual_tolerance*scale)
            {
                result.accepted_indices.push_back(index);
            }
            else
            {
                ++result.rejected_vectors;
            }
        }
        return result;
    }

    bool make_seed(
        std::size_t probe_index,
        const vector_type& innovation,
        const std::vector<std::size_t>& accepted_indices,
        vector_type& destination) const
    {
        if(!options_.enabled || accepted_indices.empty())
        {
            vector_space_.assign(innovation, destination);
            return false;
        }

        vector_space_.assign_scalar(
            scalar_type{},
            combination_.get());
        const real_type phase_increment =
            real_type(2.39996322972865332);
        for(std::size_t position = 0;
            position < accepted_indices.size();
            ++position)
        {
            const std::size_t index = accepted_indices[position];
            if(index >= committed_.size())
                continue;
            const entry_type& entry = committed_[index];
            const real_type angle =
                phase_increment*
                real_type((probe_index + 1)*(position + 1));
            vector_space_.add_lin_comb(
                scalar_type(std::cos(angle)),
                entry.real->get(),
                scalar_type(1),
                combination_.get());
            vector_space_.add_lin_comb(
                scalar_type(std::sin(angle)),
                entry.imaginary->get(),
                scalar_type(1),
                combination_.get());
        }

        const real_type combination_norm =
            vector_space_.norm(combination_.get());
        if(
            !std::isfinite(combination_norm) ||
            !(combination_norm > real_type{}))
        {
            vector_space_.assign(innovation, destination);
            return false;
        }
        vector_space_.scale(
            scalar_type(real_type(1)/combination_norm),
            combination_.get());

        vector_space_.assign(innovation, destination);
        const real_type innovation_norm =
            vector_space_.norm(destination);
        const real_type weight = options_.innovation_weight;
        if(
            std::isfinite(innovation_norm) &&
            innovation_norm > real_type{})
        {
            vector_space_.scale(
                scalar_type(weight/innovation_norm),
                destination);
        }
        else
        {
            vector_space_.assign_scalar(
                scalar_type{},
                destination);
        }
        const real_type recycled_weight =
            std::sqrt(std::max(real_type{}, real_type(1)-weight*weight));
        vector_space_.add_lin_comb(
            scalar_type(recycled_weight),
            combination_.get(),
            scalar_type(1),
            destination);

        const real_type result_norm = vector_space_.norm(destination);
        if(
            !std::isfinite(result_norm) ||
            !(result_norm > real_type{}))
        {
            vector_space_.assign(innovation, destination);
            return false;
        }
        vector_space_.scale(
            scalar_type(real_type(1)/result_norm),
            destination);
        return true;
    }

private:
    struct entry_type
    {
        estimate_type estimate;
        std::unique_ptr<detail::vector_workspace<vector_space_type>> real;
        std::unique_ptr<detail::vector_workspace<vector_space_type>> imaginary;

        entry_type(
            vector_space_type& vector_space,
            const estimate_type& estimate_value,
            const vector_type& real_value,
            const vector_type& imaginary_value)
            : estimate(estimate_value),
              real(std::make_unique<
                   detail::vector_workspace<vector_space_type>>(
                       &vector_space)),
              imaginary(std::make_unique<
                   detail::vector_workspace<vector_space_type>>(
                       &vector_space))
        {
            vector_space.assign(real_value, real->get());
            vector_space.assign(imaginary_value, imaginary->get());
        }

        entry_type(entry_type&&) = default;
        entry_type& operator=(entry_type&&) = default;
        entry_type(const entry_type&) = delete;
        entry_type& operator=(const entry_type&) = delete;
    };

    vector_space_type& vector_space_;
    options_type options_;
    mutable std::vector<entry_type> committed_;
    mutable std::vector<entry_type> staged_;
    mutable bool transaction_active_ = false;
    mutable detail::vector_workspace<vector_space_type> applied_real_;
    mutable detail::vector_workspace<vector_space_type> applied_imaginary_;
    mutable detail::vector_workspace<vector_space_type> residual_real_;
    mutable detail::vector_workspace<vector_space_type> residual_imaginary_;
    mutable detail::vector_workspace<vector_space_type> combination_;

    static void validate_options(const options_type& options)
    {
        if(
            options.maximum_vectors == 0 ||
            !std::isfinite(options.innovation_weight) ||
            !(options.innovation_weight > real_type{}) ||
            !(options.innovation_weight <= real_type(1)) ||
            !std::isfinite(options.absolute_residual_tolerance) ||
            options.absolute_residual_tolerance < real_type{} ||
            !std::isfinite(options.relative_residual_tolerance) ||
            options.relative_residual_tolerance < real_type{})
        {
            throw std::invalid_argument(
                "invalid recycled Ritz-subspace options");
        }
    }

    void trim(std::vector<entry_type>& entries) const
    {
        if(entries.size() > options_.maximum_vectors)
        {
            entries.erase(
                entries.begin() +
                    static_cast<std::ptrdiff_t>(
                        options_.maximum_vectors),
                entries.end());
        }
    }
};

} // namespace analysis
} // namespace stability

#endif
