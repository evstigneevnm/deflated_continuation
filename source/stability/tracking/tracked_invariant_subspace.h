#ifndef STABILITY_TRACKING_TRACKED_INVARIANT_SUBSPACE_H
#define STABILITY_TRACKING_TRACKED_INVARIANT_SUBSPACE_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nmfd/operations/linalg/host_small_dense_backend.h>
#include <nmfd/solvers/krylov/operator_apply.h>

#include <stability/analysis/detail/vector_workspace.h>
#include <stability/eigensolvers/eigensolver_result.h>

#include "principal_angles.h"

namespace stability
{
namespace tracking
{

template<class Real>
struct tracked_invariant_subspace_options
{
    bool enabled = false;
    std::size_t maximum_dimension = 16;
    std::size_t maximum_seed_vectors = 8;
    std::size_t coverage_recovery_maximum_seed_vectors = 0;
    std::size_t orthogonalization_passes = 2;
    Real seed_innovation_weight = Real(0.1);
    Real dependence_tolerance = Real(1.0e-8);
    Real minimum_retained_residual_ratio = Real(0.05);
    Real absolute_invariance_tolerance = Real(1.0e-8);
    Real relative_invariance_tolerance = Real(0.25);
    Real real_eigenvalue_tolerance = Real(1.0e-8);
    Real eigenvalue_group_tolerance = Real(1.0e-6);
    Real principal_angle_rank_tolerance = Real(1.0e-8);
};

/**
 * Transactional real invariant-subspace state used along a parameterized
 * operator curve. Complex Ritz vectors are represented by their real and
 * imaginary columns, so a conjugate pair contributes one real plane.
 */
template<class VectorSpace>
class tracked_invariant_subspace
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using real_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using estimate_type =
        eigensolvers::eigenpair_estimate<real_type>;
    using options_type =
        tracked_invariant_subspace_options<real_type>;
    using matrix_type =
        nmfd::operations::linalg::host_dense_matrix<real_type>;
    using overlap_result_type = principal_angle_result<real_type>;

    static_assert(
        std::is_floating_point<scalar_type>::value,
        "tracked invariant subspaces require a real floating-point "
        "vector space");

    struct column_descriptor
    {
        std::complex<real_type> source_value{};
        bool imaginary_component = false;
    };

    struct validation_result
    {
        struct group_result
        {
            std::vector<std::size_t> columns;
            bool accepted = false;
            real_type residual_frobenius =
                std::numeric_limits<real_type>::quiet_NaN();
            real_type relative_residual =
                std::numeric_limits<real_type>::quiet_NaN();
        };

        bool accepted = false;
        std::size_t dimension = 0;
        std::vector<std::size_t> accepted_indices;
        std::vector<group_result> groups;
        std::size_t operator_calls = 0;
        real_type residual_frobenius =
            std::numeric_limits<real_type>::quiet_NaN();
        real_type relative_residual =
            std::numeric_limits<real_type>::quiet_NaN();
        matrix_type projected_operator;
        std::vector<real_type> column_residuals;
        std::string diagnostic;
    };

    tracked_invariant_subspace(
        vector_space_type& vector_space,
        options_type options = {})
        : vector_space_(vector_space),
          options_(std::move(options)),
          candidate_(&vector_space_)
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

    std::size_t seed_size() const
    {
        return seed_basis().size();
    }

    bool empty() const
    {
        return committed_.empty();
    }

    void clear() const
    {
        committed_.clear();
        staged_.clear();
        pending_overlap_.reset();
        last_committed_overlap_.reset();
        pending_retained_columns_ = 0;
        last_committed_retained_columns_ = 0;
        transaction_active_ = false;
    }

    void begin_transaction() const
    {
        if(!options_.enabled)
            return;
        if(transaction_active_)
        {
            throw std::logic_error(
                "tracked invariant-subspace transaction is already active");
        }
        staged_.clear();
        pending_overlap_.reset();
        pending_retained_columns_ = 0;
        transaction_active_ = true;
    }

    void commit_transaction() const
    {
        if(!options_.enabled || !transaction_active_)
            return;
        if(!staged_.empty())
        {
            committed_ = std::move(staged_);
            last_committed_overlap_ = pending_overlap_;
            last_committed_retained_columns_ =
                pending_retained_columns_;
        }
        staged_.clear();
        pending_overlap_.reset();
        pending_retained_columns_ = 0;
        transaction_active_ = false;
    }

    void rollback_transaction() const
    {
        staged_.clear();
        pending_overlap_.reset();
        pending_retained_columns_ = 0;
        transaction_active_ = false;
    }

    template<class Aggregator>
    std::size_t stage(const Aggregator& source) const
    {
        if(!options_.enabled || source.size() == 0)
            return 0;

        basis_type candidate;
        candidate.reserve(options_.maximum_dimension);
        source.for_each_entry(
            [this, &candidate](
                const estimate_type& estimate,
                const vector_type& real,
                const vector_type& imaginary)
            {
                if(candidate.size() >= options_.maximum_dimension)
                    return;

                const real_type value_scale = std::max(
                    real_type(1),
                    std::abs(estimate.value));
                const bool complex_pair =
                    std::abs(estimate.value.imag()) >
                        options_.real_eigenvalue_tolerance*value_scale;

                append_orthonormal(
                    candidate,
                    real,
                    {estimate.value, false},
                    options_.dependence_tolerance);
                if(
                    complex_pair &&
                    candidate.size() < options_.maximum_dimension)
                {
                    append_orthonormal(
                        candidate,
                        imaginary,
                        {estimate.value, true},
                        options_.dependence_tolerance);
                }
            });

        if(candidate.empty())
            return 0;

        const std::size_t recovered_dimension = candidate.size();
        const basis_type& previous_basis = seed_basis();
        for(const auto& previous : previous_basis)
        {
            append_orthonormal(
                candidate,
                previous.vector->get(),
                previous.descriptor,
                options_.minimum_retained_residual_ratio);
        }
        const std::size_t unmatched_previous_columns =
            candidate.size() - recovered_dimension;
        if(unmatched_previous_columns != 0)
        {
            std::rotate(
                candidate.begin(),
                candidate.begin() + static_cast<std::ptrdiff_t>(
                    recovered_dimension),
                candidate.end());
        }
        trim(candidate);
        const std::size_t retained_columns = std::min(
            unmatched_previous_columns,
            candidate.size());

        const std::size_t staged_dimension = candidate.size();
        std::optional<overlap_result_type> candidate_overlap;
        if(!committed_.empty())
            candidate_overlap = overlap_between(committed_, candidate);
        if(transaction_active_)
        {
            staged_ = std::move(candidate);
            pending_overlap_ = std::move(candidate_overlap);
            pending_retained_columns_ = retained_columns;
        }
        else
        {
            committed_ = std::move(candidate);
            last_committed_overlap_ = std::move(candidate_overlap);
            last_committed_retained_columns_ = retained_columns;
        }
        return staged_dimension;
    }

    template<class RealOperator>
    validation_result validate(
        const RealOperator& real_operator) const
    {
        validation_result result;
        const basis_type& basis = seed_basis();
        result.dimension = basis.size();
        if(!options_.enabled)
        {
            result.diagnostic = "invariant-subspace tracking is disabled";
            return result;
        }
        if(basis.empty())
        {
            result.diagnostic = "tracked invariant subspace is empty";
            return result;
        }

        std::vector<std::unique_ptr<workspace_type>> applied;
        applied.reserve(basis.size());
        real_type applied_norm_sq = real_type{};
        for(const auto& column : basis)
        {
            auto value = std::make_unique<workspace_type>(&vector_space_);
            if(!nmfd::solvers::krylov::apply_operator(
                   real_operator,
                   column.vector->get(),
                   value->get()))
            {
                result.diagnostic =
                    "operator application failed while validating the "
                    "tracked invariant subspace";
                return result;
            }
            ++result.operator_calls;
            applied_norm_sq += vector_space_.norm_sq(value->get());
            applied.emplace_back(std::move(value));
        }

        const std::size_t dimension = basis.size();
        result.projected_operator.resize(dimension, dimension);
        for(std::size_t col = 0; col < dimension; ++col)
        {
            for(std::size_t row = 0; row < dimension; ++row)
            {
                result.projected_operator(row, col) =
                    static_cast<real_type>(
                        vector_space_.scalar_prod(
                            basis[row].vector->get(),
                            applied[col]->get()));
            }
        }

        result.column_residuals.reserve(dimension);
        real_type residual_norm_sq = real_type{};
        for(std::size_t col = 0; col < dimension; ++col)
        {
            workspace_type residual(&vector_space_);
            vector_space_.assign(applied[col]->get(), residual.get());
            for(std::size_t row = 0; row < dimension; ++row)
            {
                vector_space_.add_lin_comb(
                    scalar_type(-result.projected_operator(row, col)),
                    basis[row].vector->get(),
                    scalar_type(1),
                    residual.get());
            }
            const real_type column_residual =
                vector_space_.norm(residual.get());
            result.column_residuals.push_back(column_residual);
            residual_norm_sq += column_residual*column_residual;
        }

        using std::sqrt;
        result.residual_frobenius = sqrt(residual_norm_sq);
        const real_type applied_norm = sqrt(applied_norm_sq);
        result.relative_residual =
            result.residual_frobenius/
            std::max(real_type(1), applied_norm);

        std::vector<bool> grouped(dimension, false);
        for(std::size_t first = 0; first < dimension; ++first)
        {
            if(grouped[first])
                continue;
            typename validation_result::group_result group;
            for(std::size_t candidate = first;
                candidate < dimension;
                ++candidate)
            {
                if(
                    !grouped[candidate] &&
                    equivalent_source_value(
                        basis[first].descriptor.source_value,
                        basis[candidate].descriptor.source_value))
                {
                    grouped[candidate] = true;
                    group.columns.push_back(candidate);
                }
            }

            real_type group_applied_norm_sq = real_type{};
            real_type group_residual_norm_sq = real_type{};
            for(const std::size_t col : group.columns)
            {
                group_applied_norm_sq +=
                    vector_space_.norm_sq(applied[col]->get());
                workspace_type residual(&vector_space_);
                vector_space_.assign(applied[col]->get(), residual.get());
                for(const std::size_t row : group.columns)
                {
                    vector_space_.add_lin_comb(
                        scalar_type(-result.projected_operator(row, col)),
                        basis[row].vector->get(),
                        scalar_type(1),
                        residual.get());
                }
                const real_type residual_norm =
                    vector_space_.norm(residual.get());
                group_residual_norm_sq += residual_norm*residual_norm;
            }

            group.residual_frobenius = sqrt(group_residual_norm_sq);
            const real_type group_applied_norm =
                sqrt(group_applied_norm_sq);
            group.relative_residual =
                group.residual_frobenius/
                std::max(real_type(1), group_applied_norm);
            const real_type group_threshold =
                options_.absolute_invariance_tolerance +
                options_.relative_invariance_tolerance*
                    std::max(real_type(1), group_applied_norm);
            group.accepted =
                std::isfinite(group.residual_frobenius) &&
                group.residual_frobenius <= group_threshold;
            if(group.accepted)
            {
                result.accepted_indices.insert(
                    result.accepted_indices.end(),
                    group.columns.begin(),
                    group.columns.end());
            }
            result.groups.push_back(std::move(group));
        }
        result.accepted = !result.accepted_indices.empty();

        std::ostringstream diagnostic;
        diagnostic
            << "dimension=" << result.dimension
            << ", accepted_dimension="
            << result.accepted_indices.size()
            << ", accepted_groups="
            << std::count_if(
                   result.groups.begin(),
                   result.groups.end(),
                   [](const auto& group)
                   {
                       return group.accepted;
                   })
            << '/' << result.groups.size()
            << ", ||JQ-Q(Q^T JQ)||_F="
            << result.residual_frobenius
            << ", relative residual=" << result.relative_residual;
        result.diagnostic = diagnostic.str();
        return result;
    }

    overlap_result_type overlap_with(
        const tracked_invariant_subspace& other) const
    {
        return overlap_between(committed_, other.committed_);
    }

    const std::optional<overlap_result_type>& pending_overlap() const
    {
        return pending_overlap_;
    }

    const std::optional<overlap_result_type>& last_committed_overlap() const
    {
        return last_committed_overlap_;
    }

    std::size_t pending_retained_columns() const
    {
        return pending_retained_columns_;
    }

    std::size_t last_committed_retained_columns() const
    {
        return last_committed_retained_columns_;
    }

    void copy_column(
        std::size_t index,
        vector_type& destination) const
    {
        if(index >= seed_basis().size())
            throw std::out_of_range("tracked subspace column index");
        vector_space_.assign(
            seed_basis()[index].vector->get(),
            destination);
    }

    bool make_seed(
        std::size_t column_index,
        const vector_type& innovation,
        vector_type& destination) const
    {
        std::vector<std::size_t> columns(seed_basis().size());
        for(std::size_t index = 0; index < columns.size(); ++index)
            columns[index] = index;
        return make_seed(
            column_index,
            innovation,
            columns,
            destination);
    }

    bool make_seed(
        std::size_t column_index,
        const vector_type& innovation,
        const std::vector<std::size_t>& available_columns,
        vector_type& destination) const
    {
        if(
            !options_.enabled ||
            column_index >= available_columns.size())
        {
            vector_space_.assign(innovation, destination);
            return false;
        }
        const std::size_t basis_index = available_columns[column_index];
        if(basis_index >= seed_basis().size())
            throw std::out_of_range("tracked seed column index");

        vector_space_.assign(innovation, candidate_.get());
        for(std::size_t pass = 0;
            pass < options_.orthogonalization_passes;
            ++pass)
        {
            for(const std::size_t index : available_columns)
            {
                if(index >= seed_basis().size())
                    throw std::out_of_range("tracked seed column index");
                const auto& column = seed_basis()[index];
                const scalar_type coefficient =
                    vector_space_.scalar_prod(
                        column.vector->get(),
                        candidate_.get());
                vector_space_.add_lin_comb(
                    -coefficient,
                    column.vector->get(),
                    scalar_type(1),
                    candidate_.get());
            }
        }

        vector_space_.assign(
            seed_basis()[basis_index].vector->get(),
            destination);
        const real_type innovation_norm =
            vector_space_.norm(candidate_.get());
        if(
            std::isfinite(innovation_norm) &&
            innovation_norm > real_type{})
        {
            const real_type weight = options_.seed_innovation_weight;
            vector_space_.scale(
                scalar_type(std::sqrt(
                    std::max(
                        real_type{},
                        real_type(1) - weight*weight))),
                destination);
            vector_space_.add_lin_comb(
                scalar_type(weight/innovation_norm),
                candidate_.get(),
                scalar_type(1),
                destination);
        }
        const real_type norm = vector_space_.norm(destination);
        if(!std::isfinite(norm) || !(norm > real_type{}))
        {
            vector_space_.assign(innovation, destination);
            return false;
        }
        vector_space_.scale(
            scalar_type(real_type(1)/norm),
            destination);
        return true;
    }

    const column_descriptor& descriptor(std::size_t index) const
    {
        if(index >= seed_basis().size())
            throw std::out_of_range("tracked subspace descriptor index");
        return seed_basis()[index].descriptor;
    }

private:
    using workspace_type =
        analysis::detail::vector_workspace<vector_space_type>;

    struct basis_column
    {
        std::unique_ptr<workspace_type> vector;
        column_descriptor descriptor;

        basis_column(
            vector_space_type& vector_space,
            const vector_type& value,
            column_descriptor descriptor_value)
            : vector(std::make_unique<workspace_type>(&vector_space)),
              descriptor(std::move(descriptor_value))
        {
            vector_space.assign(value, vector->get());
        }

        basis_column(basis_column&&) = default;
        basis_column& operator=(basis_column&&) = default;
        basis_column(const basis_column&) = delete;
        basis_column& operator=(const basis_column&) = delete;
    };

    using basis_type = std::vector<basis_column>;

    vector_space_type& vector_space_;
    options_type options_;
    mutable basis_type committed_;
    mutable basis_type staged_;
    mutable bool transaction_active_ = false;
    mutable workspace_type candidate_;
    mutable std::optional<overlap_result_type> pending_overlap_;
    mutable std::optional<overlap_result_type> last_committed_overlap_;
    mutable std::size_t pending_retained_columns_ = 0;
    mutable std::size_t last_committed_retained_columns_ = 0;

    const basis_type& seed_basis() const
    {
        return transaction_active_ && !staged_.empty()
            ? staged_
            : committed_;
    }

    overlap_result_type overlap_between(
        const basis_type& left,
        const basis_type& right) const
    {
        matrix_type overlap(left.size(), right.size());
        for(std::size_t col = 0; col < right.size(); ++col)
        {
            for(std::size_t row = 0; row < left.size(); ++row)
            {
                overlap(row, col) = static_cast<real_type>(
                    vector_space_.scalar_prod(
                        left[row].vector->get(),
                        right[col].vector->get()));
            }
        }
        return principal_angles(
            overlap,
            options_.principal_angle_rank_tolerance);
    }

    void append_orthonormal(
        basis_type& basis,
        const vector_type& source,
        column_descriptor descriptor_value,
        real_type minimum_residual_ratio) const
    {
        vector_space_.assign(source, candidate_.get());
        const real_type input_norm = vector_space_.norm(candidate_.get());
        if(!std::isfinite(input_norm) || !(input_norm > real_type{}))
            return;

        for(std::size_t pass = 0;
            pass < options_.orthogonalization_passes;
            ++pass)
        {
            for(const auto& column : basis)
            {
                const scalar_type coefficient =
                    vector_space_.scalar_prod(
                        column.vector->get(),
                        candidate_.get());
                vector_space_.add_lin_comb(
                    -coefficient,
                    column.vector->get(),
                    scalar_type(1),
                    candidate_.get());
            }
        }

        const real_type independent_norm =
            vector_space_.norm(candidate_.get());
        if(
            !std::isfinite(independent_norm) ||
            independent_norm <=
                minimum_residual_ratio*input_norm)
        {
            return;
        }
        vector_space_.scale(
            scalar_type(real_type(1)/independent_norm),
            candidate_.get());
        basis.emplace_back(
            vector_space_,
            candidate_.get(),
            std::move(descriptor_value));
    }

    static void validate_options(const options_type& options)
    {
        if(
            options.maximum_dimension == 0 ||
            options.maximum_seed_vectors == 0 ||
            options.maximum_seed_vectors > options.maximum_dimension ||
            options.coverage_recovery_maximum_seed_vectors >
                options.maximum_dimension ||
            options.orthogonalization_passes == 0 ||
            !positive_finite(options.seed_innovation_weight) ||
            options.seed_innovation_weight > real_type(1) ||
            !positive_finite(options.dependence_tolerance) ||
            !positive_finite(
                options.minimum_retained_residual_ratio) ||
            options.minimum_retained_residual_ratio >= real_type(1) ||
            !nonnegative_finite(options.absolute_invariance_tolerance) ||
            !nonnegative_finite(options.relative_invariance_tolerance) ||
            !nonnegative_finite(options.real_eigenvalue_tolerance) ||
            !positive_finite(options.eigenvalue_group_tolerance) ||
            !positive_finite(options.principal_angle_rank_tolerance))
        {
            throw std::invalid_argument(
                "invalid tracked invariant-subspace options");
        }
    }

    static bool positive_finite(real_type value)
    {
        return std::isfinite(value) && value > real_type{};
    }

    static bool nonnegative_finite(real_type value)
    {
        return std::isfinite(value) && value >= real_type{};
    }

    bool equivalent_source_value(
        const std::complex<real_type>& left,
        const std::complex<real_type>& right) const
    {
        const real_type scale = std::max({
            real_type(1),
            std::abs(left),
            std::abs(right)});
        return std::min(
            std::abs(left - right),
            std::abs(left - std::conj(right))) <=
                options_.eigenvalue_group_tolerance*scale;
    }

    void trim(basis_type& basis) const
    {
        if(basis.size() > options_.maximum_dimension)
        {
            basis.erase(
                basis.begin() + static_cast<std::ptrdiff_t>(
                    options_.maximum_dimension),
                basis.end());
        }
    }
};

} // namespace tracking
} // namespace stability

#endif
