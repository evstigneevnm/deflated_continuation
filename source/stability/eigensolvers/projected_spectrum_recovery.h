#ifndef __STABILITY_EIGENSOLVERS_PROJECTED_SPECTRUM_RECOVERY_H__
#define __STABILITY_EIGENSOLVERS_PROJECTED_SPECTRUM_RECOVERY_H__

#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/basis_storage.h>
#include <nmfd/solvers/krylov/operator_apply.h>

#include "eigensolver_result.h"
#include "ritz_recovery.h"

namespace stability
{
namespace eigensolvers
{

template<class Real>
struct projected_spectrum_result
{
    eigensolver_status status = eigensolver_status::invalid_input;
    std::vector<std::complex<Real>> eigenvalues;
    std::vector<eigenpair_estimate<Real>> eigenpairs;
    std::size_t projection_dimension = 0;
    std::size_t original_operator_calls = 0;
    std::size_t converged_eigenpairs = 0;
    std::string diagnostic;

    bool succeeded() const
    {
        return status == eigensolver_status::success;
    }

    bool all_converged() const
    {
        return !eigenpairs.empty() &&
            converged_eigenpairs == eigenpairs.size();
    }
};

template<class Real>
struct projected_spectrum_options
{
    Real relative_basis_tolerance =
        std::sqrt(std::numeric_limits<Real>::epsilon());
    std::size_t orthogonalization_passes = 2;
    Real absolute_residual_tolerance =
        Real(64)*std::numeric_limits<Real>::epsilon();
    Real relative_residual_tolerance =
        std::sqrt(std::numeric_limits<Real>::epsilon());
};

template<class VectorSpace, class SmallDenseLapack>
class projected_spectrum_recovery
{
public:
    using vector_space_type = VectorSpace;
    using dense_lapack_type = SmallDenseLapack;
    using scalar_type = typename vector_space_type::scalar_type;
    using norm_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;
    using result_type = projected_spectrum_result<norm_type>;
    using options_type = projected_spectrum_options<norm_type>;
    using matrix_type = typename dense_lapack_type::matrix_type;

    static_assert(
        std::is_floating_point<scalar_type>::value,
        "projected spectrum recovery currently requires a real vector space");

    projected_spectrum_recovery(
        const vector_space_type& vector_space,
        const dense_lapack_type& dense_lapack)
        : vector_space_(vector_space),
          dense_lapack_(dense_lapack)
    {
    }

    template<class OriginalOperator>
    result_type execute(
        const OriginalOperator& original_operator,
        const ritz_vector_storage<vector_space_type>& transformed_vectors,
        const options_type& options = {},
        ritz_vector_storage<vector_space_type>* recovered_vectors = nullptr)
        const
    {
        result_type result;
        if(
            transformed_vectors.size() == 0 ||
            options.orthogonalization_passes == 0 ||
            !(options.relative_basis_tolerance > norm_type{}) ||
            !std::isfinite(options.relative_basis_tolerance) ||
            options.absolute_residual_tolerance < norm_type{} ||
            !std::isfinite(options.absolute_residual_tolerance) ||
            options.relative_residual_tolerance < norm_type{} ||
            !std::isfinite(options.relative_residual_tolerance))
        {
            result.diagnostic =
                "invalid projected-spectrum recovery input";
            return result;
        }

        const std::size_t maximum_dimension =
            2 * transformed_vectors.size();
        nmfd::solvers::krylov::basis_storage<vector_space_type> basis(
            vector_space_,
            maximum_dimension);
        nmfd::solvers::krylov::basis_storage<vector_space_type> applied_basis(
            vector_space_,
            maximum_dimension);
        nmfd::detail::vector_wrap<vector_space_type, true, true> candidate(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> applied(
            vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true> basis_vector(
            vector_space_);

        std::size_t dimension = 0;
        for(std::size_t source = 0;
            source < transformed_vectors.size();
            ++source)
        {
            add_candidate(
                transformed_vectors.real(),
                transformed_vectors.capacity(),
                source,
                basis.data(),
                maximum_dimension,
                dimension,
                *candidate,
                options);
            add_candidate(
                transformed_vectors.imaginary(),
                transformed_vectors.capacity(),
                source,
                basis.data(),
                maximum_dimension,
                dimension,
                *candidate,
                options);
        }

        if(dimension == 0)
        {
            result.status = eigensolver_status::numerical_breakdown;
            result.diagnostic =
                "transformed Ritz vectors span an empty real subspace";
            return result;
        }

        if(
            recovered_vectors != nullptr &&
            recovered_vectors->capacity() < dimension)
        {
            result.diagnostic =
                "physical Ritz-vector output capacity is too small";
            return result;
        }

        matrix_type projection(dimension, dimension);
        for(std::size_t column = 0; column < dimension; ++column)
        {
            vector_space_.assign(
                basis.data(),
                ordinal(maximum_dimension),
                ordinal(column),
                *basis_vector);
            ++result.original_operator_calls;
            if(!nmfd::solvers::krylov::apply_operator(
                   original_operator,
                   *basis_vector,
                   *applied))
            {
                result.status = eigensolver_status::operator_failure;
                result.diagnostic =
                    "original operator failed during projected recovery";
                return result;
            }
            vector_space_.assign(
                *applied,
                applied_basis.data(),
                ordinal(maximum_dimension),
                ordinal(column));
            for(std::size_t row = 0; row < dimension; ++row)
            {
                projection(row, column) = vector_space_.scalar_prod(
                    basis.data(),
                    ordinal(maximum_dimension),
                    ordinal(row),
                    *applied);
            }
        }

        decltype(dense_lapack_.eigensystem(projection))
            projected_eigensystem;
        try
        {
            projected_eigensystem =
                dense_lapack_.eigensystem(projection);
        }
        catch(const std::exception& error)
        {
            result.status = eigensolver_status::dense_solver_failure;
            result.diagnostic = error.what();
            return result;
        }

        result.eigenvalues = projected_eigensystem.eigenvalues;
        std::vector<std::size_t> selected(dimension);
        std::iota(selected.begin(), selected.end(), std::size_t{});

        std::unique_ptr<ritz_vector_storage<vector_space_type>>
            owned_vectors;
        if(recovered_vectors == nullptr)
        {
            owned_vectors =
                std::make_unique<ritz_vector_storage<vector_space_type>>(
                    vector_space_,
                    dimension);
            recovered_vectors = owned_vectors.get();
        }

        std::vector<norm_type> normalization_factors;
        try
        {
            recover_ritz_vectors(
                vector_space_,
                basis.data(),
                maximum_dimension,
                dimension,
                projected_eigensystem,
                selected,
                *recovered_vectors,
                &normalization_factors);
            make_physical_estimates(
                projected_eigensystem,
                applied_basis.data(),
                maximum_dimension,
                dimension,
                normalization_factors,
                *recovered_vectors,
                options,
                result);
        }
        catch(const std::exception& error)
        {
            result.status = eigensolver_status::numerical_breakdown;
            result.diagnostic = error.what();
            return result;
        }

        result.projection_dimension = dimension;
        result.status = eigensolver_status::success;
        return result;
    }

private:
    template<class Multivector>
    void add_candidate(
        const Multivector& source,
        std::size_t source_capacity,
        std::size_t source_column,
        typename vector_space_type::multivector_type& basis,
        std::size_t basis_capacity,
        std::size_t& basis_dimension,
        vector_type& candidate,
        const options_type& options) const
    {
        vector_space_.assign(
            source,
            ordinal(source_capacity),
            ordinal(source_column),
            candidate);
        const norm_type input_norm = vector_space_.norm(candidate);
        if(!(input_norm > norm_type{}) || !std::isfinite(input_norm))
            return;

        for(std::size_t pass = 0;
            pass < options.orthogonalization_passes;
            ++pass)
        {
            for(std::size_t column = 0;
                column < basis_dimension;
                ++column)
            {
                const scalar_type coefficient =
                    vector_space_.scalar_prod(
                        basis,
                        ordinal(basis_capacity),
                        ordinal(column),
                        candidate);
                vector_space_.add_lin_comb(
                    -coefficient,
                    basis,
                    ordinal(basis_capacity),
                    ordinal(column),
                    scalar_type(1),
                    candidate);
            }
        }

        const norm_type orthogonal_norm = vector_space_.norm(candidate);
        if(
            !(orthogonal_norm >
              options.relative_basis_tolerance * input_norm) ||
            !std::isfinite(orthogonal_norm))
        {
            return;
        }
        vector_space_.scale(
            scalar_type(norm_type(1) / orthogonal_norm),
            candidate);
        vector_space_.assign(
            candidate,
            basis,
            ordinal(basis_capacity),
            ordinal(basis_dimension));
        ++basis_dimension;
    }

    template<class Eigensystem>
    void make_physical_estimates(
        const Eigensystem& eigensystem,
        const typename vector_space_type::multivector_type& applied_basis,
        std::size_t basis_capacity,
        std::size_t basis_dimension,
        const std::vector<norm_type>& normalization_factors,
        const ritz_vector_storage<vector_space_type>& recovered_vectors,
        const options_type& options,
        result_type& result) const
    {
        if(
            eigensystem.eigenvalues.size() != basis_dimension ||
            eigensystem.right_eigenvectors.rows() != basis_dimension ||
            eigensystem.right_eigenvectors.cols() != basis_dimension ||
            normalization_factors.size() != basis_dimension ||
            recovered_vectors.size() != basis_dimension)
        {
            throw std::invalid_argument(
                "invalid physical Ritz-validation dimensions");
        }

        nmfd::detail::vector_wrap<vector_space_type, true, true>
            physical_real(vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true>
            physical_imaginary(vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true>
            applied_real(vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true>
            applied_imaginary(vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true>
            residual_real(vector_space_);
        nmfd::detail::vector_wrap<vector_space_type, true, true>
            residual_imaginary(vector_space_);

        result.eigenpairs.clear();
        result.eigenpairs.reserve(basis_dimension);
        result.converged_eigenpairs = 0;
        for(std::size_t index = 0; index < basis_dimension; ++index)
        {
            vector_space_.assign(
                recovered_vectors.real(),
                ordinal(recovered_vectors.capacity()),
                ordinal(index),
                *physical_real);
            vector_space_.assign(
                recovered_vectors.imaginary(),
                ordinal(recovered_vectors.capacity()),
                ordinal(index),
                *physical_imaginary);

            vector_space_.assign_scalar(scalar_type{}, *applied_real);
            vector_space_.assign_scalar(scalar_type{}, *applied_imaginary);
            const norm_type inverse_normalization =
                norm_type(1)/normalization_factors[index];
            for(std::size_t source = 0;
                source < basis_dimension;
                ++source)
            {
                const auto coefficient =
                    eigensystem.right_eigenvectors(source, index);
                vector_space_.add_lin_comb(
                    static_cast<scalar_type>(
                        coefficient.real()*inverse_normalization),
                    applied_basis,
                    ordinal(basis_capacity),
                    ordinal(source),
                    scalar_type(1),
                    *applied_real);
                vector_space_.add_lin_comb(
                    static_cast<scalar_type>(
                        coefficient.imag()*inverse_normalization),
                    applied_basis,
                    ordinal(basis_capacity),
                    ordinal(source),
                    scalar_type(1),
                    *applied_imaginary);
            }

            const auto eigenvalue = eigensystem.eigenvalues[index];
            vector_space_.assign(*applied_real, *residual_real);
            vector_space_.add_lin_comb(
                static_cast<scalar_type>(-eigenvalue.real()),
                *physical_real,
                scalar_type(1),
                *residual_real);
            vector_space_.add_lin_comb(
                static_cast<scalar_type>(eigenvalue.imag()),
                *physical_imaginary,
                scalar_type(1),
                *residual_real);

            vector_space_.assign(*applied_imaginary, *residual_imaginary);
            vector_space_.add_lin_comb(
                static_cast<scalar_type>(-eigenvalue.imag()),
                *physical_real,
                scalar_type(1),
                *residual_imaginary);
            vector_space_.add_lin_comb(
                static_cast<scalar_type>(-eigenvalue.real()),
                *physical_imaginary,
                scalar_type(1),
                *residual_imaginary);

            const norm_type residual = std::sqrt(
                vector_space_.norm_sq(*residual_real) +
                vector_space_.norm_sq(*residual_imaginary));
            const norm_type vector_norm = std::sqrt(
                vector_space_.norm_sq(*physical_real) +
                vector_space_.norm_sq(*physical_imaginary));
            const norm_type applied_norm = std::sqrt(
                vector_space_.norm_sq(*applied_real) +
                vector_space_.norm_sq(*applied_imaginary));
            const norm_type residual_scale =
                applied_norm + std::abs(eigenvalue)*vector_norm;

            eigenpair_estimate<norm_type> estimate;
            estimate.value = eigenvalue;
            estimate.residual = residual;
            estimate.relative_residual =
                residual_scale > norm_type{}
                ? residual/residual_scale
                : residual;
            estimate.converged =
                residual <= options.absolute_residual_tolerance +
                    options.relative_residual_tolerance*residual_scale;
            estimate.projected_index = index;
            if(estimate.converged)
                ++result.converged_eigenpairs;
            result.eigenpairs.emplace_back(estimate);
        }
    }

    static ordinal_type ordinal(std::size_t value)
    {
        return static_cast<ordinal_type>(value);
    }

    const vector_space_type& vector_space_;
    const dense_lapack_type& dense_lapack_;
};

} // namespace eigensolvers
} // namespace stability

#endif
