#ifndef __STABILITY_EIGENSOLVERS_HOST_DENSE_OPERATOR_EIGENSOLVER_H__
#define __STABILITY_EIGENSOLVERS_HOST_DENSE_OPERATOR_EIGENSOLVER_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <stability/analysis/detail/vector_workspace.h>

#include "eigensolver_result.h"

namespace stability
{
namespace eigensolvers
{

/**
 * Exact host oracle for a matrix-free real operator.
 *
 * The dense matrix is reconstructed one column at a time through operator
 * actions. Device vectors are transferred only through the vector-space
 * interface. This is intended for validation and small problems.
 */
template<class VectorSpace, class LinearOperator, class DenseLapack>
class host_dense_operator_eigensolver
{
public:
    using vector_space_type = VectorSpace;
    using operator_type = LinearOperator;
    using dense_lapack_type = DenseLapack;
    using scalar_type = typename vector_space_type::scalar_type;
    using real_type = typename vector_space_type::norm_type;
    using vector_type = typename vector_space_type::vector_type;
    using matrix_type = typename dense_lapack_type::matrix_type;
    using result_type = eigensolver_result<real_type>;

    struct options_type
    {
        real_type absolute_residual_tolerance = real_type(1.0e-10);
        real_type relative_residual_tolerance = real_type(1.0e-8);
    };

    host_dense_operator_eigensolver(
        vector_space_type& vector_space,
        const operator_type& linear_operator,
        dense_lapack_type& dense_lapack,
        options_type options = {})
        : vector_space_(vector_space),
          linear_operator_(linear_operator),
          dense_lapack_(dense_lapack),
          options_(options),
          basis_(&vector_space_),
          image_(&vector_space_)
    {
        if(vector_space_.get_default_size() == 0)
        {
            throw std::invalid_argument(
                "host dense operator eigensolver requires a nonempty space");
        }
        if(
            options_.absolute_residual_tolerance < real_type{} ||
            options_.relative_residual_tolerance < real_type{})
        {
            throw std::invalid_argument(
                "host dense eigensolver residual tolerances are invalid");
        }
    }

    result_type execute(const vector_type&) const
    {
        result_type result;
        const std::size_t dimension =
            vector_space_.get_default_size();
        result.scans_requested = 1;
        result.effective_subspace_dimension = dimension;

        matrix_type matrix(dimension, dimension);
        std::vector<scalar_type> host_basis(
            dimension,
            scalar_type{});
        std::vector<scalar_type> host_image(
            dimension,
            scalar_type{});
        for(std::size_t column = 0; column < dimension; ++column)
        {
            std::fill(
                host_basis.begin(),
                host_basis.end(),
                scalar_type{});
            host_basis[column] = scalar_type(1);
            vector_space_.set(
                host_basis.data(),
                basis_.get(),
                dimension);
            if(!apply(basis_.get(), image_.get()))
            {
                result.status = eigensolver_status::operator_failure;
                result.coverage_complete = false;
                result.operator_calls = column + 1;
                result.diagnostic =
                    "host dense oracle operator application failed";
                return result;
            }
            vector_space_.get(
                image_.get(),
                host_image.data(),
                dimension);
            for(std::size_t row = 0; row < dimension; ++row)
            {
                matrix(row, column) =
                    static_cast<real_type>(host_image[row]);
            }
        }
        result.operator_calls = dimension;

        try
        {
            const auto eigensystem = dense_lapack_.eigensystem(matrix);
            result.eigenpairs.reserve(dimension);
            const real_type matrix_norm = frobenius_norm(matrix);
            bool all_converged = true;
            for(std::size_t column = 0;
                column < dimension;
                ++column)
            {
                const auto value = eigensystem.eigenvalues[column];
                const auto residuals = eigenpair_residual(
                    matrix,
                    eigensystem.right_eigenvectors,
                    column,
                    value,
                    matrix_norm);
                eigenpair_estimate<real_type> estimate;
                estimate.value = value;
                estimate.residual = residuals.first;
                estimate.relative_residual = residuals.second;
                estimate.converged =
                    std::isfinite(estimate.residual) &&
                    std::isfinite(estimate.relative_residual) &&
                    (estimate.residual <=
                         options_.absolute_residual_tolerance ||
                     estimate.relative_residual <=
                         options_.relative_residual_tolerance);
                estimate.projected_index = column;
                all_converged =
                    all_converged && estimate.converged;
                result.eigenpairs.push_back(estimate);
            }
            result.status = all_converged
                ? eigensolver_status::success
                : eigensolver_status::no_convergence;
            result.scans_succeeded = all_converged ? 1 : 0;
            result.coverage_complete = all_converged;
            if(!all_converged)
            {
                result.diagnostic =
                    "host dense oracle eigenpair residual check failed";
            }
        }
        catch(const std::exception& error)
        {
            result.status = eigensolver_status::dense_solver_failure;
            result.coverage_complete = false;
            result.diagnostic = error.what();
        }
        return result;
    }

private:
    vector_space_type& vector_space_;
    const operator_type& linear_operator_;
    dense_lapack_type& dense_lapack_;
    options_type options_;
    mutable analysis::detail::vector_workspace<vector_space_type> basis_;
    mutable analysis::detail::vector_workspace<vector_space_type> image_;

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        using return_type = decltype(
            linear_operator_.apply(source, destination));
        if constexpr(std::is_void<return_type>::value)
        {
            linear_operator_.apply(source, destination);
            return vector_space_.check_is_valid_number(destination);
        }
        else
        {
            return
                static_cast<bool>(
                    linear_operator_.apply(source, destination)) &&
                vector_space_.check_is_valid_number(destination);
        }
    }

    static real_type frobenius_norm(const matrix_type& matrix)
    {
        real_type norm_squared = real_type{};
        for(std::size_t column = 0; column < matrix.cols(); ++column)
        {
            for(std::size_t row = 0; row < matrix.rows(); ++row)
            {
                norm_squared +=
                    matrix(row, column)*matrix(row, column);
            }
        }
        using std::sqrt;
        return sqrt(norm_squared);
    }

    template<class ComplexMatrix>
    static std::pair<real_type, real_type> eigenpair_residual(
        const matrix_type& matrix,
        const ComplexMatrix& eigenvectors,
        std::size_t column,
        std::complex<real_type> eigenvalue,
        real_type matrix_norm)
    {
        real_type residual_squared = real_type{};
        real_type vector_norm_squared = real_type{};
        for(std::size_t row = 0; row < matrix.rows(); ++row)
        {
            std::complex<real_type> applied{};
            for(std::size_t inner = 0;
                inner < matrix.cols();
                ++inner)
            {
                applied +=
                    matrix(row, inner)*
                    eigenvectors(inner, column);
            }
            const auto residual =
                applied -
                eigenvalue*eigenvectors(row, column);
            residual_squared += std::norm(residual);
            vector_norm_squared +=
                std::norm(eigenvectors(row, column));
        }
        using std::abs;
        using std::sqrt;
        const real_type residual = sqrt(residual_squared);
        const real_type vector_norm = sqrt(vector_norm_squared);
        const real_type scale = std::max(
            std::numeric_limits<real_type>::min(),
            std::max(matrix_norm, abs(eigenvalue))*vector_norm);
        return {residual, residual/scale};
    }
};

} // namespace eigensolvers
} // namespace stability

#endif
