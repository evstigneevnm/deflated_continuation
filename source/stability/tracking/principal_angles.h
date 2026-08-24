#ifndef STABILITY_TRACKING_PRINCIPAL_ANGLES_H
#define STABILITY_TRACKING_PRINCIPAL_ANGLES_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nmfd/operations/linalg/host_small_dense_backend.h>

namespace stability
{
namespace tracking
{

template<class Real>
struct principal_angle_result
{
    std::vector<Real> singular_values;
    std::vector<Real> angles;
    std::size_t numerical_rank = 0;
    std::size_t left_dimension = 0;
    std::size_t right_dimension = 0;
    std::size_t dimension_gap = 0;
    Real minimum_cosine = Real{};
    Real maximum_angle = Real{};
};

namespace detail
{

template<class Real>
std::vector<Real> symmetric_eigenvalues_jacobi(
    std::vector<Real> matrix,
    std::size_t dimension)
{
    if(matrix.size() != dimension*dimension)
        throw std::invalid_argument("invalid symmetric matrix dimensions");
    if(dimension == 0)
        return {};

    const auto index = [dimension](std::size_t row, std::size_t col)
    {
        return row + dimension*col;
    };
    const Real epsilon = std::numeric_limits<Real>::epsilon();
    const std::size_t maximum_iterations =
        std::max<std::size_t>(32, 64*dimension*dimension);

    bool converged = dimension == 1;
    for(std::size_t iteration = 0;
        iteration < maximum_iterations;
        ++iteration)
    {
        std::size_t pivot_row = 0;
        std::size_t pivot_col = 0;
        Real largest = Real{};
        Real diagonal_scale = Real(1);
        for(std::size_t col = 0; col < dimension; ++col)
        {
            diagonal_scale = std::max(
                diagonal_scale,
                std::abs(matrix[index(col, col)]));
            for(std::size_t row = 0; row < col; ++row)
            {
                const Real value =
                    std::abs(matrix[index(row, col)]);
                if(value > largest)
                {
                    largest = value;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if(largest <= Real(32)*epsilon*diagonal_scale)
        {
            converged = true;
            break;
        }

        const Real app = matrix[index(pivot_row, pivot_row)];
        const Real aqq = matrix[index(pivot_col, pivot_col)];
        const Real apq = matrix[index(pivot_row, pivot_col)];
        const Real angle = Real(0.5)*std::atan2(
            Real(2)*apq,
            aqq - app);
        const Real cosine = std::cos(angle);
        const Real sine = std::sin(angle);

        for(std::size_t k = 0; k < dimension; ++k)
        {
            if(k == pivot_row || k == pivot_col)
                continue;
            const Real akp = matrix[index(k, pivot_row)];
            const Real akq = matrix[index(k, pivot_col)];
            const Real rotated_p = cosine*akp - sine*akq;
            const Real rotated_q = sine*akp + cosine*akq;
            matrix[index(k, pivot_row)] = rotated_p;
            matrix[index(pivot_row, k)] = rotated_p;
            matrix[index(k, pivot_col)] = rotated_q;
            matrix[index(pivot_col, k)] = rotated_q;
        }
        matrix[index(pivot_row, pivot_row)] =
            cosine*cosine*app - Real(2)*sine*cosine*apq +
            sine*sine*aqq;
        matrix[index(pivot_col, pivot_col)] =
            sine*sine*app + Real(2)*sine*cosine*apq +
            cosine*cosine*aqq;
        matrix[index(pivot_row, pivot_col)] = Real{};
        matrix[index(pivot_col, pivot_row)] = Real{};
    }
    if(!converged)
    {
        throw std::runtime_error(
            "principal-angle Jacobi eigensolver did not converge");
    }

    std::vector<Real> result(dimension);
    for(std::size_t index_value = 0;
        index_value < dimension;
        ++index_value)
    {
        result[index_value] = matrix[index(index_value, index_value)];
    }
    std::sort(result.begin(), result.end(), std::greater<Real>());
    return result;
}

} // namespace detail

/**
 * Computes principal angles from an overlap matrix Q_left^T Q_right.
 * Both bases are expected to have orthonormal columns.
 */
template<class Real>
principal_angle_result<Real> principal_angles(
    const nmfd::operations::linalg::host_dense_matrix<Real>& overlap,
    Real rank_tolerance = Real(1.0e-8))
{
    if(
        !std::isfinite(rank_tolerance) ||
        !(rank_tolerance > Real{}))
    {
        throw std::invalid_argument("invalid principal-angle rank tolerance");
    }

    principal_angle_result<Real> result;
    result.left_dimension = overlap.rows();
    result.right_dimension = overlap.cols();
    result.dimension_gap =
        result.left_dimension > result.right_dimension
        ? result.left_dimension - result.right_dimension
        : result.right_dimension - result.left_dimension;

    const std::size_t dimension = std::min(
        result.left_dimension,
        result.right_dimension);
    if(dimension == 0)
    {
        result.maximum_angle = std::acos(Real(-1))/Real(2);
        return result;
    }

    std::vector<Real> gram(dimension*dimension, Real{});
    const auto index = [dimension](std::size_t row, std::size_t col)
    {
        return row + dimension*col;
    };
    if(result.left_dimension >= result.right_dimension)
    {
        for(std::size_t col = 0; col < dimension; ++col)
        {
            for(std::size_t row = 0; row <= col; ++row)
            {
                Real value = Real{};
                for(std::size_t k = 0;
                    k < result.left_dimension;
                    ++k)
                {
                    value += overlap(k, row)*overlap(k, col);
                }
                gram[index(row, col)] = value;
                gram[index(col, row)] = value;
            }
        }
    }
    else
    {
        for(std::size_t col = 0; col < dimension; ++col)
        {
            for(std::size_t row = 0; row <= col; ++row)
            {
                Real value = Real{};
                for(std::size_t k = 0;
                    k < result.right_dimension;
                    ++k)
                {
                    value += overlap(row, k)*overlap(col, k);
                }
                gram[index(row, col)] = value;
                gram[index(col, row)] = value;
            }
        }
    }

    const auto eigenvalues =
        detail::symmetric_eigenvalues_jacobi<Real>(
            std::move(gram),
            dimension);
    result.singular_values.reserve(dimension);
    result.angles.reserve(dimension);
    for(const Real eigenvalue : eigenvalues)
    {
        const Real eigenvalue_tolerance =
            Real(256)*std::numeric_limits<Real>::epsilon();
        if(
            eigenvalue < -eigenvalue_tolerance ||
            eigenvalue > Real(1) + eigenvalue_tolerance)
        {
            throw std::runtime_error(
                "principal angles require orthonormal input bases");
        }
        const Real singular_value = std::sqrt(
            std::max(Real{}, std::min(Real(1), eigenvalue)));
        result.singular_values.push_back(singular_value);
        result.angles.push_back(std::acos(singular_value));
        if(singular_value > rank_tolerance)
            ++result.numerical_rank;
    }
    result.minimum_cosine = result.singular_values.back();
    result.maximum_angle = *std::max_element(
        result.angles.begin(),
        result.angles.end());
    if(result.dimension_gap != 0)
        result.maximum_angle = std::acos(Real(-1))/Real(2);
    return result;
}

} // namespace tracking
} // namespace stability

#endif
