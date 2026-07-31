#ifndef __STABILITY_TESTS_COMMON_ANALYTICAL_MATRIX_SOLVER_H__
#define __STABILITY_TESTS_COMMON_ANALYTICAL_MATRIX_SOLVER_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stability
{
namespace tests
{

template<class VectorSpace>
class analytical_matrix_solver
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;

    analytical_matrix_solver(
        const vector_space_type& vector_space,
        std::size_t dimension,
        std::vector<scalar_type> row_major_matrix)
        : vector_space_(vector_space),
          dimension_(dimension),
          matrix_(std::move(row_major_matrix))
    {
        if(
            dimension_ == 0 ||
            matrix_.size() != dimension_ * dimension_)
        {
            throw std::invalid_argument(
                "analytical_matrix_solver requires a nonempty square matrix");
        }
    }

    bool solve(const vector_type& rhs, vector_type& solution) const
    {
        ++solve_calls_;
        if(solve_calls_ > fail_after_)
            return false;

        std::vector<scalar_type> matrix = matrix_;
        std::vector<scalar_type> host_rhs(dimension_, scalar_type{});
        vector_space_.get(rhs, host_rhs.data(), dimension_);

        scalar_type matrix_scale = scalar_type{};
        for(const scalar_type value : matrix)
            matrix_scale = std::max(matrix_scale, std::abs(value));
        const scalar_type pivot_tolerance =
            scalar_type(64) *
            std::numeric_limits<scalar_type>::epsilon() *
            std::max(scalar_type(1), matrix_scale);

        for(std::size_t pivot = 0; pivot < dimension_; ++pivot)
        {
            std::size_t best = pivot;
            for(std::size_t row = pivot + 1; row < dimension_; ++row)
            {
                if(
                    std::abs(matrix[row * dimension_ + pivot]) >
                    std::abs(matrix[best * dimension_ + pivot]))
                {
                    best = row;
                }
            }
            if(!(std::abs(matrix[best * dimension_ + pivot]) > pivot_tolerance))
                return false;
            if(best != pivot)
            {
                for(std::size_t column = pivot;
                    column < dimension_;
                    ++column)
                {
                    std::swap(
                        matrix[pivot * dimension_ + column],
                        matrix[best * dimension_ + column]);
                }
                std::swap(host_rhs[pivot], host_rhs[best]);
            }

            for(std::size_t row = pivot + 1; row < dimension_; ++row)
            {
                const scalar_type factor =
                    matrix[row * dimension_ + pivot] /
                    matrix[pivot * dimension_ + pivot];
                matrix[row * dimension_ + pivot] = scalar_type{};
                for(std::size_t column = pivot + 1;
                    column < dimension_;
                    ++column)
                {
                    matrix[row * dimension_ + column] -=
                        factor * matrix[pivot * dimension_ + column];
                }
                host_rhs[row] -= factor * host_rhs[pivot];
            }
        }

        for(std::size_t row = dimension_; row-- > 0;)
        {
            for(std::size_t column = row + 1;
                column < dimension_;
                ++column)
            {
                host_rhs[row] -=
                    matrix[row * dimension_ + column] * host_rhs[column];
            }
            host_rhs[row] /= matrix[row * dimension_ + row];
        }
        vector_space_.set(host_rhs.data(), solution, dimension_);
        return true;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

    void fail_after(std::size_t successful_calls)
    {
        fail_after_ = successful_calls;
    }

private:
    const vector_space_type& vector_space_;
    std::size_t dimension_;
    std::vector<scalar_type> matrix_;
    mutable std::size_t solve_calls_ = 0;
    std::size_t fail_after_ = std::numeric_limits<std::size_t>::max();
};

} // namespace tests
} // namespace stability

#endif
