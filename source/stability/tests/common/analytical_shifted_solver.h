#ifndef __STABILITY_TESTS_COMMON_ANALYTICAL_SHIFTED_SOLVER_H__
#define __STABILITY_TESTS_COMMON_ANALYTICAL_SHIFTED_SOLVER_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{

template<class VectorSpace, class Real>
class analytical_shifted_solver
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;
    using multivector_type = typename vector_space_type::multivector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;

    analytical_shifted_solver(
        const vector_space_type& vector_space,
        const analytical_eigenproblem<Real>& problem,
        Real shift)
        : vector_space_(vector_space),
          dimension_(problem.dimension()),
          shifted_matrix_(dimension_*dimension_, scalar_type{})
    {
        vector_space_.init_multivector(coordinate_basis_, ordinal_dimension());
        vector_space_.start_use_multivector(
            coordinate_basis_,
            ordinal_dimension());
        vector_space_.init_vector(temporary_);
        vector_space_.start_use_vector(temporary_);

        std::vector<scalar_type> host(dimension_, scalar_type{});
        for(std::size_t col = 0; col < dimension_; ++col)
        {
            std::fill(host.begin(), host.end(), scalar_type{});
            host[col] = scalar_type(1);
            vector_space_.set(host.data(), temporary_, host.size());
            vector_space_.assign(
                temporary_,
                coordinate_basis_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col));

            for(std::size_t row = 0; row < dimension_; ++row)
            {
                shifted_matrix_[row*dimension_ + col] =
                    static_cast<scalar_type>(problem.matrix(row, col)) -
                    (row == col ? static_cast<scalar_type>(shift) :
                                  scalar_type{});
            }
        }
    }

    analytical_shifted_solver(const analytical_shifted_solver&) = delete;
    analytical_shifted_solver& operator=(const analytical_shifted_solver&) =
        delete;

    ~analytical_shifted_solver()
    {
        vector_space_.stop_use_vector(temporary_);
        vector_space_.free_vector(temporary_);
        vector_space_.stop_use_multivector(
            coordinate_basis_,
            ordinal_dimension());
        vector_space_.free_multivector(
            coordinate_basis_,
            ordinal_dimension());
    }

    bool solve(const vector_type& rhs, vector_type& solution) const
    {
        ++solve_calls_;
        if(solve_calls_ > fail_after_)
            return false;

        std::vector<scalar_type> matrix = shifted_matrix_;
        std::vector<scalar_type> host_rhs(dimension_, scalar_type{});
        for(std::size_t row = 0; row < dimension_; ++row)
        {
            host_rhs[row] = vector_space_.scalar_prod(
                coordinate_basis_,
                ordinal_dimension(),
                static_cast<ordinal_type>(row),
                rhs);
        }

        for(std::size_t pivot = 0; pivot < dimension_; ++pivot)
        {
            std::size_t best = pivot;
            for(std::size_t row = pivot + 1; row < dimension_; ++row)
            {
                if(
                    std::abs(matrix[row*dimension_ + pivot]) >
                    std::abs(matrix[best*dimension_ + pivot]))
                {
                    best = row;
                }
            }
            if(
                !(std::abs(matrix[best*dimension_ + pivot]) >
                  scalar_type(64)*std::numeric_limits<scalar_type>::epsilon()))
            {
                return false;
            }
            if(best != pivot)
            {
                for(std::size_t col = pivot; col < dimension_; ++col)
                {
                    std::swap(
                        matrix[pivot*dimension_ + col],
                        matrix[best*dimension_ + col]);
                }
                std::swap(host_rhs[pivot], host_rhs[best]);
            }

            for(std::size_t row = pivot + 1; row < dimension_; ++row)
            {
                const scalar_type factor =
                    matrix[row*dimension_ + pivot]/
                    matrix[pivot*dimension_ + pivot];
                matrix[row*dimension_ + pivot] = scalar_type{};
                for(std::size_t col = pivot + 1; col < dimension_; ++col)
                {
                    matrix[row*dimension_ + col] -=
                        factor*matrix[pivot*dimension_ + col];
                }
                host_rhs[row] -= factor*host_rhs[pivot];
            }
        }

        for(std::size_t row = dimension_; row-- > 0;)
        {
            for(std::size_t col = row + 1; col < dimension_; ++col)
                host_rhs[row] -= matrix[row*dimension_ + col]*host_rhs[col];
            host_rhs[row] /= matrix[row*dimension_ + row];
        }

        vector_space_.assign_scalar(scalar_type{}, solution);
        for(std::size_t col = 0; col < dimension_; ++col)
        {
            vector_space_.add_lin_comb(
                host_rhs[col],
                coordinate_basis_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col),
                scalar_type(1),
                solution);
        }
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
    ordinal_type ordinal_dimension() const
    {
        return static_cast<ordinal_type>(dimension_);
    }

    const vector_space_type& vector_space_;
    std::size_t dimension_;
    std::vector<scalar_type> shifted_matrix_;
    multivector_type coordinate_basis_;
    vector_type temporary_;
    mutable std::size_t solve_calls_ = 0;
    std::size_t fail_after_ = std::numeric_limits<std::size_t>::max();
};

} // namespace tests
} // namespace stability

#endif
