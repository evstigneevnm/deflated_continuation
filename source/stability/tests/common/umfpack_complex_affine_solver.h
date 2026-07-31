#ifndef __STABILITY_TESTS_UMFPACK_COMPLEX_AFFINE_SOLVER_H__
#define __STABILITY_TESTS_UMFPACK_COMPLEX_AFFINE_SOLVER_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <suitesparse/umfpack.h>

namespace stability
{
namespace tests
{

template<class ProductVectorSpace, class HostCsrMatrix>
class umfpack_complex_affine_solver
{
public:
    using vector_space_type = ProductVectorSpace;
    using matrix_type = HostCsrMatrix;
    using vector_type = typename vector_space_type::vector_type;
    using norm_type = typename vector_space_type::norm_type;
    using complex_type = std::complex<double>;

    umfpack_complex_affine_solver(
        const vector_space_type& vector_space,
        const matrix_type& matrix,
        complex_type operator_scale,
        complex_type diagonal_shift)
        : vector_space_(vector_space),
          dimension_(checked_dimension(matrix)),
          right_real_(dimension_),
          right_imag_(dimension_),
          solution_real_(dimension_),
          solution_imag_(dimension_)
    {
        if(
            vector_space_.first_size() != dimension_ ||
            vector_space_.second_size() != dimension_)
        {
            throw std::invalid_argument(
                "UMFPACK affine solver vector-space size mismatch");
        }
        build_csc(matrix, operator_scale, diagonal_shift);
        factorize();
    }

    ~umfpack_complex_affine_solver()
    {
        if(numeric_ != nullptr)
            umfpack_zi_free_numeric(&numeric_);
    }

    umfpack_complex_affine_solver(
        const umfpack_complex_affine_solver&) = delete;
    umfpack_complex_affine_solver& operator=(
        const umfpack_complex_affine_solver&) = delete;

    bool solve(
        const vector_type& right_hand_side,
        vector_type& solution) const
    {
        ++solve_calls_;
        vector_space_.first_space().get(
            right_hand_side.first,
            right_real_.data(),
            dimension_);
        vector_space_.second_space().get(
            right_hand_side.second,
            right_imag_.data(),
            dimension_);

        const int status = umfpack_zi_solve(
            UMFPACK_A,
            column_offsets_.data(),
            row_indices_.data(),
            values_real_.data(),
            values_imag_.data(),
            solution_real_.data(),
            solution_imag_.data(),
            right_real_.data(),
            right_imag_.data(),
            numeric_,
            control_.data(),
            solve_info_.data());
        if(status != UMFPACK_OK)
        {
            ++failed_solves_;
            return false;
        }

        vector_space_.first_space().set(
            solution_real_.data(),
            solution.first,
            dimension_);
        vector_space_.second_space().set(
            solution_imag_.data(),
            solution.second,
            dimension_);
        return true;
    }

    std::size_t solve_calls() const
    {
        return solve_calls_;
    }

    std::size_t failed_solves() const
    {
        return failed_solves_;
    }

    double reciprocal_condition_estimate() const
    {
        return numeric_info_[UMFPACK_RCOND];
    }

    std::size_t nonzeros() const
    {
        return values_real_.size();
    }

private:
    struct entry
    {
        std::int32_t row;
        std::int32_t column;
        complex_type value;
    };

    static std::size_t checked_dimension(const matrix_type& matrix)
    {
        const std::size_t rows =
            static_cast<std::size_t>(matrix.rows());
        const std::size_t columns =
            static_cast<std::size_t>(matrix.columns());
        if(rows == 0 || rows != columns)
            throw std::invalid_argument(
                "UMFPACK affine solver requires a nonempty square matrix");
        if(
            rows >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()))
        {
            throw std::overflow_error(
                "UMFPACK affine solver dimension exceeds int32");
        }
        return rows;
    }

    void build_csc(
        const matrix_type& matrix,
        complex_type operator_scale,
        complex_type diagonal_shift)
    {
        std::vector<entry> entries;
        entries.reserve(matrix.nonzeros() + dimension_);
        for(std::size_t row = 0; row < dimension_; ++row)
        {
            for(
                std::size_t at = matrix.row_offsets()[row];
                at < matrix.row_offsets()[row + 1];
                ++at)
            {
                entries.push_back(
                    entry{
                        static_cast<std::int32_t>(row),
                        static_cast<std::int32_t>(
                            matrix.column_indices()[at]),
                        operator_scale *
                            static_cast<double>(matrix.values()[at])});
            }
            entries.push_back(
                entry{
                    static_cast<std::int32_t>(row),
                    static_cast<std::int32_t>(row),
                    diagonal_shift});
        }

        std::sort(
            entries.begin(),
            entries.end(),
            [](const entry& lhs, const entry& rhs)
            {
                if(lhs.column != rhs.column)
                    return lhs.column < rhs.column;
                return lhs.row < rhs.row;
            });

        std::vector<entry> merged;
        merged.reserve(entries.size());
        for(const auto& current : entries)
        {
            if(
                !merged.empty() &&
                merged.back().column == current.column &&
                merged.back().row == current.row)
            {
                merged.back().value += current.value;
            }
            else
            {
                merged.push_back(current);
            }
        }

        column_offsets_.assign(dimension_ + 1, std::int32_t{});
        row_indices_.reserve(merged.size());
        values_real_.reserve(merged.size());
        values_imag_.reserve(merged.size());
        for(const auto& current : merged)
        {
            if(current.value == complex_type{})
                continue;
            ++column_offsets_[
                static_cast<std::size_t>(current.column) + 1];
            row_indices_.push_back(current.row);
            values_real_.push_back(current.value.real());
            values_imag_.push_back(current.value.imag());
        }
        for(std::size_t column = 0; column < dimension_; ++column)
        {
            column_offsets_[column + 1] +=
                column_offsets_[column];
        }
    }

    void factorize()
    {
        control_.resize(UMFPACK_CONTROL);
        symbolic_info_.resize(UMFPACK_INFO);
        numeric_info_.resize(UMFPACK_INFO);
        solve_info_.resize(UMFPACK_INFO);
        umfpack_zi_defaults(control_.data());
        control_[UMFPACK_PRL] = 0.0;

        void* symbolic = nullptr;
        const auto dimension =
            static_cast<std::int32_t>(dimension_);
        const int symbolic_status = umfpack_zi_symbolic(
            dimension,
            dimension,
            column_offsets_.data(),
            row_indices_.data(),
            values_real_.data(),
            values_imag_.data(),
            &symbolic,
            control_.data(),
            symbolic_info_.data());
        if(symbolic_status != UMFPACK_OK)
            throw std::runtime_error(
                "UMFPACK symbolic factorization failed with status " +
                std::to_string(symbolic_status));

        const int numeric_status = umfpack_zi_numeric(
            column_offsets_.data(),
            row_indices_.data(),
            values_real_.data(),
            values_imag_.data(),
            symbolic,
            &numeric_,
            control_.data(),
            numeric_info_.data());
        umfpack_zi_free_symbolic(&symbolic);
        if(numeric_status != UMFPACK_OK)
            throw std::runtime_error(
                "UMFPACK numeric factorization failed with status " +
                std::to_string(numeric_status));
    }

    const vector_space_type& vector_space_;
    std::size_t dimension_;
    std::vector<std::int32_t> column_offsets_;
    std::vector<std::int32_t> row_indices_;
    std::vector<double> values_real_;
    std::vector<double> values_imag_;
    std::vector<double> control_;
    std::vector<double> symbolic_info_;
    std::vector<double> numeric_info_;
    mutable std::vector<double> solve_info_;
    mutable std::vector<double> right_real_;
    mutable std::vector<double> right_imag_;
    mutable std::vector<double> solution_real_;
    mutable std::vector<double> solution_imag_;
    void* numeric_ = nullptr;
    mutable std::size_t solve_calls_ = 0;
    mutable std::size_t failed_solves_ = 0;
};

} // namespace tests
} // namespace stability

#endif
