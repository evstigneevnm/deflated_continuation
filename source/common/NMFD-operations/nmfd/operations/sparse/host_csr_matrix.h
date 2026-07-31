#ifndef __NMFD_OPERATIONS_SPARSE_HOST_CSR_MATRIX_H__
#define __NMFD_OPERATIONS_SPARSE_HOST_CSR_MATRIX_H__

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace nmfd
{
namespace operations
{
namespace sparse
{

enum class host_csr_execution
{
    serial,
    openmp
};

template<class Scalar, class Index = std::size_t>
struct coordinate_entry
{
    Index row = Index{};
    Index column = Index{};
    Scalar value = Scalar{};
};

template<class Scalar, class Index = std::size_t>
class host_csr_matrix
{
public:
    using scalar_type = Scalar;
    using index_type = Index;
    using entry_type = coordinate_entry<scalar_type, index_type>;

    static_assert(
        std::is_integral<index_type>::value,
        "host_csr_matrix requires an integral index type");

    host_csr_matrix() = default;

    host_csr_matrix(
        index_type rows,
        index_type columns,
        std::vector<entry_type> entries,
        bool drop_merged_zeros = true)
    {
        assign(
            rows,
            columns,
            std::move(entries),
            drop_merged_zeros);
    }

    void assign(
        index_type rows,
        index_type columns,
        std::vector<entry_type> entries,
        bool drop_merged_zeros = true)
    {
        validate_dimension(rows, "row");
        validate_dimension(columns, "column");
        for(const auto& entry : entries)
        {
            if(entry.row >= rows || entry.column >= columns)
                throw std::out_of_range(
                    "host_csr_matrix: coordinate is outside matrix bounds");
        }

        std::sort(
            entries.begin(),
            entries.end(),
            [](const entry_type& lhs, const entry_type& rhs)
            {
                return std::tie(lhs.row, lhs.column) <
                       std::tie(rhs.row, rhs.column);
            });

        std::vector<entry_type> merged;
        merged.reserve(entries.size());
        for(const auto& entry : entries)
        {
            if(
                !merged.empty() &&
                merged.back().row == entry.row &&
                merged.back().column == entry.column)
            {
                merged.back().value += entry.value;
            }
            else
            {
                merged.push_back(entry);
            }
        }

        rows_ = rows;
        columns_ = columns;
        row_offsets_.assign(to_size(rows_) + 1, std::size_t{});
        column_indices_.clear();
        values_.clear();
        column_indices_.reserve(merged.size());
        values_.reserve(merged.size());

        for(const auto& entry : merged)
        {
            if(drop_merged_zeros && entry.value == scalar_type{})
                continue;
            ++row_offsets_[to_size(entry.row) + 1];
            column_indices_.push_back(entry.column);
            values_.push_back(entry.value);
        }
        for(std::size_t row = 0; row < to_size(rows_); ++row)
            row_offsets_[row + 1] += row_offsets_[row];
    }

    index_type rows() const
    {
        return rows_;
    }

    index_type columns() const
    {
        return columns_;
    }

    std::size_t nonzeros() const
    {
        return values_.size();
    }

    const std::vector<std::size_t>& row_offsets() const
    {
        return row_offsets_;
    }

    const std::vector<index_type>& column_indices() const
    {
        return column_indices_;
    }

    const std::vector<scalar_type>& values() const
    {
        return values_;
    }

    void apply(
        const scalar_type* source,
        scalar_type* destination,
        host_csr_execution execution = host_csr_execution::serial) const
    {
        if(source == nullptr || destination == nullptr)
            throw std::invalid_argument(
                "host_csr_matrix::apply: source and destination must be non-null");

        const bool use_openmp =
            execution == host_csr_execution::openmp;
#ifndef _OPENMP
        (void)use_openmp;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
        for(std::ptrdiff_t signed_row = 0;
            signed_row < static_cast<std::ptrdiff_t>(to_size(rows_));
            ++signed_row)
        {
            const std::size_t row = static_cast<std::size_t>(signed_row);
            scalar_type sum = scalar_type{};
            for(
                std::size_t at = row_offsets_[row];
                at < row_offsets_[row + 1];
                ++at)
            {
                sum += values_[at] * source[to_size(column_indices_[at])];
            }
            destination[row] = sum;
        }
    }

    std::vector<scalar_type> apply(
        const std::vector<scalar_type>& source,
        host_csr_execution execution = host_csr_execution::serial) const
    {
        if(source.size() != to_size(columns_))
            throw std::invalid_argument(
                "host_csr_matrix::apply: source size does not match matrix columns");
        std::vector<scalar_type> destination(to_size(rows_), scalar_type{});
        apply(source.data(), destination.data(), execution);
        return destination;
    }

private:
    static std::size_t to_size(index_type value)
    {
        return static_cast<std::size_t>(value);
    }

    static void validate_dimension(index_type value, const char* name)
    {
        if(std::is_signed<index_type>::value && value < index_type{})
            throw std::invalid_argument(
                std::string("host_csr_matrix: negative ") + name +
                " dimension");
    }

    index_type rows_ = index_type{};
    index_type columns_ = index_type{};
    std::vector<std::size_t> row_offsets_{std::size_t{}};
    std::vector<index_type> column_indices_;
    std::vector<scalar_type> values_;
};

} // namespace sparse
} // namespace operations
} // namespace nmfd

#endif
