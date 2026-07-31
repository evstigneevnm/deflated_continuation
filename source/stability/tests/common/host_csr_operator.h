#ifndef __STABILITY_TESTS_COMMON_HOST_CSR_OPERATOR_H__
#define __STABILITY_TESTS_COMMON_HOST_CSR_OPERATOR_H__

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nmfd/operations/sparse/host_csr_matrix.h>

namespace stability
{
namespace tests
{

template<class VectorSpace>
class host_csr_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;
    using matrix_type =
        nmfd::operations::sparse::host_csr_matrix<scalar_type>;
    using execution_type =
        nmfd::operations::sparse::host_csr_execution;

    host_csr_operator(
        const vector_space_type& vector_space,
        matrix_type matrix,
        execution_type execution = execution_type::serial,
        std::vector<scalar_type> left_scale = {})
        : vector_space_(vector_space),
          matrix_(std::move(matrix)),
          execution_(execution),
          left_scale_(std::move(left_scale))
    {
        static_assert(
            vector_space_type::memory_type::is_host_visible,
            "host_csr_operator is a CPU/OMP reference adapter");
        const std::size_t vector_size = vector_space_.get_default_size();
        if(
            static_cast<std::size_t>(matrix_.rows()) != vector_size ||
            static_cast<std::size_t>(matrix_.columns()) != vector_size)
        {
            throw std::invalid_argument(
                "host_csr_operator: matrix and vector-space dimensions differ");
        }
        if(
            !left_scale_.empty() &&
            left_scale_.size() != vector_size)
        {
            throw std::invalid_argument(
                "host_csr_operator: left scale has the wrong dimension");
        }
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        const auto source_view = vector_space_.view(source);
        auto destination_view =
            vector_space_.view(destination, false, true);
        matrix_.apply(
            source_view.raw_ptr(),
            destination_view.raw_ptr(),
            execution_);

        if(!left_scale_.empty())
        {
            const std::size_t rows =
                static_cast<std::size_t>(matrix_.rows());
#ifdef _OPENMP
            const bool parallel =
                execution_ == execution_type::openmp;
#pragma omp parallel for schedule(static) if(parallel)
#endif
            for(std::ptrdiff_t signed_row = 0;
                signed_row < static_cast<std::ptrdiff_t>(rows);
                ++signed_row)
            {
                const std::size_t row =
                    static_cast<std::size_t>(signed_row);
                destination_view(row) *= left_scale_[row];
            }
        }
        return true;
    }

    const matrix_type& matrix() const
    {
        return matrix_;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    void reset_operator_calls() const
    {
        operator_calls_ = 0;
    }

private:
    const vector_space_type& vector_space_;
    matrix_type matrix_;
    execution_type execution_;
    std::vector<scalar_type> left_scale_;
    mutable std::size_t operator_calls_ = 0;
};

} // namespace tests
} // namespace stability

#endif
