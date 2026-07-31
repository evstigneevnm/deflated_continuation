#ifndef __STABILITY_TESTS_COMMON_ANALYTICAL_DENSE_OPERATOR_H__
#define __STABILITY_TESTS_COMMON_ANALYTICAL_DENSE_OPERATOR_H__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "analytical_eigenproblem.h"

namespace stability
{
namespace tests
{

struct default_host_vector_loader
{
    template<class VectorSpace, class HostVector, class Vector>
    void operator()(
        const VectorSpace& vector_space,
        const HostVector& host,
        Vector& destination) const
    {
        vector_space.set(host.data(), destination, host.size());
    }
};

template<class VectorSpace, class Real, class HostVectorLoader = default_host_vector_loader>
class analytical_dense_operator
{
public:
    using vector_space_type = VectorSpace;
    using scalar_type = typename vector_space_type::scalar_type;
    using vector_type = typename vector_space_type::vector_type;
    using multivector_type = typename vector_space_type::multivector_type;
    using ordinal_type = typename vector_space_type::ordinal_type;
    using problem_type = analytical_eigenproblem<Real>;

    analytical_dense_operator(
        const vector_space_type& vector_space,
        const problem_type& problem,
        HostVectorLoader loader = {})
        : vector_space_(vector_space),
          dimension_(problem.dimension()),
          loader_(std::move(loader))
    {
        vector_space_.init_multivector(coordinate_basis_, ordinal_dimension());
        vector_space_.init_multivector(columns_, ordinal_dimension());
        vector_space_.start_use_multivector(coordinate_basis_, ordinal_dimension());
        vector_space_.start_use_multivector(columns_, ordinal_dimension());
        vector_space_.init_vector(temporary_);
        vector_space_.start_use_vector(temporary_);

        std::vector<scalar_type> host(dimension_, scalar_type{});
        for(std::size_t col = 0; col < dimension_; ++col)
        {
            std::fill(host.begin(), host.end(), scalar_type{});
            host[col] = scalar_type(1);
            loader_(vector_space_, host, temporary_);
            vector_space_.assign(
                temporary_,
                coordinate_basis_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col));

            for(std::size_t row = 0; row < dimension_; ++row)
                host[row] = static_cast<scalar_type>(problem.matrix(row, col));
            loader_(vector_space_, host, temporary_);
            vector_space_.assign(
                temporary_,
                columns_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col));
        }
    }

    analytical_dense_operator(const analytical_dense_operator&) = delete;
    analytical_dense_operator& operator=(const analytical_dense_operator&) = delete;

    ~analytical_dense_operator()
    {
        vector_space_.stop_use_vector(temporary_);
        vector_space_.free_vector(temporary_);
        vector_space_.stop_use_multivector(columns_, ordinal_dimension());
        vector_space_.free_multivector(columns_, ordinal_dimension());
        vector_space_.stop_use_multivector(coordinate_basis_, ordinal_dimension());
        vector_space_.free_multivector(coordinate_basis_, ordinal_dimension());
    }

    bool apply(const vector_type& source, vector_type& destination) const
    {
        ++operator_calls_;
        if(operator_calls_ > fail_after_)
            return false;

        vector_space_.assign_scalar(scalar_type{}, destination);
        for(std::size_t col = 0; col < dimension_; ++col)
        {
            const auto coefficient = vector_space_.scalar_prod(
                coordinate_basis_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col),
                source);
            vector_space_.add_lin_comb(
                coefficient,
                columns_,
                ordinal_dimension(),
                static_cast<ordinal_type>(col),
                scalar_type(1),
                destination);
        }
        return true;
    }

    std::size_t dimension() const
    {
        return dimension_;
    }

    std::size_t operator_calls() const
    {
        return operator_calls_;
    }

    void reset_operator_calls() const
    {
        operator_calls_ = 0;
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
    HostVectorLoader loader_;
    multivector_type coordinate_basis_;
    multivector_type columns_;
    vector_type temporary_;
    mutable std::size_t operator_calls_ = 0;
    std::size_t fail_after_ = std::numeric_limits<std::size_t>::max();
};

} // namespace tests
} // namespace stability

#endif
