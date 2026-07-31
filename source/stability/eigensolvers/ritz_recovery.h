#ifndef __STABILITY_EIGENSOLVERS_RITZ_RECOVERY_H__
#define __STABILITY_EIGENSOLVERS_RITZ_RECOVERY_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/solvers/krylov/basis_storage.h>

namespace stability
{
namespace eigensolvers
{

template<class VectorSpace>
class ritz_vector_storage
{
public:
    using vector_space_type = VectorSpace;
    using multivector_type = typename vector_space_type::multivector_type;

    ritz_vector_storage(
        const vector_space_type& vector_space,
        std::size_t capacity)
        : real_(vector_space, capacity),
          imaginary_(vector_space, capacity),
          capacity_(capacity)
    {
    }

    multivector_type& real()
    {
        return real_.data();
    }

    const multivector_type& real() const
    {
        return real_.data();
    }

    multivector_type& imaginary()
    {
        return imaginary_.data();
    }

    const multivector_type& imaginary() const
    {
        return imaginary_.data();
    }

    std::size_t capacity() const
    {
        return capacity_;
    }

    std::size_t size() const
    {
        return size_;
    }

    void set_size(std::size_t size)
    {
        if(size > capacity_)
            throw std::out_of_range("Ritz vector storage capacity exceeded");
        size_ = size;
    }

private:
    nmfd::solvers::krylov::basis_storage<vector_space_type> real_;
    nmfd::solvers::krylov::basis_storage<vector_space_type> imaginary_;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

template<class VectorSpace, class Eigensystem>
void recover_ritz_vectors(
    const VectorSpace& vector_space,
    const typename VectorSpace::multivector_type& basis,
    std::size_t basis_capacity,
    std::size_t basis_dimension,
    const Eigensystem& projected_eigensystem,
    const std::vector<std::size_t>& selected_indices,
    ritz_vector_storage<VectorSpace>& destination,
    std::vector<typename VectorSpace::norm_type>*
        normalization_factors = nullptr)
{
    using scalar_type = typename VectorSpace::scalar_type;
    using norm_type = typename VectorSpace::norm_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    if(
        basis_dimension == 0 ||
        projected_eigensystem.right_eigenvectors.rows() != basis_dimension ||
        projected_eigensystem.right_eigenvectors.cols() != basis_dimension ||
        selected_indices.size() > destination.capacity())
    {
        throw std::invalid_argument("invalid Ritz-recovery dimensions");
    }

    nmfd::detail::vector_wrap<VectorSpace, true, true> real(vector_space);
    nmfd::detail::vector_wrap<VectorSpace, true, true> imaginary(vector_space);
    if(normalization_factors != nullptr)
    {
        normalization_factors->clear();
        normalization_factors->reserve(selected_indices.size());
    }

    for(std::size_t output = 0; output < selected_indices.size(); ++output)
    {
        const std::size_t projected = selected_indices[output];
        if(projected >= basis_dimension)
            throw std::out_of_range("Ritz-recovery eigenvector index");

        vector_space.assign_scalar(scalar_type{}, *real);
        vector_space.assign_scalar(scalar_type{}, *imaginary);
        for(std::size_t basis_index = 0;
            basis_index < basis_dimension;
            ++basis_index)
        {
            const auto coefficient =
                projected_eigensystem.right_eigenvectors(
                    basis_index,
                    projected);
            vector_space.add_lin_comb(
                static_cast<scalar_type>(coefficient.real()),
                basis,
                static_cast<ordinal_type>(basis_capacity),
                static_cast<ordinal_type>(basis_index),
                scalar_type(1),
                *real);
            vector_space.add_lin_comb(
                static_cast<scalar_type>(coefficient.imag()),
                basis,
                static_cast<ordinal_type>(basis_capacity),
                static_cast<ordinal_type>(basis_index),
                scalar_type(1),
                *imaginary);
        }

        const norm_type combined_norm = std::sqrt(
            vector_space.norm_sq(*real) +
            vector_space.norm_sq(*imaginary));
        if(!(combined_norm > norm_type{}) || !std::isfinite(combined_norm))
            throw std::runtime_error(
                "Ritz recovery produced a zero or non-finite vector");
        if(normalization_factors != nullptr)
            normalization_factors->push_back(combined_norm);
        vector_space.scale(
            scalar_type(norm_type(1)/combined_norm),
            *real);
        vector_space.scale(
            scalar_type(norm_type(1)/combined_norm),
            *imaginary);
        vector_space.assign(
            *real,
            destination.real(),
            static_cast<ordinal_type>(destination.capacity()),
            static_cast<ordinal_type>(output));
        vector_space.assign(
            *imaginary,
            destination.imaginary(),
            static_cast<ordinal_type>(destination.capacity()),
            static_cast<ordinal_type>(output));
    }
    destination.set_size(selected_indices.size());
}

} // namespace eigensolvers
} // namespace stability

#endif
