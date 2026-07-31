#ifndef __NMFD_CPU_REFERENCE_VECTOR_SPACE_H__
#define __NMFD_CPU_REFERENCE_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <nmfd/operations/vector_space_base.h>

namespace nmfd
{
namespace tests
{

template<class T, class Ordinal = std::ptrdiff_t>
class cpu_reference_vector_space :
    public nmfd::operations::vector_space_base<
        T,
        std::vector<T>,
        std::vector<std::vector<T>>,
        Ordinal,
        T>
{
public:
    using scalar_type = T;
    using norm_type = T;
    using ordinal_type = Ordinal;
    using Ord = ordinal_type;
    using vector_type = std::vector<T>;
    using multivector_type = std::vector<vector_type>;

    explicit cpu_reference_vector_space(ordinal_type size)
        : size_(size)
    {
        if(size_ <= ordinal_type{})
            throw std::invalid_argument(
                "cpu_reference_vector_space requires positive size");
    }

    ordinal_type size() const
    {
        return size_;
    }

    void init_vector(vector_type& vector) const override
    {
        vector.clear();
    }

    void start_use_vector(vector_type& vector) const override
    {
        vector.assign(as_size(size_), scalar_type{});
    }

    void stop_use_vector(vector_type&) const override
    {
    }

    void free_vector(vector_type& vector) const override
    {
        vector.clear();
    }

    void init_multivector(
        multivector_type& multivector,
        ordinal_type count) const override
    {
        multivector.assign(as_size(count), vector_type{});
    }

    void start_use_multivector(
        multivector_type& multivector,
        ordinal_type count) const override
    {
        if(multivector.size() != as_size(count))
            multivector.assign(as_size(count), vector_type{});
        for(auto& vector : multivector)
            start_use_vector(vector);
    }

    void stop_use_multivector(
        multivector_type&,
        ordinal_type) const override
    {
    }

    void free_multivector(
        multivector_type& multivector,
        ordinal_type) const override
    {
        multivector.clear();
    }

    vector_type& at(
        multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        check_multivector_index(multivector, count, index);
        return multivector[as_size(index)];
    }

    const vector_type& at(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        check_multivector_index(multivector, count, index);
        return multivector[as_size(index)];
    }

    void assign(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        vector_type& vector) const override
    {
        vector = at(multivector, count, index);
    }

    void assign(
        const vector_type& vector,
        multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const override
    {
        at(multivector, count, index) = vector;
    }

    scalar_type scalar_prod(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        const vector_type& vector) const override
    {
        return scalar_prod(at(multivector, count, index), vector);
    }

    scalar_type scalar_prod_l2(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        const vector_type& vector) const override
    {
        return scalar_prod(multivector, count, index, vector);
    }

    void add_lin_comb(
        scalar_type vector_coefficient,
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index,
        scalar_type result_coefficient,
        vector_type& result) const override
    {
        add_lin_comb(
            vector_coefficient,
            at(multivector, count, index),
            result_coefficient,
            result);
    }

    bool is_valid_number(const vector_type& vector) const override
    {
        check_vector_size(vector);
        for(const scalar_type value : vector)
            if(!std::isfinite(value))
                return false;
        return true;
    }

    scalar_type scalar_prod(
        const vector_type& left,
        const vector_type& right) const override
    {
        check_vector_size(left);
        check_vector_size(right);
        scalar_type result{};
        for(std::size_t index = 0; index < left.size(); ++index)
            result += left[index]*right[index];
        return result;
    }

    scalar_type scalar_prod_l2(
        const vector_type& left,
        const vector_type& right) const override
    {
        return scalar_prod(left, right);
    }

    scalar_type sum(const vector_type& vector) const override
    {
        check_vector_size(vector);
        scalar_type result{};
        for(const scalar_type value : vector)
            result += value;
        return result;
    }

    norm_type asum(const vector_type& vector) const override
    {
        check_vector_size(vector);
        norm_type result{};
        for(const scalar_type value : vector)
            result += std::abs(value);
        return result;
    }

    norm_type norm(const vector_type& vector) const override
    {
        return std::sqrt(norm_sq(vector));
    }

    norm_type norm_l2(const vector_type& vector) const override
    {
        return norm(vector)/std::sqrt(static_cast<norm_type>(size_));
    }

    norm_type norm_sq(const vector_type& vector) const override
    {
        return scalar_prod(vector, vector);
    }

    norm_type norm2_sq(const vector_type& vector) const override
    {
        return norm_sq(vector)/static_cast<norm_type>(size_);
    }

    void assign_scalar(
        scalar_type value,
        vector_type& vector) const override
    {
        check_vector_size(vector);
        std::fill(vector.begin(), vector.end(), value);
    }

    void add_mul_scalar(
        scalar_type value,
        scalar_type vector_coefficient,
        vector_type& vector) const override
    {
        check_vector_size(vector);
        for(auto& entry : vector)
            entry = vector_coefficient*entry + value;
    }

    void scale(
        scalar_type coefficient,
        vector_type& vector) const override
    {
        check_vector_size(vector);
        for(auto& value : vector)
            value *= coefficient;
    }

    void assign(
        const vector_type& source,
        vector_type& destination) const override
    {
        check_vector_size(source);
        destination = source;
    }

    void assign_lin_comb(
        scalar_type coefficient,
        const vector_type& source,
        vector_type& destination) const override
    {
        check_vector_size(source);
        check_vector_size(destination);
        for(std::size_t index = 0; index < source.size(); ++index)
            destination[index] = coefficient*source[index];
    }

    void assign_lin_comb(
        scalar_type left_coefficient,
        const vector_type& left,
        scalar_type right_coefficient,
        const vector_type& right,
        vector_type& destination) const override
    {
        check_vector_size(left);
        check_vector_size(right);
        check_vector_size(destination);
        for(std::size_t index = 0; index < left.size(); ++index)
        {
            destination[index] =
                left_coefficient*left[index] +
                right_coefficient*right[index];
        }
    }

    void add_lin_comb(
        scalar_type source_coefficient,
        const vector_type& source,
        scalar_type destination_coefficient,
        vector_type& destination) const override
    {
        check_vector_size(source);
        check_vector_size(destination);
        for(std::size_t index = 0; index < source.size(); ++index)
        {
            destination[index] =
                source_coefficient*source[index] +
                destination_coefficient*destination[index];
        }
    }

private:
    static std::size_t as_size(ordinal_type value)
    {
        if constexpr(std::is_signed<ordinal_type>::value)
        {
            if(value < ordinal_type{})
                throw std::out_of_range("negative vector-space index");
        }
        return static_cast<std::size_t>(value);
    }

    void check_vector_size(const vector_type& vector) const
    {
        if(vector.size() != as_size(size_))
            throw std::invalid_argument(
                "cpu_reference_vector_space vector size mismatch");
    }

    void check_multivector_index(
        const multivector_type& multivector,
        ordinal_type count,
        ordinal_type index) const
    {
        if(
            multivector.size() != as_size(count) ||
            index < ordinal_type{} ||
            index >= count)
        {
            throw std::out_of_range(
                "cpu_reference_vector_space multivector index");
        }
    }

    ordinal_type size_;
};

} // namespace tests
} // namespace nmfd

#endif
