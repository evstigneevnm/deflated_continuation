#ifndef __NMFD_OPERATIONS_PRODUCT_VECTOR_SPACE_H__
#define __NMFD_OPERATIONS_PRODUCT_VECTOR_SPACE_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nmfd
{
namespace operations
{

template<class VectorSpace1, class VectorSpace2>
class product_vector_space
{
public:
    using first_space_type = VectorSpace1;
    using second_space_type = VectorSpace2;
    using scalar_type = typename first_space_type::scalar_type;
    using norm_type = typename first_space_type::norm_type;
    using ordinal_type = typename first_space_type::ordinal_type;
    using Ord = ordinal_type;
    using first_vector_type = typename first_space_type::vector_type;
    using second_vector_type = typename second_space_type::vector_type;
    using vector_type = std::pair<first_vector_type, second_vector_type>;
    using multivector_type = std::vector<vector_type>;

    static_assert(
        std::is_same<
            scalar_type,
            typename second_space_type::scalar_type>::value,
        "product vector spaces must have the same scalar type");
    static_assert(
        std::is_same<
            norm_type,
            typename second_space_type::norm_type>::value,
        "product vector spaces must have the same norm type");
    static_assert(
        std::is_same<
            ordinal_type,
            typename second_space_type::ordinal_type>::value,
        "product vector spaces must have the same ordinal type");

    product_vector_space(
        const first_space_type& first,
        const second_space_type& second)
        : first_(&first),
          second_(&second)
    {
    }

    const first_space_type& first_space() const
    {
        return *first_;
    }

    const second_space_type& second_space() const
    {
        return *second_;
    }

    std::size_t first_size() const
    {
        return first_->get_default_size();
    }

    std::size_t second_size() const
    {
        return second_->get_default_size();
    }

    std::size_t get_default_size() const
    {
        return first_size() + second_size();
    }

    std::size_t size() const
    {
        return get_default_size();
    }

    std::size_t get_size(const vector_type& vector) const
    {
        return first_->get_size(vector.first) +
            second_->get_size(vector.second);
    }

    void init_vector(vector_type& vector) const
    {
        first_->init_vector(vector.first);
        second_->init_vector(vector.second);
    }

    void start_use_vector(vector_type& vector) const
    {
        first_->start_use_vector(vector.first);
        second_->start_use_vector(vector.second);
    }

    void stop_use_vector(vector_type& vector) const
    {
        second_->stop_use_vector(vector.second);
        first_->stop_use_vector(vector.first);
    }

    void free_vector(vector_type& vector) const
    {
        second_->free_vector(vector.second);
        first_->free_vector(vector.first);
    }

    void init_multivector(
        multivector_type& vectors,
        ordinal_type count) const
    {
        const std::size_t size_count = checked_count(count);
        vectors.clear();
        vectors.resize(size_count);
        for(auto& vector : vectors)
            init_vector(vector);
    }

    void start_use_multivector(
        multivector_type& vectors,
        ordinal_type count) const
    {
        const std::size_t size_count = checked_count(count);
        check_multivector_size(vectors, size_count);
        for(std::size_t index = 0; index < size_count; ++index)
            start_use_vector(vectors[index]);
    }

    void stop_use_multivector(
        multivector_type& vectors,
        ordinal_type count) const
    {
        const std::size_t size_count = checked_count(count);
        check_multivector_size(vectors, size_count);
        for(std::size_t index = size_count; index > 0; --index)
            stop_use_vector(vectors[index - 1]);
    }

    void free_multivector(
        multivector_type& vectors,
        ordinal_type count) const
    {
        const std::size_t size_count = checked_count(count);
        check_multivector_size(vectors, size_count);
        for(std::size_t index = size_count; index > 0; --index)
            free_vector(vectors[index - 1]);
        vectors.clear();
    }

    void assign_scalar(
        scalar_type value,
        vector_type& destination) const
    {
        first_->assign_scalar(value, destination.first);
        second_->assign_scalar(value, destination.second);
    }

    void assign(
        const vector_type& source,
        vector_type& destination) const
    {
        first_->assign(source.first, destination.first);
        second_->assign(source.second, destination.second);
    }

    void assign_lin_comb(
        scalar_type source_scale,
        const vector_type& source,
        vector_type& destination) const
    {
        first_->assign_lin_comb(
            source_scale,
            source.first,
            destination.first);
        second_->assign_lin_comb(
            source_scale,
            source.second,
            destination.second);
    }

    void assign_lin_comb(
        scalar_type first_scale,
        const vector_type& first,
        scalar_type second_scale,
        const vector_type& second,
        vector_type& destination) const
    {
        first_->assign_lin_comb(
            first_scale,
            first.first,
            second_scale,
            second.first,
            destination.first);
        second_->assign_lin_comb(
            first_scale,
            first.second,
            second_scale,
            second.second,
            destination.second);
    }

    void add_lin_comb(
        scalar_type source_scale,
        const vector_type& source,
        scalar_type destination_scale,
        vector_type& destination) const
    {
        first_->add_lin_comb(
            source_scale,
            source.first,
            destination_scale,
            destination.first);
        second_->add_lin_comb(
            source_scale,
            source.second,
            destination_scale,
            destination.second);
    }

    void scale(
        scalar_type value,
        vector_type& vector) const
    {
        first_->scale(value, vector.first);
        second_->scale(value, vector.second);
    }

    scalar_type scalar_prod(
        const vector_type& first,
        const vector_type& second) const
    {
        return first_->scalar_prod(first.first, second.first) +
            second_->scalar_prod(first.second, second.second);
    }

    scalar_type scalar_prod_l2(
        const vector_type& first,
        const vector_type& second) const
    {
        return scalar_prod(first, second);
    }

    norm_type norm_sq(const vector_type& vector) const
    {
        return first_->norm_sq(vector.first) +
            second_->norm_sq(vector.second);
    }

    norm_type norm(const vector_type& vector) const
    {
        using std::sqrt;
        return sqrt(norm_sq(vector));
    }

    bool check_is_valid_number(const vector_type& vector) const
    {
        return first_->check_is_valid_number(vector.first) &&
            second_->check_is_valid_number(vector.second);
    }

    bool is_valid_number(const vector_type& vector) const
    {
        return check_is_valid_number(vector);
    }

    void assign(
        const multivector_type& source,
        ordinal_type capacity,
        ordinal_type column,
        vector_type& destination) const
    {
        assign(source[checked_column(source, capacity, column)], destination);
    }

    void assign(
        const vector_type& source,
        multivector_type& destination,
        ordinal_type capacity,
        ordinal_type column) const
    {
        assign(source, destination[checked_column(
            destination,
            capacity,
            column)]);
    }

    scalar_type scalar_prod(
        const multivector_type& source,
        ordinal_type capacity,
        ordinal_type column,
        const vector_type& vector) const
    {
        return scalar_prod(
            source[checked_column(source, capacity, column)],
            vector);
    }

    scalar_type scalar_prod_l2(
        const multivector_type& source,
        ordinal_type capacity,
        ordinal_type column,
        const vector_type& vector) const
    {
        return scalar_prod(source, capacity, column, vector);
    }

    void add_lin_comb(
        scalar_type source_scale,
        const multivector_type& source,
        ordinal_type capacity,
        ordinal_type column,
        scalar_type destination_scale,
        vector_type& destination) const
    {
        add_lin_comb(
            source_scale,
            source[checked_column(source, capacity, column)],
            destination_scale,
            destination);
    }

    void set(
        const scalar_type* host,
        vector_type& destination,
        std::size_t count) const
    {
        if(count != get_default_size())
            throw std::invalid_argument(
                "product_vector_space::set size mismatch");
        first_->set(host, destination.first, first_size());
        second_->set(
            host + first_size(),
            destination.second,
            second_size());
    }

    void get(
        const vector_type& source,
        scalar_type* host,
        std::size_t count) const
    {
        if(count != get_default_size())
            throw std::invalid_argument(
                "product_vector_space::get size mismatch");
        first_->get(source.first, host, first_size());
        second_->get(
            source.second,
            host + first_size(),
            second_size());
    }

    void set_value_at_point(
        scalar_type value,
        std::size_t index,
        vector_type& vector) const
    {
        if(index < first_size())
        {
            first_->set_value_at_point(value, index, vector.first);
            return;
        }
        const std::size_t second_index = index - first_size();
        if(second_index >= second_size())
            throw std::out_of_range(
                "product_vector_space::set_value_at_point");
        second_->set_value_at_point(
            value,
            second_index,
            vector.second);
    }

    scalar_type get_value_at_point(
        std::size_t index,
        const vector_type& vector) const
    {
        if(index < first_size())
            return first_->get_value_at_point(index, vector.first);
        const std::size_t second_index = index - first_size();
        if(second_index >= second_size())
            throw std::out_of_range(
                "product_vector_space::get_value_at_point");
        return second_->get_value_at_point(
            second_index,
            vector.second);
    }

private:
    static std::size_t checked_count(ordinal_type count)
    {
        if(count < ordinal_type{})
            throw std::invalid_argument(
                "product_vector_space requires a nonnegative count");
        return static_cast<std::size_t>(count);
    }

    static void check_multivector_size(
        const multivector_type& vectors,
        std::size_t count)
    {
        if(vectors.size() < count)
            throw std::out_of_range(
                "product_vector_space multivector capacity");
    }

    static std::size_t checked_column(
        const multivector_type& vectors,
        ordinal_type capacity,
        ordinal_type column)
    {
        const std::size_t size_capacity = checked_count(capacity);
        const std::size_t size_column = checked_count(column);
        check_multivector_size(vectors, size_capacity);
        if(size_column >= size_capacity)
            throw std::out_of_range(
                "product_vector_space multivector column");
        return size_column;
    }

    const first_space_type* first_;
    const second_space_type* second_;
};

template<class VectorSpace>
using two_block_vector_space =
    product_vector_space<VectorSpace, VectorSpace>;

} // namespace operations
} // namespace nmfd

#endif
