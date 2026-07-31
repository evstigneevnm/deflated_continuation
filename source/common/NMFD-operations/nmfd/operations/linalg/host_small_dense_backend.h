#ifndef __NMFD_OPERATIONS_LINALG_HOST_SMALL_DENSE_BACKEND_H__
#define __NMFD_OPERATIONS_LINALG_HOST_SMALL_DENSE_BACKEND_H__

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nmfd
{
namespace operations
{
namespace linalg
{

template<class T, class = void>
struct host_small_dense_real_type
{
    using type = T;
};

template<class T>
struct host_small_dense_real_type<
    T,
    std::void_t<decltype(std::declval<const T&>().real())>>
{
    using type = std::decay_t<
        decltype(std::declval<const T&>().real())>;
};

template<class T>
class host_dense_vector
{
public:
    using scalar_type = T;

    host_dense_vector() = default;
    explicit host_dense_vector(std::size_t size)
        : values_(size, T{})
    {
    }

    void resize(std::size_t size)
    {
        values_.assign(size, T{});
    }

    void clear()
    {
        values_.clear();
    }

    std::size_t size() const
    {
        return values_.size();
    }

    T& operator()(std::size_t index)
    {
        return values_.at(index);
    }

    const T& operator()(std::size_t index) const
    {
        return values_.at(index);
    }

    T& operator[](std::size_t index)
    {
        return values_[index];
    }

    const T& operator[](std::size_t index) const
    {
        return values_[index];
    }

    T* data()
    {
        return values_.data();
    }

    const T* data() const
    {
        return values_.data();
    }

private:
    std::vector<T> values_;
};

template<class T>
class host_dense_matrix
{
public:
    using scalar_type = T;

    host_dense_matrix() = default;
    host_dense_matrix(std::size_t rows, std::size_t cols)
    {
        resize(rows, cols);
    }

    void resize(std::size_t rows, std::size_t cols)
    {
        rows_ = rows;
        cols_ = cols;
        values_.assign(rows_*cols_, T{});
    }

    void clear()
    {
        rows_ = 0;
        cols_ = 0;
        values_.clear();
    }

    std::size_t rows() const
    {
        return rows_;
    }

    std::size_t cols() const
    {
        return cols_;
    }

    T& operator()(std::size_t row, std::size_t col)
    {
        check_index(row, col);
        return values_[row + rows_*col];
    }

    const T& operator()(std::size_t row, std::size_t col) const
    {
        check_index(row, col);
        return values_[row + rows_*col];
    }

    T* data()
    {
        return values_.data();
    }

    const T* data() const
    {
        return values_.data();
    }

private:
    void check_index(std::size_t row, std::size_t col) const
    {
        if(row >= rows_ || col >= cols_)
            throw std::out_of_range("host_dense_matrix index out of range");
    }

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> values_;
};

template<class Ordinal, class T>
class host_small_dense_backend
{
    static_assert(
        std::is_integral<Ordinal>::value,
        "host_small_dense_backend ordinal type must be integral");

public:
    using ordinal_type = Ordinal;
    using scalar_type = T;
    using vector_type = host_dense_vector<T>;
    using matrix_type = host_dense_matrix<T>;

    host_small_dense_backend() = default;

    host_small_dense_backend(ordinal_type rows, ordinal_type cols)
    {
        init(rows, cols);
    }

    void init(ordinal_type rows, ordinal_type cols)
    {
        if(rows <= ordinal_type{} || cols <= ordinal_type{})
            throw std::invalid_argument(
                "host_small_dense_backend dimensions must be positive");
        rows_ = rows;
        cols_ = cols;
    }

    std::pair<ordinal_type, ordinal_type> size() const
    {
        check_initialized();
        return {rows_, cols_};
    }

    void init_row_vector(vector_type& vector) const
    {
        check_initialized();
        vector.resize(as_size(cols_));
    }

    void init_col_vector(vector_type& vector) const
    {
        check_initialized();
        vector.resize(as_size(rows_));
    }

    template<class... Vectors>
    void init_row_vectors(Vectors&... vectors) const
    {
        (init_row_vector(vectors), ...);
    }

    template<class... Vectors>
    void init_col_vectors(Vectors&... vectors) const
    {
        (init_col_vector(vectors), ...);
    }

    void init_matrix(matrix_type& matrix) const
    {
        check_initialized();
        matrix.resize(as_size(rows_), as_size(cols_));
    }

    template<class... Matrices>
    void init_matrices(Matrices&... matrices) const
    {
        (init_matrix(matrices), ...);
    }

    void free_row_vector(vector_type& vector) const
    {
        vector.clear();
    }

    void free_col_vector(vector_type& vector) const
    {
        vector.clear();
    }

    template<class... Vectors>
    void free_row_vectors(Vectors&... vectors) const
    {
        (free_row_vector(vectors), ...);
    }

    template<class... Vectors>
    void free_col_vectors(Vectors&... vectors) const
    {
        (free_col_vector(vectors), ...);
    }

    void free_matrix(matrix_type& matrix) const
    {
        matrix.clear();
    }

    template<class... Matrices>
    void free_matrices(Matrices&... matrices) const
    {
        (free_matrix(matrices), ...);
    }

    void assign_scalar_row_vector(const scalar_type value, vector_type& vector) const
    {
        std::fill(vector.data(), vector.data() + vector.size(), value);
    }

    void assign_scalar_col_vector(const scalar_type value, vector_type& vector) const
    {
        std::fill(vector.data(), vector.data() + vector.size(), value);
    }

    void assign_scalar_matrix(const scalar_type value, matrix_type& matrix) const
    {
        std::fill(
            matrix.data(),
            matrix.data() + matrix.rows()*matrix.cols(),
            value);
    }

    void assign_row_vector(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }

    void assign_col_vector(const vector_type& source, vector_type& destination) const
    {
        destination = source;
    }

    void assign_matrix(const matrix_type& source, matrix_type& destination) const
    {
        destination = source;
    }

    scalar_type& matrix_at(
        matrix_type& matrix,
        ordinal_type row,
        ordinal_type col) const
    {
        return matrix(as_size(row), as_size(col));
    }

    const scalar_type& matrix_at(
        const matrix_type& matrix,
        ordinal_type row,
        ordinal_type col) const
    {
        return matrix(as_size(row), as_size(col));
    }

    scalar_type& vector_at(vector_type& vector, ordinal_type index) const
    {
        return vector(as_size(index));
    }

    const scalar_type& vector_at(
        const vector_type& vector,
        ordinal_type index) const
    {
        return vector(as_size(index));
    }

    void matrix_set_column(
        const vector_type& column,
        ordinal_type col,
        matrix_type& matrix) const
    {
        if(column.size() < matrix.rows())
            throw std::invalid_argument("matrix_set_column source is too small");
        for(std::size_t row = 0; row < matrix.rows(); ++row)
            matrix(row, as_size(col)) = column(row);
    }

    void matrix_set_row(
        const vector_type& row_vector,
        ordinal_type row,
        matrix_type& matrix) const
    {
        if(row_vector.size() < matrix.cols())
            throw std::invalid_argument("matrix_set_row source is too small");
        for(std::size_t col = 0; col < matrix.cols(); ++col)
            matrix(as_size(row), col) = row_vector(col);
    }

    bool is_valid_col_vector(const vector_type& vector) const
    {
        return is_valid(vector);
    }

    bool is_valid_row_vector(const vector_type& vector) const
    {
        return is_valid(vector);
    }

    bool is_valid_matrix(const matrix_type& matrix) const
    {
        for(std::size_t col = 0; col < matrix.cols(); ++col)
            for(std::size_t row = 0; row < matrix.rows(); ++row)
                if(!finite(matrix(row, col)))
                    return false;
        return true;
    }

    void solve_upper_triangular_subsystem(
        const matrix_type& matrix,
        vector_type& solution,
        ordinal_type dimension) const
    {
        const std::size_t count = as_size(dimension);
        if(
            count > matrix.rows() ||
            count > matrix.cols() ||
            count > solution.size())
        {
            throw std::invalid_argument(
                "upper triangular subsystem dimension mismatch");
        }

        for(std::size_t row = count; row-- > 0;)
        {
            const scalar_type diagonal = matrix(row, row);
            if(!(magnitude(diagonal) > std::numeric_limits<real_type>::min()))
                throw std::runtime_error(
                    "upper triangular subsystem is singular");
            solution(row) /= diagonal;
            for(std::size_t preceding = 0; preceding < row; ++preceding)
                solution(preceding) -= matrix(preceding, row)*solution(row);
        }
    }

    void solve_upper_triangular_subsystem(
        const matrix_type& matrix,
        const vector_type& rhs,
        vector_type& solution,
        ordinal_type dimension) const
    {
        solution = rhs;
        solve_upper_triangular_subsystem(matrix, solution, dimension);
    }

    void apply_plane_rotation(
        scalar_type& first,
        scalar_type& second,
        const scalar_type& cosine,
        const scalar_type& sine) const
    {
        const scalar_type original_first = first;
        first = cosine*first + sine*second;
        if constexpr(std::is_floating_point<scalar_type>::value)
        {
            second = -sine*original_first + cosine*second;
        }
        else
        {
            using std::conj;
            second =
                -conj(sine)*original_first +
                conj(cosine)*second;
        }
    }

    void generate_plane_rotation(
        const scalar_type& first,
        const scalar_type& second,
        scalar_type& cosine,
        scalar_type& sine) const
    {
        if constexpr(std::is_floating_point<scalar_type>::value)
        {
            if(second == scalar_type{})
            {
                cosine = scalar_type(1);
                sine = scalar_type{};
                return;
            }

            const scalar_type radius = std::hypot(first, second);
            cosine = first/radius;
            sine = second/radius;
        }
        else
        {
            const real_type first_magnitude = magnitude(first);
            const real_type second_magnitude = magnitude(second);
            if(second_magnitude == real_type{})
            {
                cosine = scalar_type(real_type(1), real_type{});
                sine = scalar_type{};
                return;
            }
            if(first_magnitude == real_type{})
            {
                using std::conj;
                cosine = scalar_type{};
                sine = conj(second)/second_magnitude;
                return;
            }

            using std::conj;
            const real_type radius =
                std::hypot(first_magnitude, second_magnitude);
            const scalar_type phase = first/first_magnitude;
            cosine = scalar_type(
                first_magnitude/radius,
                real_type{});
            sine = phase*conj(second)/radius;
        }
    }

    void plane_rotation_col(
        matrix_type& hessenberg,
        vector_type& cosines,
        vector_type& sines,
        vector_type& residual,
        ordinal_type column) const
    {
        const std::size_t col = as_size(column);
        for(std::size_t row = 0; row < col; ++row)
        {
            apply_plane_rotation(
                hessenberg(row, col),
                hessenberg(row + 1, col),
                cosines(row),
                sines(row));
        }

        generate_plane_rotation(
            hessenberg(col, col),
            hessenberg(col + 1, col),
            cosines(col),
            sines(col));
        apply_plane_rotation(
            hessenberg(col, col),
            hessenberg(col + 1, col),
            cosines(col),
            sines(col));
        hessenberg(col + 1, col) = scalar_type{};
        apply_plane_rotation(
            residual(col),
            residual(col + 1),
            cosines(col),
            sines(col));
    }

private:
    using real_type =
        typename host_small_dense_real_type<scalar_type>::type;

    static std::size_t as_size(ordinal_type value)
    {
        if constexpr(std::is_signed<ordinal_type>::value)
        {
            if(value < ordinal_type{})
                throw std::out_of_range("negative small-dense index");
        }
        return static_cast<std::size_t>(value);
    }

    static real_type magnitude(const scalar_type& value)
    {
        using std::abs;
        return abs(value);
    }

    static bool finite(const scalar_type& value)
    {
        if constexpr(std::is_floating_point<scalar_type>::value)
            return std::isfinite(value);
        else
            return std::isfinite(value.real()) && std::isfinite(value.imag());
    }

    static bool is_valid(const vector_type& vector)
    {
        for(std::size_t index = 0; index < vector.size(); ++index)
            if(!finite(vector(index)))
                return false;
        return true;
    }

    void check_initialized() const
    {
        if(rows_ <= ordinal_type{} || cols_ <= ordinal_type{})
            throw std::logic_error(
                "host_small_dense_backend used before init");
    }

    ordinal_type rows_ = ordinal_type{};
    ordinal_type cols_ = ordinal_type{};
};

template<class Backend, class = void>
struct is_host_small_dense_backend : std::false_type
{
};

template<class Backend>
struct is_host_small_dense_backend<
    Backend,
    std::void_t<
        typename Backend::ordinal_type,
        typename Backend::scalar_type,
        typename Backend::vector_type,
        typename Backend::matrix_type,
        decltype(std::declval<Backend&>().init(
            std::declval<typename Backend::ordinal_type>(),
            std::declval<typename Backend::ordinal_type>())),
        decltype(std::declval<const Backend&>().init_matrix(
            std::declval<typename Backend::matrix_type&>())),
        decltype(std::declval<const Backend&>().init_col_vector(
            std::declval<typename Backend::vector_type&>()))>>
    : std::true_type
{
};

} // namespace linalg
} // namespace operations
} // namespace nmfd

#endif
