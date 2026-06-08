#ifndef __NMFD_OPERATIONS_LINALG_SMALL_DENSE_H__
#define __NMFD_OPERATIONS_LINALG_SMALL_DENSE_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace nmfd
{
namespace operations
{
namespace linalg
{

enum class small_solve_status
{
    success,
    invalid_size,
    singular,
    ill_conditioned
};

inline const char *small_solve_status_name( small_solve_status status )
{
    switch ( status )
    {
    case small_solve_status::success:
        return "success";
    case small_solve_status::invalid_size:
        return "invalid_size";
    case small_solve_status::singular:
        return "singular";
    case small_solve_status::ill_conditioned:
        return "ill_conditioned";
    }
    return "unknown";
}

template <class T>
struct small_solve_info
{
    small_solve_status status = small_solve_status::success;
    std::size_t        size = 0;
    std::size_t        rank = 0;
    T                  determinant = T{};
    T                  min_abs_pivot = T{};
    T                  max_abs_entry = T{};
    T                  pivot_ratio = T{};

    bool ok() const
    {
        return status == small_solve_status::success;
    }
};

template <class T, std::size_t MaxSize>
class small_vector
{
    static_assert( MaxSize > 0, "small_vector requires MaxSize > 0" );

public:
    using scalar_type = T;

    small_vector() = default;

    explicit small_vector( std::size_t size )
    {
        resize( size );
        fill( T{} );
    }

    small_vector( std::initializer_list<T> values )
    {
        resize( values.size() );
        std::copy( values.begin(), values.end(), data_.begin() );
    }

    void resize( std::size_t size )
    {
        if ( size > MaxSize )
            throw std::out_of_range( "small_vector::resize exceeds MaxSize" );
        size_ = size;
    }

    void fill( const T value )
    {
        std::fill( data_.begin(), data_.end(), value );
    }

    std::size_t size() const
    {
        return size_;
    }

    T &operator[]( std::size_t i )
    {
        if ( i >= size_ )
            throw std::out_of_range( "small_vector::operator[]" );
        return data_[i];
    }

    const T &operator[]( std::size_t i ) const
    {
        if ( i >= size_ )
            throw std::out_of_range( "small_vector::operator[] const" );
        return data_[i];
    }

private:
    std::size_t            size_ = 0;
    std::array<T, MaxSize> data_{};
};

template <class T, std::size_t MaxRows, std::size_t MaxCols = MaxRows>
class small_matrix
{
    static_assert( MaxRows > 0, "small_matrix requires MaxRows > 0" );
    static_assert( MaxCols > 0, "small_matrix requires MaxCols > 0" );

public:
    using scalar_type = T;

    small_matrix() = default;

    small_matrix( std::size_t rows, std::size_t cols )
    {
        resize( rows, cols );
        fill( T{} );
    }

    small_matrix( std::initializer_list<std::initializer_list<T>> rows )
    {
        const std::size_t row_count = rows.size();
        const std::size_t col_count = row_count == 0 ? 0 : rows.begin()->size();
        resize( row_count, col_count );
        std::size_t i = 0;
        for ( const auto &row : rows )
        {
            if ( row.size() != col_count )
                throw std::invalid_argument( "small_matrix initializer rows must have equal length" );
            std::size_t j = 0;
            for ( const auto &value : row )
            {
                ( *this )( i, j ) = value;
                ++j;
            }
            ++i;
        }
    }

    static small_matrix identity( std::size_t n )
    {
        small_matrix result( n, n );
        for ( std::size_t i = 0; i < n; ++i )
            result( i, i ) = T{ 1 };
        return result;
    }

    void resize( std::size_t rows, std::size_t cols )
    {
        if ( rows > MaxRows || cols > MaxCols )
            throw std::out_of_range( "small_matrix::resize exceeds static capacity" );
        rows_ = rows;
        cols_ = cols;
    }

    void fill( const T value )
    {
        std::fill( data_.begin(), data_.end(), value );
    }

    std::size_t rows() const
    {
        return rows_;
    }

    std::size_t cols() const
    {
        return cols_;
    }

    T &operator()( std::size_t i, std::size_t j )
    {
        if ( i >= rows_ || j >= cols_ )
            throw std::out_of_range( "small_matrix::operator()" );
        return data_[i * MaxCols + j];
    }

    const T &operator()( std::size_t i, std::size_t j ) const
    {
        if ( i >= rows_ || j >= cols_ )
            throw std::out_of_range( "small_matrix::operator() const" );
        return data_[i * MaxCols + j];
    }

private:
    std::size_t                       rows_ = 0;
    std::size_t                       cols_ = 0;
    std::array<T, MaxRows * MaxCols>  data_{};
};

template <class T, std::size_t MaxN>
small_solve_info<T> factor_lu(
    small_matrix<T, MaxN, MaxN> &lu, std::array<std::size_t, MaxN> &pivots, T singular_tol = T{},
    T condition_tol = T{}
)
{
    static_assert( std::is_floating_point<T>::value, "factor_lu currently expects a real floating point type" );

    small_solve_info<T> info;
    const std::size_t   n = lu.rows();
    info.size = n;

    if ( n == 0 || lu.cols() != n )
    {
        info.status = small_solve_status::invalid_size;
        return info;
    }

    for ( std::size_t k = 0; k < n; ++k )
        pivots[k] = k;

    T max_abs_entry = T{};
    for ( std::size_t i = 0; i < n; ++i )
        for ( std::size_t j = 0; j < n; ++j )
            max_abs_entry = std::max( max_abs_entry, static_cast<T>( std::abs( lu( i, j ) ) ) );

    const T scale = std::max( T{ 1 }, max_abs_entry );
    const T tol =
        singular_tol > T{} ? singular_tol : std::numeric_limits<T>::epsilon() * scale * static_cast<T>( n );

    info.max_abs_entry = max_abs_entry;
    info.min_abs_pivot = std::numeric_limits<T>::infinity();

    T det_sign = T{ 1 };
    for ( std::size_t k = 0; k < n; ++k )
    {
        std::size_t pivot = k;
        T           pivot_abs = static_cast<T>( std::abs( lu( k, k ) ) );
        for ( std::size_t i = k + 1; i < n; ++i )
        {
            const T candidate = static_cast<T>( std::abs( lu( i, k ) ) );
            if ( candidate > pivot_abs )
            {
                pivot = i;
                pivot_abs = candidate;
            }
        }

        if ( pivot_abs <= tol )
        {
            info.status = small_solve_status::singular;
            info.rank = k;
            info.determinant = T{};
            info.min_abs_pivot = std::isfinite( info.min_abs_pivot ) ? info.min_abs_pivot : T{};
            info.pivot_ratio = info.min_abs_pivot / scale;
            return info;
        }

        pivots[k] = pivot;
        if ( pivot != k )
        {
            for ( std::size_t j = 0; j < n; ++j )
                std::swap( lu( k, j ), lu( pivot, j ) );
            det_sign = -det_sign;
        }

        info.min_abs_pivot = std::min( info.min_abs_pivot, pivot_abs );
        ++info.rank;

        for ( std::size_t i = k + 1; i < n; ++i )
        {
            lu( i, k ) /= lu( k, k );
            const T factor = lu( i, k );
            for ( std::size_t j = k + 1; j < n; ++j )
                lu( i, j ) -= factor * lu( k, j );
        }
    }

    T det = det_sign;
    for ( std::size_t i = 0; i < n; ++i )
        det *= lu( i, i );

    info.determinant = det;
    info.pivot_ratio = info.min_abs_pivot / scale;
    if ( condition_tol > T{} && info.pivot_ratio <= condition_tol )
        info.status = small_solve_status::ill_conditioned;
    return info;
}

template <class T, std::size_t MaxN>
void solve_lu_in_place(
    const small_matrix<T, MaxN, MaxN> &lu, const std::array<std::size_t, MaxN> &pivots, small_vector<T, MaxN> &x
)
{
    const std::size_t n = lu.rows();
    for ( std::size_t k = 0; k < n; ++k )
        if ( pivots[k] != k )
            std::swap( x[k], x[pivots[k]] );

    for ( std::size_t i = 0; i < n; ++i )
        for ( std::size_t j = 0; j < i; ++j )
            x[i] -= lu( i, j ) * x[j];

    for ( std::size_t ii = n; ii-- > 0; )
    {
        for ( std::size_t j = ii + 1; j < n; ++j )
            x[ii] -= lu( ii, j ) * x[j];
        x[ii] /= lu( ii, ii );
    }
}

template <class T, std::size_t MaxN>
small_solve_info<T> solve(
    const small_matrix<T, MaxN, MaxN> &a, const small_vector<T, MaxN> &b, small_vector<T, MaxN> &x,
    T singular_tol = T{}, T condition_tol = T{}
)
{
    if ( a.rows() != a.cols() || b.size() != a.rows() )
    {
        small_solve_info<T> info;
        info.status = small_solve_status::invalid_size;
        info.size = a.rows();
        return info;
    }

    small_matrix<T, MaxN, MaxN> lu = a;
    std::array<std::size_t, MaxN> pivots{};
    auto info = factor_lu( lu, pivots, singular_tol, condition_tol );
    if ( !info.ok() )
        return info;

    x = b;
    solve_lu_in_place( lu, pivots, x );
    return info;
}

template <class T, std::size_t MaxN, std::size_t MaxRhs>
small_solve_info<T> solve_multiple_rhs(
    const small_matrix<T, MaxN, MaxN> &a, const small_matrix<T, MaxN, MaxRhs> &b,
    small_matrix<T, MaxN, MaxRhs> &x, T singular_tol = T{}, T condition_tol = T{}
)
{
    if ( a.rows() != a.cols() || b.rows() != a.rows() )
    {
        small_solve_info<T> info;
        info.status = small_solve_status::invalid_size;
        info.size = a.rows();
        return info;
    }

    small_matrix<T, MaxN, MaxN> lu = a;
    std::array<std::size_t, MaxN> pivots{};
    auto info = factor_lu( lu, pivots, singular_tol, condition_tol );
    if ( !info.ok() )
        return info;

    const std::size_t n = a.rows();
    const std::size_t rhs_count = b.cols();
    x.resize( n, rhs_count );
    for ( std::size_t i = 0; i < n; ++i )
        for ( std::size_t r = 0; r < rhs_count; ++r )
            x( i, r ) = b( i, r );

    for ( std::size_t k = 0; k < n; ++k )
        if ( pivots[k] != k )
            for ( std::size_t r = 0; r < rhs_count; ++r )
                std::swap( x( k, r ), x( pivots[k], r ) );

    for ( std::size_t i = 0; i < n; ++i )
        for ( std::size_t j = 0; j < i; ++j )
            for ( std::size_t r = 0; r < rhs_count; ++r )
                x( i, r ) -= lu( i, j ) * x( j, r );

    for ( std::size_t ii = n; ii-- > 0; )
    {
        for ( std::size_t j = ii + 1; j < n; ++j )
            for ( std::size_t r = 0; r < rhs_count; ++r )
                x( ii, r ) -= lu( ii, j ) * x( j, r );
        for ( std::size_t r = 0; r < rhs_count; ++r )
            x( ii, r ) /= lu( ii, ii );
    }

    return info;
}

template <class T, std::size_t MaxN>
small_solve_info<T> inverse(
    const small_matrix<T, MaxN, MaxN> &a, small_matrix<T, MaxN, MaxN> &a_inv, T singular_tol = T{},
    T condition_tol = T{}
)
{
    const auto identity = small_matrix<T, MaxN, MaxN>::identity( a.rows() );
    return solve_multiple_rhs( a, identity, a_inv, singular_tol, condition_tol );
}

template <class T, std::size_t MaxRows, std::size_t MaxInner, std::size_t MaxCols>
void multiply(
    const small_matrix<T, MaxRows, MaxInner> &a, const small_matrix<T, MaxInner, MaxCols> &b,
    small_matrix<T, MaxRows, MaxCols> &c
)
{
    if ( a.cols() != b.rows() )
        throw std::invalid_argument( "small_dense::multiply incompatible dimensions" );

    c.resize( a.rows(), b.cols() );
    for ( std::size_t i = 0; i < a.rows(); ++i )
    {
        for ( std::size_t j = 0; j < b.cols(); ++j )
        {
            T sum = T{};
            for ( std::size_t k = 0; k < a.cols(); ++k )
                sum += a( i, k ) * b( k, j );
            c( i, j ) = sum;
        }
    }
}

} // namespace linalg
} // namespace operations
} // namespace nmfd

#endif
