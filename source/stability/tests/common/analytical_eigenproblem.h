#ifndef __STABILITY_TESTS_COMMON_ANALYTICAL_EIGENPROBLEM_H__
#define __STABILITY_TESTS_COMMON_ANALYTICAL_EIGENPROBLEM_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stability
{
namespace tests
{

template<class Real>
struct analytical_eigenpair
{
    using complex_type = std::complex<Real>;

    complex_type value{};
    std::vector<complex_type> right_eigenvector;
};

template<class Real>
class analytical_eigenproblem
{
public:
    using real_type = Real;
    using complex_type = std::complex<Real>;
    using eigenpair_type = analytical_eigenpair<Real>;

    analytical_eigenproblem(
        std::string name,
        std::size_t dimension,
        std::vector<Real> matrix_column_major,
        std::vector<eigenpair_type> eigenpairs)
        : name_(std::move(name)),
          dimension_(dimension),
          matrix_(std::move(matrix_column_major)),
          eigenpairs_(std::move(eigenpairs))
    {
        if(dimension_ == 0)
            throw std::invalid_argument("analytical_eigenproblem requires a nonzero dimension");
        if(matrix_.size() != dimension_*dimension_)
            throw std::invalid_argument("analytical_eigenproblem matrix size mismatch");
        for(const auto& pair : eigenpairs_)
        {
            if(!pair.right_eigenvector.empty() && pair.right_eigenvector.size() != dimension_)
                throw std::invalid_argument("analytical_eigenproblem eigenvector size mismatch");
        }
    }

    const std::string& name() const
    {
        return name_;
    }

    std::size_t dimension() const
    {
        return dimension_;
    }

    const std::vector<Real>& matrix_column_major() const
    {
        return matrix_;
    }

    const std::vector<eigenpair_type>& eigenpairs() const
    {
        return eigenpairs_;
    }

    Real matrix(std::size_t row, std::size_t col) const
    {
        if(row >= dimension_ || col >= dimension_)
            throw std::out_of_range("analytical_eigenproblem::matrix");
        return matrix_[row + dimension_*col];
    }

    std::vector<complex_type> apply(const std::vector<complex_type>& x) const
    {
        if(x.size() != dimension_)
            throw std::invalid_argument("analytical_eigenproblem::apply vector size mismatch");

        std::vector<complex_type> result(dimension_, complex_type{});
        for(std::size_t col = 0; col < dimension_; ++col)
            for(std::size_t row = 0; row < dimension_; ++row)
                result[row] += matrix(row, col)*x[col];
        return result;
    }

    Real frobenius_norm() const
    {
        Real norm_sq = Real{};
        for(const Real value : matrix_)
            norm_sq += value*value;
        using std::sqrt;
        return sqrt(norm_sq);
    }

private:
    std::string name_;
    std::size_t dimension_;
    std::vector<Real> matrix_;
    std::vector<eigenpair_type> eigenpairs_;
};

namespace detail
{

template<class Real>
std::vector<std::complex<Real>> normalized(std::vector<std::complex<Real>> vector)
{
    Real norm_sq = Real{};
    for(const auto& value : vector)
        norm_sq += std::norm(value);
    using std::sqrt;
    const Real norm = sqrt(norm_sq);
    if(!(norm > Real{}))
        throw std::invalid_argument("cannot normalize a zero analytical eigenvector");
    for(auto& value : vector)
        value /= norm;
    return vector;
}

template<class Real>
std::vector<Real> column_major(
    std::initializer_list<std::initializer_list<Real>> rows)
{
    const std::size_t row_count = rows.size();
    if(row_count == 0)
        return {};
    const std::size_t col_count = rows.begin()->size();
    if(col_count == 0)
        return {};

    std::vector<std::vector<Real>> row_storage;
    row_storage.reserve(row_count);
    for(const auto& row : rows)
    {
        if(row.size() != col_count)
            throw std::invalid_argument("column_major requires equally sized rows");
        row_storage.emplace_back(row);
    }

    std::vector<Real> result(row_count*col_count);
    for(std::size_t col = 0; col < col_count; ++col)
        for(std::size_t row = 0; row < row_count; ++row)
            result[row + row_count*col] = row_storage[row][col];
    return result;
}

} // namespace detail

template<class Real>
analytical_eigenproblem<Real> diagonal_eigenproblem()
{
    using complex = std::complex<Real>;
    using pair = analytical_eigenpair<Real>;
    return analytical_eigenproblem<Real>(
        "diagonal_distinct",
        4,
        detail::column_major<Real>({
            {Real(-4), Real(0), Real(0), Real(0)},
            {Real(0), Real(-1), Real(0), Real(0)},
            {Real(0), Real(0), Real(2), Real(0)},
            {Real(0), Real(0), Real(0), Real(7)}
        }),
        {
            pair{complex(Real(-4), Real(0)), {complex(Real(1)), complex(Real(0)), complex(Real(0)), complex(Real(0))}},
            pair{complex(Real(-1), Real(0)), {complex(Real(0)), complex(Real(1)), complex(Real(0)), complex(Real(0))}},
            pair{complex(Real(2), Real(0)), {complex(Real(0)), complex(Real(0)), complex(Real(1)), complex(Real(0))}},
            pair{complex(Real(7), Real(0)), {complex(Real(0)), complex(Real(0)), complex(Real(0)), complex(Real(1))}}
        });
}

template<class Real>
analytical_eigenproblem<Real> symmetric_eigenproblem()
{
    using complex = std::complex<Real>;
    using pair = analytical_eigenpair<Real>;
    const Real inv_sqrt_two = Real(1)/std::sqrt(Real(2));
    return analytical_eigenproblem<Real>(
        "symmetric_two_by_two",
        2,
        detail::column_major<Real>({
            {Real(2), Real(1)},
            {Real(1), Real(2)}
        }),
        {
            pair{
                complex(Real(3), Real(0)),
                {complex(inv_sqrt_two), complex(inv_sqrt_two)}
            },
            pair{
                complex(Real(1), Real(0)),
                {complex(inv_sqrt_two), complex(-inv_sqrt_two)}
            }
        });
}

template<class Real>
analytical_eigenproblem<Real> nonnormal_eigenproblem()
{
    using complex = std::complex<Real>;
    using pair = analytical_eigenpair<Real>;
    return analytical_eigenproblem<Real>(
        "upper_triangular_nonnormal",
        3,
        detail::column_major<Real>({
            {Real(1), Real(4), Real(0)},
            {Real(0), Real(2), Real(3)},
            {Real(0), Real(0), Real(5)}
        }),
        {
            pair{
                complex(Real(1), Real(0)),
                detail::normalized<Real>({complex(Real(1)), complex(Real(0)), complex(Real(0))})
            },
            pair{
                complex(Real(2), Real(0)),
                detail::normalized<Real>({complex(Real(4)), complex(Real(1)), complex(Real(0))})
            },
            pair{
                complex(Real(5), Real(0)),
                detail::normalized<Real>({complex(Real(1)), complex(Real(1)), complex(Real(1))})
            }
        });
}

template<class Real>
analytical_eigenproblem<Real> complex_pair_eigenproblem()
{
    using complex = std::complex<Real>;
    using pair = analytical_eigenpair<Real>;
    const Real inv_sqrt_two = Real(1)/std::sqrt(Real(2));
    return analytical_eigenproblem<Real>(
        "real_rotation_block",
        3,
        detail::column_major<Real>({
            {Real(0), Real(-2), Real(0)},
            {Real(2), Real(0), Real(0)},
            {Real(0), Real(0), Real(-3)}
        }),
        {
            pair{
                complex(Real(0), Real(2)),
                {complex(inv_sqrt_two), complex(Real(0), -inv_sqrt_two), complex(Real(0))}
            },
            pair{
                complex(Real(0), Real(-2)),
                {complex(inv_sqrt_two), complex(Real(0), inv_sqrt_two), complex(Real(0))}
            },
            pair{
                complex(Real(-3), Real(0)),
                {complex(Real(0)), complex(Real(0)), complex(Real(1))}
            }
        });
}

template<class Real>
analytical_eigenproblem<Real> clustered_eigenproblem()
{
    using complex = std::complex<Real>;
    using pair = analytical_eigenpair<Real>;
    const Real delta = Real(64)*std::numeric_limits<Real>::epsilon();
    return analytical_eigenproblem<Real>(
        "clustered_diagonal",
        3,
        detail::column_major<Real>({
            {Real(1), Real(0), Real(0)},
            {Real(0), Real(1) + delta, Real(0)},
            {Real(0), Real(0), Real(4)}
        }),
        {
            pair{complex(Real(1)), {complex(Real(1)), complex(Real(0)), complex(Real(0))}},
            pair{complex(Real(1) + delta), {complex(Real(0)), complex(Real(1)), complex(Real(0))}},
            pair{complex(Real(4)), {complex(Real(0)), complex(Real(0)), complex(Real(1))}}
        });
}

template<class Real>
std::vector<analytical_eigenproblem<Real>> analytical_eigenproblems()
{
    std::vector<analytical_eigenproblem<Real>> result;
    result.emplace_back(diagonal_eigenproblem<Real>());
    result.emplace_back(symmetric_eigenproblem<Real>());
    result.emplace_back(nonnormal_eigenproblem<Real>());
    result.emplace_back(complex_pair_eigenproblem<Real>());
    result.emplace_back(clustered_eigenproblem<Real>());
    return result;
}

} // namespace tests
} // namespace stability

#endif
