#ifndef NMFD_TIME_STEPPERS_BUTCHER_TABLE_H
#define NMFD_TIME_STEPPERS_BUTCHER_TABLE_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{

class butcher_table
{
public:
    using scalar_type = long double;
    using vector_type = std::vector<scalar_type>;
    using matrix_type = std::vector<vector_type>;
    enum class scheme_type { explicit_rk, dirk, sdirk, irk };

    butcher_table(matrix_type a, vector_type b, unsigned int order,
        vector_type c = {}, vector_type embedded_b = {}, unsigned int embedded_order = 0,
        scalar_type coefficient_tolerance = 2e-15L):
        a_(std::move(a)), b_(std::move(b)), c_(std::move(c)),
        embedded_b_(std::move(embedded_b)), order_(order), embedded_order_(embedded_order),
        tolerance_(coefficient_tolerance)
    {
        if (b_.empty() || a_.size() != b_.size() || order_ == 0 ||
            !std::isfinite(tolerance_) || tolerance_ <= 0 ||
            embedded_b_.empty() != (embedded_order_ == 0) ||
            (!embedded_b_.empty() && embedded_order_ >= order_))
            throw std::invalid_argument("Invalid Butcher table dimensions/order/tolerance");
        for (const auto& row : a_)
            validate_vector(row, size());
        validate_vector(b_, size());
        check_equal(std::accumulate(b_.begin(), b_.end(), 0.L), 1.L);
        if (c_.empty())
        {
            for (const auto& row : a_)
                c_.push_back(std::accumulate(row.begin(), row.end(), 0.L));
        }
        validate_vector(c_, size());
        for (std::size_t i = 0; i < size(); ++i)
            check_equal(c_[i], std::accumulate(a_[i].begin(), a_[i].end(), 0.L));
        if (is_embedded())
        {
            validate_vector(embedded_b_, size());
            check_equal(std::accumulate(embedded_b_.begin(), embedded_b_.end(), 0.L), 1.L);
        }
    }

    std::size_t size() const { return b_.size(); }
    unsigned int order() const { return order_; }
    unsigned int embedded_order() const { return embedded_order_; }
    unsigned int error_order() const { return is_embedded() ? embedded_order_ + 1 : 0; }
    bool is_embedded() const { return !embedded_b_.empty(); }
    scalar_type coefficient_tolerance() const { return tolerance_; }
    scalar_type a(std::size_t i, std::size_t j) const { return a_.at(i).at(j); }
    scalar_type b(std::size_t i) const { return b_.at(i); }
    scalar_type c(std::size_t i) const { return c_.at(i); }
    scalar_type embedded_b(std::size_t i) const { return embedded_b_.at(i); }
    scalar_type error_b(std::size_t i) const { return b(i) - embedded_b(i); }

    scheme_type type() const
    {
        // Structural zeros are exact: a tiny nonzero upper entry is still implicit.
        for (std::size_t i = 0; i < size(); ++i)
            for (std::size_t j = i + 1; j < size(); ++j)
                if (a_[i][j] != 0) return scheme_type::irk;
        bool zero_diagonal = true, same_diagonal = true;
        for (std::size_t i = 0; i < size(); ++i)
        {
            zero_diagonal = zero_diagonal && a_[i][i] == 0;
            same_diagonal = same_diagonal && a_[i][i] == a_[0][0];
        }
        if (zero_diagonal) return scheme_type::explicit_rk;
        return same_diagonal ? scheme_type::sdirk : scheme_type::dirk;
    }

private:
    matrix_type a_;
    vector_type b_, c_, embedded_b_;
    unsigned int order_, embedded_order_;
    scalar_type tolerance_;

    static void validate_vector(const vector_type& v, std::size_t size)
    {
        if (v.size() != size) throw std::invalid_argument("Butcher table shape mismatch");
        for (const auto value : v)
            if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite Butcher coefficient");
    }
    void check_equal(scalar_type x, scalar_type y) const
    {
        if (std::abs(x - y) > tolerance_ * std::max({1.L, std::abs(x), std::abs(y)}))
            throw std::invalid_argument("Inconsistent Butcher weights or stage times");
    }
};
}
}
}
#endif
