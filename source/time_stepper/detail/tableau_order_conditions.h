#ifndef TIME_STEPPER_DETAIL_TABLEAU_ORDER_CONDITIONS_H
#define TIME_STEPPER_DETAIL_TABLEAU_ORDER_CONDITIONS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <time_stepper/detail/butcher_tables.h>

namespace time_steppers
{
namespace detail
{

inline bool tableau_close(
    const long double actual,
    const long double expected,
    const long double tolerance)
{
    const long double scale = std::max<long double>(1, std::abs(expected));
    return std::abs(actual - expected) <= tolerance*scale;
}

inline long double tableau_weight(
    const tableu& table,
    const std::size_t index,
    const bool embedded)
{
    return embedded ? table.get_b_hat<long double>(index)
                    : table.get_b<long double>(index);
}

inline bool tableau_has_consistent_abscissae(
    const tableu& table,
    const long double tolerance)
{
    if(table.is_autonomous())
    {
        return false;
    }
    for(std::size_t i = 0; i < table.get_size(); ++i)
    {
        long double row_sum = 0;
        for(std::size_t j = 0; j < table.get_size(); ++j)
        {
            row_sum += table.get_A<long double>(i, j);
        }
        if(!tableau_close(row_sum, table.get_c<long double>(i), tolerance))
        {
            return false;
        }
    }
    return true;
}

inline bool tableau_satisfies_order(
    const tableu& table,
    const unsigned int order,
    const long double tolerance,
    const bool embedded = false)
{
    if(order > 4)
    {
        throw std::invalid_argument("tableau_satisfies_order supports classical RK conditions through order four.");
    }
    if(embedded && !table.is_embedded())
    {
        return false;
    }
    if(order == 0)
    {
        return true;
    }
    if(!tableau_has_consistent_abscissae(table, tolerance))
    {
        return false;
    }

    const std::size_t stages = table.get_size();
    long double sum_b = 0;
    long double sum_bc = 0;
    long double sum_bc2 = 0;
    long double sum_bc3 = 0;
    long double sum_bac = 0;
    long double sum_bac2 = 0;
    long double sum_bca_c = 0;
    long double sum_baa_c = 0;

    for(std::size_t i = 0; i < stages; ++i)
    {
        const long double bi = tableau_weight(table, i, embedded);
        const long double ci = table.get_c<long double>(i);
        sum_b += bi;
        sum_bc += bi*ci;
        sum_bc2 += bi*ci*ci;
        sum_bc3 += bi*ci*ci*ci;
        for(std::size_t j = 0; j < stages; ++j)
        {
            const long double aij = table.get_A<long double>(i, j);
            const long double cj = table.get_c<long double>(j);
            sum_bac += bi*aij*cj;
            sum_bac2 += bi*aij*cj*cj;
            sum_bca_c += bi*ci*aij*cj;
            for(std::size_t k = 0; k < stages; ++k)
            {
                sum_baa_c += bi*aij*table.get_A<long double>(j, k)*
                             table.get_c<long double>(k);
            }
        }
    }

    if(!tableau_close(sum_b, 1, tolerance))
    {
        return false;
    }
    if(order >= 2 && !tableau_close(sum_bc, 0.5L, tolerance))
    {
        return false;
    }
    if(order >= 3 &&
       (!tableau_close(sum_bc2, 1.0L/3.0L, tolerance) ||
        !tableau_close(sum_bac, 1.0L/6.0L, tolerance)))
    {
        return false;
    }
    if(order >= 4 &&
       (!tableau_close(sum_bc3, 0.25L, tolerance) ||
        !tableau_close(sum_bca_c, 0.125L, tolerance) ||
        !tableau_close(sum_bac2, 1.0L/12.0L, tolerance) ||
        !tableau_close(sum_baa_c, 1.0L/24.0L, tolerance)))
    {
        return false;
    }
    return true;
}

inline bool additive_tableau_satisfies_order(
    const std::pair<tableu, tableu>& tables,
    const unsigned int order,
    const long double tolerance)
{
    if(order > 2)
    {
        throw std::invalid_argument("additive_tableau_satisfies_order supports additive RK conditions through order two.");
    }
    if(tables.first.get_size() != tables.second.get_size() ||
       !tableau_has_consistent_abscissae(tables.first, tolerance) ||
       !tableau_has_consistent_abscissae(tables.second, tolerance))
    {
        return false;
    }
    if(order == 0)
    {
        return true;
    }

    const tableu* components[2] = {&tables.first, &tables.second};
    for(const auto* weights: components)
    {
        long double sum_b = 0;
        for(std::size_t i = 0; i < weights->get_size(); ++i)
        {
            sum_b += weights->get_b<long double>(i);
        }
        if(!tableau_close(sum_b, 1, tolerance))
        {
            return false;
        }
        if(order >= 2)
        {
            for(const auto* abscissae: components)
            {
                long double sum_bc = 0;
                for(std::size_t i = 0; i < weights->get_size(); ++i)
                {
                    sum_bc += weights->get_b<long double>(i)*
                              abscissae->get_c<long double>(i);
                }
                if(!tableau_close(sum_bc, 0.5L, tolerance))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace detail
} // namespace time_steppers

#endif
