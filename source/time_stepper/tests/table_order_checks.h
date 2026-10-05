#ifndef NMFD_TEST_TABLE_ORDER_CHECKS_H
#define NMFD_TEST_TABLE_ORDER_CHECKS_H
#include <iostream>
#include <string>
#include <time_stepper/runge_kutta/butcher_tables.h>

namespace table_tests
{
inline void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

// Rooted-tree conditions test nonlinear order, not just the stability polynomial.
struct tree
{
    unsigned int order;
    long double density;
    std::vector<std::size_t> children;
};
inline void append_trees(std::vector<tree>& trees, unsigned int order, unsigned int remaining,
    std::size_t first, std::size_t available, std::vector<std::size_t>& children)
{
    if (remaining == 0)
    {
        long double density = order;
        for (auto child : children) density *= trees[child].density;
        trees.push_back({order, density, children});
        return;
    }
    for (auto i = first; i < available; ++i)
    {
        if (trees[i].order > remaining) continue;
        children.push_back(i);
        append_trees(trees, order, remaining-trees[i].order, i, available, children);
        children.pop_back();
    }
}

inline void check_order(const nmfd::time_steppers::runge_kutta::butcher_table& table,
    const std::string& name, bool embedded = false)
{
    const auto order = embedded ? table.embedded_order() : table.order();
    std::vector<tree> trees{{1,1,{}}};
    std::vector<std::size_t> children;
    for (unsigned int n = 2; n <= order+1; ++n)
        append_trees(trees, n, n-1, 0, trees.size(), children);
    std::vector<std::vector<long double>> weights;
    bool next_order_fails = false;
    for (const auto& t : trees)
    {
        std::vector<long double> phi(table.size(), 1);
        for (std::size_t i = 0; i < table.size(); ++i)
            for (const auto child : t.children)
            {
                long double product = 0;
                for (std::size_t j = 0; j < table.size(); ++j)
                    product += table.a(i,j)*weights[child][j];
                phi[i] *= product;
            }
        long double sum = 0;
        for (std::size_t i = 0; i < table.size(); ++i)
            sum += (embedded ? table.embedded_b(i) : table.b(i))*phi[i];
        const auto error = std::abs(sum-1/t.density);
        const auto tolerance = 8*table.coefficient_tolerance();
        if (t.order <= order) require(error <= tolerance, name+": order "+std::to_string(t.order));
        else next_order_fails = next_order_fails || error > tolerance;
        weights.push_back(std::move(phi));
    }
    require(next_order_fails, name+": declared order is too low");
    std::cout << name << (embedded ? " embedded" : " primary") << " order " << order << " verified\n";
}
}
#endif
