#include <limits>
#include "table_order_checks.h"

using namespace nmfd::time_steppers::runge_kutta;
using table_tests::require;

int main()
try
{
    for (const auto* name : {"EE", "HE", "BS32", "RK33SSP", "RK43SSP", "RK64SSP", "DOPRI54"})
    {
        const auto table = make_butcher_table(name);
        require(table.type() == butcher_table::scheme_type::explicit_rk, name);
        table_tests::check_order(table, name);
        if (table.is_embedded())
        {
            table_tests::check_order(table, name, true);
        }
        if (table.has_dense_output())
        {
            table_tests::check_order(table, name, false, true);
        }
    }
    const butcher_table inferred({{0, 0}, {1, 0}}, {.5L, .5L}, 2);
    require(inferred.c(1) == 1, "Missing c must mean row sums, not an autonomous problem");
    const butcher_table upper({{0, .5L}, {0, 0}}, {.5L, .5L}, 1);
    require(upper.type() == butcher_table::scheme_type::irk, "Zero diagonal is not enough for ERK");
    const butcher_table tiny({{0, 1e-30L}, {0, 0}}, {.5L, .5L}, 1);
    require(tiny.type() == butcher_table::scheme_type::irk, "Do not erase small nonzero coefficients");
    unsigned int rejected = 0;
    try
    {
        butcher_table bad({}, {}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0, 0}}, {1}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {2}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {1});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 2, {}, {2}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 2, {}, {1});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{std::numeric_limits<long double>::infinity()}}, {1}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        make_butcher_table("unknown");
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    require(rejected == 8, "Malformed tables must be rejected");
    rejected = 0;
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{1}}, 0);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{.5L}}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{0, 1}}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{std::numeric_limits<long double>::quiet_NaN()}}, 1);
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    require(rejected == 5, "Malformed dense extensions must be rejected");
    const auto dp = make_butcher_table("DOPRI54");
    require(dp.size() == 7 && dp.order() == 5 && dp.embedded_order() == 4,
        "DOPRI54 denotes the seven-stage Dormand-Prince 5(4) pair");
    require(dp.a(1, 0) == 1.L / 5, "Coefficients must not be rounded to double first");
    const auto bs = make_butcher_table("BS32");
    require(bs.size() == 4 && bs.order() == 3 && bs.embedded_order() == 2 && bs.error_order() == 3,
        "BS32 is a third-order method with a second-order estimator and endpoint derivative");
    require(bs.dense_order() == 3 && bs.dense_degree() == 3, "BS32 has a native cubic dense-output polynomial");
    require(bs.a(1, 0) == 1.L / 2 && bs.a(2, 1) == 3.L / 4 && bs.b(0) == 2.L / 9 && bs.embedded_b(0) == 7.L / 24 &&
                bs.embedded_b(3) == 1.L / 8,
        "BS32 coefficients must retain long-double rational precision");
    for (std::size_t i = 0; i < bs.size(); ++i)
    {
        require(bs.a(3, i) == bs.b(i), "BS32 endpoint derivative uses the primary weights");
    }
    std::cout << "Explicit tables: PASS\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
