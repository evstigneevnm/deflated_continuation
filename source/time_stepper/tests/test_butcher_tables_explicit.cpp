#include <limits>
#include "table_order_checks.h"

using namespace nmfd::time_steppers::runge_kutta;
using table_tests::require;

int main()
try
{
    for (const auto* name : {"EE","HE","RK33SSP","RK43SSP","RK64SSP","DOPRI54"})
    {
        const auto table = make_butcher_table(name);
        require(table.type() == butcher_table::scheme_type::explicit_rk, name);
        table_tests::check_order(table, name);
        if (table.is_embedded()) table_tests::check_order(table, name, true);
    }
    const butcher_table inferred({{0,0},{1,0}}, {.5L,.5L}, 2);
    require(inferred.c(1) == 1, "Missing c must mean row sums, not an autonomous problem");
    const butcher_table upper({{0,.5L},{0,0}}, {.5L,.5L}, 1);
    require(upper.type() == butcher_table::scheme_type::irk, "Zero diagonal is not enough for ERK");
    const butcher_table tiny({{0,1e-30L},{0,0}}, {.5L,.5L}, 1);
    require(tiny.type() == butcher_table::scheme_type::irk, "Do not erase small nonzero coefficients");
    unsigned int rejected = 0;
    try { butcher_table bad({}, {}, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{0,0}}, {1}, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{0}}, {2}, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{0}}, {1}, 1, {1}); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{0}}, {1}, 2, {}, {2}, 1); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{0}}, {1}, 2, {}, {1}); } catch (const std::invalid_argument&) { ++rejected; }
    try { butcher_table bad({{std::numeric_limits<long double>::infinity()}}, {1}, 1); }
    catch (const std::invalid_argument&) { ++rejected; }
    try { make_butcher_table("unknown"); } catch (const std::invalid_argument&) { ++rejected; }
    require(rejected == 8, "Malformed tables must be rejected");
    const auto dp = make_butcher_table("DOPRI54");
    require(dp.size() == 7 && dp.order() == 5 && dp.embedded_order() == 4,
        "DOPRI54 denotes the seven-stage Dormand-Prince 5(4) pair");
    require(dp.a(1,0) == 1.L/5, "Coefficients must not be rounded to double first");
    std::cout << "Explicit tables: PASS\n";
}
catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
