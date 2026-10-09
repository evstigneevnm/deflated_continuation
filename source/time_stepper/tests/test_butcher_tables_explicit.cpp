#include <limits>
#include "table_order_checks.h"

using namespace nmfd::time_steppers::runge_kutta;
using table_tests::require;

int main()
try
{
    for (const auto* name : {"EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK64SSP", "RK45", "DOP853"})
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
    const auto dp = make_butcher_table("RK45");
    require(dp.size() == 7 && dp.order() == 5 && dp.embedded_order() == 4,
        "RK45 denotes the seven-stage Dormand-Prince 5(4) pair");
    require(dp.a(1, 0) == 1.L / 5, "Coefficients must not be rounded to double first");
    const auto bs = make_butcher_table("RK23");
    require(bs.size() == 4 && bs.order() == 3 && bs.embedded_order() == 2 && bs.error_order() == 3,
        "RK23 is a third-order method with a second-order estimator and endpoint derivative");
    require(bs.dense_order() == 3 && bs.dense_degree() == 3, "RK23 has a native cubic dense-output polynomial");
    require(bs.a(1, 0) == 1.L / 2 && bs.a(2, 1) == 3.L / 4 && bs.b(0) == 2.L / 9 && bs.embedded_b(0) == 7.L / 24 &&
                bs.embedded_b(3) == 1.L / 8,
        "RK23 coefficients must retain long-double rational precision");
    for (std::size_t i = 0; i < bs.size(); ++i)
    {
        require(bs.a(3, i) == bs.b(i), "RK23 endpoint derivative uses the primary weights");
    }
    const auto dop = make_butcher_table("DOP853");
    require(dop.size() == 13 && dop.order() == 8 && dop.c(12) == 1 && dop.b(12) == 0,
        "Fixed DOP853 stores twelve primary stages and the endpoint derivative");
    require(!dop.is_embedded() && dop.embedded_order() == 0 && dop.error_order() == 8 && dop.has_error_estimate() &&
                dop.error_estimator() == butcher_table::error_estimator_type::dop853_combined,
        "DOP853 advertises its combined estimator without fabricating an ordinary embedded pair");
    require(dop.has_dense_output() && dop.dense_order() == 7 && dop.dense_degree() == 7 &&
                dop.dense_outout_stage_count() == 16,
        "DOP853 has a seventh-order extension with three additional derivatives");
    require(dop.dense_outout_c(13) == .1L && dop.dense_outout_c(14) == .2L &&
                dop.dense_outout_c(15) == 7.L / 9 && dop.dense_outout_a(14, 13) != 0 &&
                dop.dense_outout_a(15, 14) != 0,
        "DOP853 dense stages use the prescribed times and earlier additional derivatives");
    require(dop.a(1, 0) == 5.26001519587677318785587544488e-2L,
        "DOP853 coefficients retain published digits without intermediate double rounding");
    for (std::size_t i = 0; i < dop.size(); ++i)
    {
        require(dop.a(12, i) == dop.b(i), "DOP853 endpoint row equals the primary weights");
    }
    butcher_table::matrix_type a(dop.size(), butcher_table::vector_type(dop.size()));
    butcher_table::vector_type c(dop.size()), lower(dop.size());
    for (std::size_t i = 0; i < dop.size(); ++i)
    {
        c[i] = dop.c(i);
        for (std::size_t j = 0; j < dop.size(); ++j)
        {
            a[i][j] = dop.a(i, j);
        }
    }
    for (const unsigned int order : {5u, 3u})
    {
        for (std::size_t i = 0; i < dop.size(); ++i)
        {
            lower[i] = dop.b(i) - (order == 5 ? dop.error_b(i) : dop.secondary_error_b(i));
        }
        table_tests::check_order(butcher_table(a, lower, order, c), order == 5 ? "DOP853 E5" : "DOP853 E3");
    }
    require(dop.error_b(12) == 0 && dop.secondary_error_b(12) == 0,
        "DOP853 error components do not require dense-only derivatives");
    require(dp.error_estimator() == butcher_table::error_estimator_type::embedded_pair && dp.has_error_estimate() &&
                !make_butcher_table("EE").has_error_estimate(),
        "Ordinary table estimator capabilities are unchanged");
    require(bs.dense_outout_stage_count() == bs.size() && bs.dense_outout_a(2, 1) == bs.a(2, 1),
        "Methods without extra interpolation stages retain the ordinary stage layout");
    rejected = 0;
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{1}, {0}}, 1, {{.5L, 0}});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {}, 0, {{.5L, 0}}, {.5L});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{1}, {0}}, 1, {{.5L}}, {.5L});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{1}, {0}}, 1, {{.5L, 0}}, {.6L});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    try
    {
        butcher_table bad({{0}}, {1}, 1, {}, {}, 0, 2e-15L, {{1}, {0}}, 1, {{0, .5L}}, {.5L});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    require(rejected == 5, "Malformed additional dense stages must be rejected");
    rejected = 0;
    const auto nan = std::numeric_limits<long double>::quiet_NaN();
    for (const auto& error : {butcher_table::matrix_type{{0}}, butcher_table::matrix_type{{1}, {0}},
        butcher_table::matrix_type{{nan}, {0}}, butcher_table::matrix_type{{0, 0}, {0}}})
    {
        try
        {
            butcher_table bad({{0}}, {1}, 8, {}, {}, 0, 2e-15L, {}, 0, {}, {}, error);
        }
        catch (const std::invalid_argument&)
        {
            ++rejected;
        }
    }
    try
    {
        butcher_table bad({{0}}, {1}, 8, {}, {1}, 1, 2e-15L, {}, 0, {}, {}, {{0}, {0}});
    }
    catch (const std::invalid_argument&)
    {
        ++rejected;
    }
    require(rejected == 5, "Malformed combined estimators must be rejected");
    std::cout << "Explicit tables: PASS\n";
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
