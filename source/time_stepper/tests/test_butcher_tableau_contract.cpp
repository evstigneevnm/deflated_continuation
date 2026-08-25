#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <time_stepper/detail/butcher_tables.h>
#include <time_stepper/detail/tableau_order_conditions.h>

namespace
{

struct test_context
{
    int checks = 0;
    int failures = 0;

    void check(const bool condition, const std::string& message)
    {
        ++checks;
        if(!condition)
        {
            ++failures;
            std::cerr << "FAIL: " << message << '\n';
        }
    }
};

} // namespace

int main()
{
    using namespace time_steppers::detail;
    constexpr long double tolerance = 5.0e-10L;
    test_context test;
    butcher_tables tables;
    composite_butcher_tables additive_tables;

    struct method_expectation
    {
        const char* name;
        unsigned int order;
        unsigned int embedded_order;
        tableu::type type;
    };

    const std::vector<method_expectation> methods = {
        {"EE", 1, 0, tableu::ERK},
        {"HE", 2, 1, tableu::ERK},
        {"RK33SSP", 3, 2, tableu::ERK},
        {"RK43SSP", 3, 2, tableu::ERK},
        {"RKDP45", 4, 4, tableu::ERK},
        {"RK64SSP", 4, 3, tableu::ERK},
        {"IE", 1, 0, tableu::SDIRK},
        {"IM", 2, 0, tableu::SDIRK},
        {"CN", 2, 0, tableu::DIRK},
        {"SDIRK2A1", 2, 1, tableu::SDIRK},
        {"ESDIRK3A2", 2, 1, tableu::DIRK},
        {"SDIRK3A3", 3, 1, tableu::SDIRK}
    };

    for(const auto& expected: methods)
    {
        auto table = tables.set_table_by_name(expected.name);
        test.check(
            tableau_has_consistent_abscissae(table, tolerance),
            std::string(expected.name) + " has c = A*1");
        test.check(
            tableau_satisfies_order(table, expected.order, tolerance),
            std::string(expected.name) + " satisfies its primary order conditions");
        if(expected.embedded_order > 0)
        {
            test.check(table.is_embedded(), std::string(expected.name) + " is embedded");
            test.check(
                tableau_satisfies_order(table, expected.embedded_order, tolerance, true),
                std::string(expected.name) + " satisfies its embedded order conditions");
        }
        else
        {
            test.check(!table.is_embedded(), std::string(expected.name) + " is not embedded");
        }
        test.check(table.get_type() == expected.type, std::string(expected.name) + " has the expected structure");

        auto repeated_lookup = tables.set_table_by_name(expected.name);
        test.check(
            repeated_lookup.get_size() == table.get_size(),
            std::string(expected.name) + " lookup is repeatable");
    }

    struct additive_expectation
    {
        const char* name;
        unsigned int order;
    };
    const std::vector<additive_expectation> additive_methods = {
        {"IMEX_EULER", 1},
        {"IMEX_TR2", 2},
        {"IMEX_ARS3", 2},
        {"IMEX_AS2", 2}
    };
    for(const auto& expected: additive_methods)
    {
        auto pair = additive_tables.set_table_by_name(expected.name);
        test.check(
            additive_tableau_satisfies_order(pair, expected.order, tolerance),
            std::string(expected.name) + " satisfies additive order conditions");
        test.check(pair.first.get_type() == tableu::ERK, std::string(expected.name) + " has an explicit first table");
        auto repeated_lookup = additive_tables.set_table_by_name(expected.name);
        test.check(
            repeated_lookup.first.get_size() == pair.first.get_size(),
            std::string(expected.name) + " lookup is repeatable");
    }

    bool malformed_dimensions_rejected = false;
    try
    {
        tableu malformed(EXPLICIT_EULER, {{0, 0}, {0, 0}}, {1}, {0});
    }
    catch(const std::logic_error&)
    {
        malformed_dimensions_rejected = true;
    }
    test.check(malformed_dimensions_rejected, "malformed dimensions are rejected");

    bool nonfinite_rejected = false;
    try
    {
        tableu malformed(
            EXPLICIT_EULER,
            {{std::numeric_limits<long double>::quiet_NaN()}},
            {1},
            {0});
    }
    catch(const std::logic_error&)
    {
        nonfinite_rejected = true;
    }
    test.check(nonfinite_rejected, "non-finite coefficients are rejected");

    bool unknown_name_rejected = false;
    try
    {
        auto unknown = tables.set_table_by_name("NOT_A_METHOD");
        (void)unknown;
    }
    catch(const std::logic_error&)
    {
        unknown_name_rejected = true;
    }
    test.check(unknown_name_rejected, "unknown method names are rejected without mutating the catalog");

    std::cout << "Butcher tableau checks: " << test.checks
              << ", failures: " << test.failures << '\n';
    return test.failures == 0 ? 0 : 1;
}
