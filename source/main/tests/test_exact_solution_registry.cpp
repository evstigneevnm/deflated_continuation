#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <main/deflation_continuation/exact_solution_registry.h>

namespace
{

struct legacy_operator
{
    void exact_solution(const double& parameter, std::vector<double>& value)
    {
        value = {parameter, -parameter};
    }
};

struct indexed_operator
{
    std::size_t exact_solution_count() const { return 3; }
    bool exact_solution(
        std::size_t branch,
        const double& parameter,
        std::vector<double>& value)
    {
        value = {static_cast<double>(branch), parameter};
        return branch != 2;
    }
    std::string exact_solution_name(std::size_t branch) const
    {
        return "branch_" + std::to_string(branch);
    }
};

struct no_exact_solution
{
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

}

int main()
{
    try
    {
        legacy_operator legacy;
        main_classes::deflation_continuation_detail::exact_solution_registry<
            legacy_operator,
            double,
            std::vector<double>> legacy_registry(&legacy);
        require(legacy_registry.count() == 1, "legacy count");
        std::vector<double> value;
        require(legacy_registry.evaluate(0, 4.0, value), "legacy evaluation");
        require(value == std::vector<double>({4.0, -4.0}), "legacy value");
        require(legacy_registry.name(0) == "exact_solution_0", "legacy name");

        indexed_operator indexed;
        main_classes::deflation_continuation_detail::exact_solution_registry<
            indexed_operator,
            double,
            std::vector<double>> indexed_registry(&indexed);
        require(indexed_registry.count() == 3, "indexed count");
        require(indexed_registry.evaluate(1, 7.0, value), "indexed evaluation");
        require(value == std::vector<double>({1.0, 7.0}), "indexed value");
        require(!indexed_registry.evaluate(2, 7.0, value), "indexed unavailable branch");
        require(indexed_registry.name(1) == "branch_1", "indexed name");

        int invalid_count = 0;
        const auto selected = indexed_registry.select(
            std::vector<unsigned int>{2, 7, 0},
            [&invalid_count](unsigned int, std::size_t)
            {
                ++invalid_count;
            });
        require(selected == std::vector<std::size_t>({2, 0}), "selected branches");
        require(invalid_count == 1, "invalid branch callback");

        no_exact_solution absent;
        main_classes::deflation_continuation_detail::exact_solution_registry<
            no_exact_solution,
            double,
            std::vector<double>> absent_registry(&absent);
        require(absent_registry.count() == 0, "absent count");
        require(!absent_registry.evaluate(0, 1.0, value), "absent evaluation");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
