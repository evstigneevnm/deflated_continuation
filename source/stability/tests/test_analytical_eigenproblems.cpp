#include <cstdlib>
#include <iostream>
#include <string>

#include "common/analytical_eigenproblem.h"
#include "common/eigensolver_test_harness.h"

namespace
{

std::size_t checks = 0;
std::size_t failures = 0;

void require(bool condition, const std::string& message)
{
    ++checks;
    if(!condition)
    {
        ++failures;
        std::cout << "FAIL " << message << std::endl;
    }
}

template<class Real>
void test_reference_problems(const std::string& label)
{
    const auto problems = stability::tests::analytical_eigenproblems<Real>();
    require(problems.size() == 5, label + " fixture count");

    for(const auto& problem : problems)
    {
        const auto result = stability::tests::exact_reference_result(problem);
        const auto report = stability::tests::validate_eigensolver_result(problem, result);
        checks += report.checks;
        for(const auto& failure : report.failures)
        {
            ++failures;
            std::cout << "FAIL " << label << " " << failure << std::endl;
        }
    }
}

void test_harness_rejects_bad_results()
{
    using namespace stability::tests;
    const auto problem = symmetric_eigenproblem<double>();

    {
        auto result = exact_reference_result(problem);
        result.eigenpairs[0].value += 0.25;
        const auto report = validate_eigensolver_result(problem, result);
        require(!report.passed(), "harness rejects a wrong eigenvalue");
    }

    {
        auto result = exact_reference_result(problem);
        result.eigenpairs[0].right_eigenvector = {{1.0, 0.0}, {0.0, 0.0}};
        const auto report = validate_eigensolver_result(problem, result);
        require(!report.passed(), "harness rejects a wrong eigenvector");
    }

    {
        auto result = exact_reference_result(problem);
        result.operator_calls = 0;
        const auto report = validate_eigensolver_result(problem, result);
        require(!report.passed(), "harness rejects inconsistent solver metadata");
    }
}

} // namespace

int main()
{
    test_reference_problems<float>("float");
    test_reference_problems<double>("double");
    test_harness_rejects_bad_results();

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
