#ifndef TIME_STEPPER_TESTS_COMMON_TEST_CONTEXT_H
#define TIME_STEPPER_TESTS_COMMON_TEST_CONTEXT_H

#include <iostream>
#include <string>

namespace time_steppers
{
namespace tests
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

} // namespace tests
} // namespace time_steppers

#endif
