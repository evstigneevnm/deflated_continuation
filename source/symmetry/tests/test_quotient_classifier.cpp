#include <iostream>
#include <string>

#include <symmetry/quotient_classifier.h>

namespace
{

int checks = 0;
int failures = 0;

void require(bool condition, const std::string& label)
{
    ++checks;
    if(!condition)
    {
        std::cout << "FAIL " << label << std::endl;
        ++failures;
    }
}

void require_kind(
    const std::string& label,
    symmetry::quotient_solution_kind value,
    symmetry::quotient_solution_kind expected)
{
    ++checks;
    if(value != expected)
    {
        std::cout << "FAIL " << label << " value="
                  << symmetry::quotient_solution_kind_name(value)
                  << " expected="
                  << symmetry::quotient_solution_kind_name(expected)
                  << std::endl;
        ++failures;
    }
}

} // namespace

int main()
{
    {
        const auto report = symmetry::classify_quotient_residual(2e-8, 1e-12, 1e-8, 1e-10);
        require_kind("not converged", report.kind, symmetry::quotient_solution_kind::not_converged);
        require(!report.quotient_converged(), "not converged quotient flag");
    }

    {
        const auto report = symmetry::classify_quotient_residual(1e-12, 5e-12, 1e-8, 1e-10);
        require_kind("stationary", report.kind, symmetry::quotient_solution_kind::stationary);
        require(report.quotient_converged(), "stationary quotient flag");
        require(report.stationary(), "stationary flag");
    }

    {
        const auto report = symmetry::classify_quotient_residual(1e-12, 1e-4, 1e-8, 1e-10);
        require_kind("relative equilibrium", report.kind, symmetry::quotient_solution_kind::relative_equilibrium);
        require(report.quotient_converged(), "relative quotient flag");
        require(report.relative_equilibrium(), "relative flag");
    }

    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if(failures != 0)
    {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
