#include <iostream>
#include <stdexcept>
#include <string>

#include <numerical_algos/lin_solvers/linear_solve_recovery.h>

namespace
{

void require(
    const bool condition,
    const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

struct recoverable_solver
{
    void set_preconditioner_bypass(const bool value)
    {
        bypass = value;
        ++bypass_changes;
    }

    bool bypass = false;
    unsigned int bypass_changes = 0;
};

struct unsupported_solver
{
};

} // namespace

int main()
{
    try
    {
        recoverable_solver solver;
        unsigned int solves = 0;
        unsigned int resets = 0;
        unsigned int retries = 0;
        const bool recovered =
            numerical_algos::lin_solvers::recovery::
                solve_with_unpreconditioned_retry(
                    &solver,
                    [&]()
                    {
                        ++solves;
                        return solver.bypass;
                    },
                    [&]()
                    {
                        ++resets;
                    },
                    [&]()
                    {
                        ++retries;
                    });
        require(recovered, "unpreconditioned retry succeeds");
        require(solves == 2, "primary and retry are both executed");
        require(resets == 1, "retry output is reset once");
        require(retries == 1, "retry callback executes once");
        require(!solver.bypass, "bypass state is restored");
        require(
            solver.bypass_changes == 2,
            "bypass is enabled and disabled transactionally");

        unsupported_solver unsupported;
        solves = 0;
        const bool unsupported_result =
            numerical_algos::lin_solvers::recovery::
                solve_with_unpreconditioned_retry(
                    &unsupported,
                    [&]()
                    {
                        ++solves;
                        return false;
                    },
                    [](){});
        require(
            !unsupported_result && solves == 1,
            "unsupported solvers retain their original path");
    }
    catch(const std::exception& exception)
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
