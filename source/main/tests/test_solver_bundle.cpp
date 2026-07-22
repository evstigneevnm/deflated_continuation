#include <iostream>
#include <stdexcept>
#include <string>

#include <main/deflation_continuation/solver_bundle.h>

namespace
{

struct vector_operations
{
    using scalar_type = double;
    int get_l2_size() const { return 8; }
};

struct file_operations {};
struct log_type {};
struct nonlinear_operations {};

struct linear_operator
{
    explicit linear_operator(nonlinear_operations*&) {}
};

struct preconditioner
{
    explicit preconditioner(nonlinear_operations*&) {}
};

struct linear_system
{
    linear_system(preconditioner*&, vector_operations*&, log_type*&) {}
};

struct convergence
{
    convergence(vector_operations*&, log_type*&) {}
};

struct system_operator
{
    system_operator(vector_operations*&, linear_operator*&, linear_system*&) {}
};

struct newton
{
    newton(vector_operations*&, system_operator*&, convergence*&) {}
};

struct knots {};

struct solution_storage
{
    solution_storage()
    {
        ++alive;
    }

    solution_storage(vector_operations*&, int, int, double, log_type*&)
    {
        ++alive;
    }

    ~solution_storage()
    {
        --alive;
    }

    static int alive;
};

int solution_storage::alive = 0;

struct continuation
{
    continuation(
        vector_operations*&,
        file_operations*&,
        log_type*&,
        nonlinear_operations*&,
        linear_operator*&,
        knots*&,
        linear_system*&,
        newton*&)
    {
    }
};

struct diagram
{
    diagram(
        vector_operations*&,
        file_operations*&,
        log_type*&,
        nonlinear_operations*&,
        newton*&,
        const std::string&,
        unsigned int)
    {
    }
};

struct deflation
{
    deflation(
        vector_operations*&,
        file_operations*&,
        log_type*&,
        nonlinear_operations*&,
        linear_operator*&,
        linear_system*&,
        solution_storage*&)
    {
    }
};

struct types
{
    using vector_operations_type = vector_operations;
    using vector_file_operations_type = file_operations;
    using log_type = ::log_type;
    using nonlinear_operations_type = nonlinear_operations;
    using linear_operator_type = linear_operator;
    using preconditioner_type = preconditioner;
    using linear_system_type = linear_system;
    using convergence_type = convergence;
    using system_operator_type = system_operator;
    using newton_type = newton;
    using knots_type = knots;
    using solution_storage_type = solution_storage;
    using continuation_type = continuation;
    using analytical_continuation_type = continuation;
    using diagram_type = diagram;
    using deflation_type = deflation;
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
    using bundle_type =
        main_classes::deflation_continuation_detail::solver_bundle<types>;

    try
    {
        vector_operations vector_ops;
        file_operations file_ops;
        log_type log;
        nonlinear_operations nonlinear_ops;

        {
            bundle_type bundle(
                &vector_ops,
                &file_ops,
                &log,
                &log,
                &nonlinear_ops,
                "project",
                3);
            require(bundle.owns_solution_storage(), "internal storage ownership");
            require(solution_storage::alive == 1, "internal storage lifetime");
            require(bundle.linear_operator() != nullptr, "linear operator handle");
            require(bundle.continuation() != nullptr, "continuation handle");
            require(bundle.analytical_continuation() != nullptr, "analytical continuation handle");
            require(bundle.diagram() != nullptr, "diagram handle");
            require(bundle.deflation() != nullptr, "deflation handle");
        }
        require(solution_storage::alive == 0, "internal storage cleanup");

        {
            solution_storage external;
            require(solution_storage::alive == 1, "external storage setup");
            {
                bundle_type bundle(
                    &vector_ops,
                    &file_ops,
                    &log,
                    &log,
                    &nonlinear_ops,
                    "project",
                    3,
                    &external);
                require(!bundle.owns_solution_storage(), "external storage ownership");
                require(bundle.solution_storage() == &external, "external storage handle");
            }
            require(solution_storage::alive == 1, "external storage survives bundle");
        }
        require(solution_storage::alive == 0, "external storage final cleanup");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
