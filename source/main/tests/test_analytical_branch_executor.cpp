#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <main/deflation_continuation/analytical_branch_executor.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector(vector_type&) {}
    void start_use_vector(vector_type&) { ++active_vectors; }
    void stop_use_vector(vector_type&) { --active_vectors; }
    void free_vector(vector_type& value) { value.clear(); }

    int active_vectors = 0;
};

struct fake_log
{
    void warning(const char*) {}
    template<class... Args> void warning_f(const char*, Args&&...) {}
    template<class... Args> void info_f(const char*, Args&&...) {}
};

struct fake_registry
{
    std::size_t count() const { return 2; }
    bool evaluate(std::size_t branch, const double& parameter, std::vector<double>& value)
    {
        value = {static_cast<double>(branch), parameter};
        return true;
    }
    std::string name(std::size_t branch) const
    {
        return "exact_" + std::to_string(branch);
    }
    template<class Integer, class Invalid>
    std::vector<std::size_t> select(const std::vector<Integer>& requested, Invalid&& invalid) const
    {
        std::vector<std::size_t> result;
        for(const auto id: requested)
        {
            if(static_cast<std::size_t>(id) >= count())
            {
                invalid(id, count());
            }
            else
            {
                result.push_back(static_cast<std::size_t>(id));
            }
        }
        return result;
    }
};

struct fake_curve
{
};

struct fake_continuation
{
    using provider_type = std::function<bool(const double&, std::vector<double>&)>;

    void set_exact_solution_provider(provider_type provider_)
    {
        provider = std::move(provider_);
    }
    void clear_exact_solution_provider()
    {
        provider = {};
        ++clear_calls;
    }
    bool continuate_curve(fake_curve*, std::vector<double>& value, const double& parameter)
    {
        ++continuation_calls;
        std::vector<double> next;
        if(!provider || !provider(parameter + 1.0, next))
        {
            return false;
        }
        value = next;
        return succeeds;
    }

    provider_type provider;
    bool succeeds = true;
    int continuation_calls = 0;
    int clear_calls = 0;
};

struct fake_diagram
{
    void init_new_curve() { ++initialized; }
    void get_current_ref(fake_curve*& output) { output = &curve; }
    void close_curve() { ++closed; }
    bool commit_current_curve_symmetry_events() { ++committed; return true; }
    void discard_current_curve() { ++discarded; }

    fake_curve curve;
    int initialized = 0;
    int closed = 0;
    int committed = 0;
    int discarded = 0;
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
        fake_vector_operations vector_operations;
        fake_log log;
        fake_registry registry;
        fake_continuation continuation;
        fake_diagram diagram;
        using executor_type =
            main_classes::deflation_continuation_detail::analytical_branch_executor<
                fake_vector_operations,
                fake_log,
                fake_registry,
                fake_continuation,
                fake_diagram,
                fake_curve>;
        executor_type executor(
            &vector_operations,
            &log,
            &registry,
            &continuation,
            &diagram);

        int stabilized = 0;
        int saved = 0;
        const bool success = executor.build_if_available(
            false,
            true,
            std::vector<unsigned int>{1, 9},
            4.0,
            false,
            [&stabilized](std::vector<double>&) { ++stabilized; },
            [&saved]() { ++saved; });
        require(success, "analytical branch success");
        require(continuation.continuation_calls == 1, "selected branch count");
        require(continuation.clear_calls == 1 && !continuation.provider, "provider cleanup");
        require(diagram.committed == 1 && diagram.discarded == 0, "accepted curve");
        require(stabilized == 1 && saved == 1, "callbacks");
        require(vector_operations.active_vectors == 0, "vector cleanup");

        continuation.succeeds = false;
        const bool failed = executor.build_if_available(
            false,
            true,
            std::vector<unsigned int>{0},
            4.0,
            false,
            [](std::vector<double>&) {},
            []() {});
        require(!failed, "failed analytical continuation result");
        require(diagram.discarded == 1, "failed curve discarded");

        const int calls_before_skip = continuation.continuation_calls;
        require(
            !executor.build_if_available(
                true,
                true,
                std::vector<unsigned int>{0},
                4.0,
                false,
                [](std::vector<double>&) {},
                []() {}),
            "archive restart skips analytical branches");
        require(continuation.continuation_calls == calls_before_skip, "skip has no continuation");
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
