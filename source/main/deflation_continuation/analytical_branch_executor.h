#ifndef __MAIN_DEFLATION_CONTINUATION_ANALYTICAL_BRANCH_EXECUTOR_H__
#define __MAIN_DEFLATION_CONTINUATION_ANALYTICAL_BRANCH_EXECUTOR_H__

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace main_classes
{
namespace deflation_continuation_detail
{

template<
    class VectorOperations,
    class Log,
    class ExactSolutionRegistry,
    class AnalyticalContinuation,
    class Diagram,
    class Curve>
class analytical_branch_executor
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    analytical_branch_executor(
        VectorOperations* vector_operations,
        Log* log,
        ExactSolutionRegistry* exact_solutions,
        AnalyticalContinuation* continuation,
        Diagram* diagram):
        vector_operations_(vector_operations),
        log_(log),
        exact_solutions_(exact_solutions),
        continuation_(continuation),
        diagram_(diagram)
    {
    }

    template<class BranchId, class Stabilize, class Save>
    bool build_if_available(
        const bool archive_exists,
        const bool enabled,
        const std::vector<BranchId>& requested_branches,
        const scalar_type& initial_parameter,
        const bool allow_failed_curve_save,
        Stabilize&& stabilize,
        Save&& save)
    {
        if(archive_exists || !enabled)
        {
            return false;
        }

        const std::size_t branch_count = exact_solutions_->count();
        if(branch_count == 0)
        {
            log_->warning(
                "MAIN:deflation_continuation: analytical branch requested, but nonlinear operator has no compatible exact solution registry; skipping analytical branch.");
            return false;
        }

        provider_guard guard(continuation_);
        const auto branches = exact_solutions_->select(
            requested_branches,
            [this](const BranchId branch_id, const std::size_t count)
            {
                log_->warning_f(
                    "MAIN:deflation_continuation: requested analytical branch %u is outside available branch count %llu; skipping it.",
                    static_cast<unsigned int>(branch_id),
                    static_cast<unsigned long long>(count));
            });

        bool any_success = false;
        for(const std::size_t branch_id: branches)
        {
            any_success = build_branch(
                branch_id,
                initial_parameter,
                allow_failed_curve_save,
                stabilize,
                save) || any_success;
        }
        return any_success;
    }

private:
    template<class Stabilize, class Save>
    bool build_branch(
        const std::size_t branch_id,
        const scalar_type& initial_parameter,
        const bool allow_failed_curve_save,
        Stabilize& stabilize,
        Save& save)
    {
        owned_vector exact_value(vector_operations_);
        if(!exact_solutions_->evaluate(
               branch_id,
               initial_parameter,
               exact_value.get()))
        {
            log_->warning_f(
                "MAIN:deflation_continuation: analytical branch %llu is not defined at initial lambda = %le; skipping it.",
                static_cast<unsigned long long>(branch_id),
                double(initial_parameter));
            return false;
        }

        log_->info_f(
            "MAIN:deflation_continuation: building analytical branch %llu (%s) as an ordinary curve...",
            static_cast<unsigned long long>(branch_id),
            exact_solutions_->name(branch_id).c_str());
        stabilize(exact_value.get());
        continuation_->set_exact_solution_provider(
            [this, branch_id](
                const scalar_type& parameter,
                vector_type& value) -> bool
            {
                return exact_solutions_->evaluate(branch_id, parameter, value);
            });

        Curve* curve = nullptr;
        diagram_->init_new_curve();
        diagram_->get_current_ref(curve);
        const bool success = continuation_->continuate_curve(
            curve,
            exact_value.get(),
            initial_parameter);
        diagram_->close_curve();
        if(success || allow_failed_curve_save)
        {
            if(!diagram_->commit_current_curve_symmetry_events())
            {
                throw std::runtime_error(
                    "MAIN:deflation_continuation: failed to commit symmetry events for an accepted analytical curve");
            }
            save();
        }
        else
        {
            log_->warning_f(
                "MAIN:deflation_continuation: analytical branch %llu continuation failed; discarding curve according to restart policy.",
                static_cast<unsigned long long>(branch_id));
            diagram_->discard_current_curve();
        }

        log_->info_f(
            "MAIN:deflation_continuation: analytical branch %llu construction finished.",
            static_cast<unsigned long long>(branch_id));
        return success;
    }

    class owned_vector
    {
    public:
        explicit owned_vector(VectorOperations* vector_operations):
            vector_operations_(vector_operations)
        {
            vector_operations_->init_vector(value_);
            vector_operations_->start_use_vector(value_);
        }

        ~owned_vector()
        {
            vector_operations_->stop_use_vector(value_);
            vector_operations_->free_vector(value_);
        }

        vector_type& get()
        {
            return value_;
        }

    private:
        VectorOperations* vector_operations_;
        vector_type value_;
    };

    class provider_guard
    {
    public:
        explicit provider_guard(AnalyticalContinuation* continuation):
            continuation_(continuation)
        {
        }

        ~provider_guard()
        {
            continuation_->clear_exact_solution_provider();
        }

    private:
        AnalyticalContinuation* continuation_;
    };

    VectorOperations* vector_operations_;
    Log* log_;
    ExactSolutionRegistry* exact_solutions_;
    AnalyticalContinuation* continuation_;
    Diagram* diagram_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_ANALYTICAL_BRANCH_EXECUTOR_H__
