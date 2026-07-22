#ifndef __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__
#define __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__

#include <memory>
#include <string>

namespace main_classes
{
namespace deflation_continuation_detail
{

template<class Types>
class solver_bundle
{
public:
    using vector_operations_type = typename Types::vector_operations_type;
    using vector_file_operations_type = typename Types::vector_file_operations_type;
    using log_type = typename Types::log_type;
    using nonlinear_operations_type = typename Types::nonlinear_operations_type;
    using linear_operator_type = typename Types::linear_operator_type;
    using preconditioner_type = typename Types::preconditioner_type;
    using linear_system_type = typename Types::linear_system_type;
    using convergence_type = typename Types::convergence_type;
    using system_operator_type = typename Types::system_operator_type;
    using newton_type = typename Types::newton_type;
    using knots_type = typename Types::knots_type;
    using solution_storage_type = typename Types::solution_storage_type;
    using continuation_type = typename Types::continuation_type;
    using analytical_continuation_type = typename Types::analytical_continuation_type;
    using diagram_type = typename Types::diagram_type;
    using deflation_type = typename Types::deflation_type;
    using scalar_type = typename vector_operations_type::scalar_type;

    solver_bundle(
        vector_operations_type* vector_operations,
        vector_file_operations_type* file_operations,
        log_type* log,
        log_type* linear_solver_log,
        nonlinear_operations_type* nonlinear_operations,
        const std::string& project_directory,
        const unsigned int skip_files,
        solution_storage_type* external_solution_storage = nullptr)
    {
        linear_operator_ = std::make_unique<linear_operator_type>(nonlinear_operations);
        linear_operator_type* linear_operator = linear_operator_.get();
        preconditioner_ = std::make_unique<preconditioner_type>(nonlinear_operations);
        preconditioner_type* preconditioner = preconditioner_.get();
        linear_system_ = std::make_unique<linear_system_type>(
            preconditioner,
            vector_operations,
            linear_solver_log);
        linear_system_type* linear_system = linear_system_.get();
        convergence_ = std::make_unique<convergence_type>(vector_operations, log);
        convergence_type* convergence = convergence_.get();
        system_operator_ = std::make_unique<system_operator_type>(
            vector_operations,
            linear_operator,
            linear_system);
        system_operator_type* system_operator = system_operator_.get();
        newton_ = std::make_unique<newton_type>(
            vector_operations,
            system_operator,
            convergence);
        newton_type* newton = newton_.get();
        knots_ = std::make_unique<knots_type>();
        knots_type* knots = knots_.get();

        if(external_solution_storage != nullptr)
        {
            solution_storage_ = external_solution_storage;
        }
        else
        {
            owned_solution_storage_ = std::make_unique<solution_storage_type>(
                vector_operations,
                50,
                vector_operations->get_l2_size(),
                scalar_type(2),
                log);
            solution_storage_ = owned_solution_storage_.get();
        }

        continuation_ = std::make_unique<continuation_type>(
            vector_operations,
            file_operations,
            log,
            nonlinear_operations,
            linear_operator,
            knots,
            linear_system,
            newton);
        analytical_continuation_ = std::make_unique<analytical_continuation_type>(
            vector_operations,
            file_operations,
            log,
            nonlinear_operations,
            linear_operator,
            knots,
            linear_system,
            newton);
        diagram_ = std::make_unique<diagram_type>(
            vector_operations,
            file_operations,
            log,
            nonlinear_operations,
            newton,
            project_directory,
            skip_files);
        deflation_ = std::make_unique<deflation_type>(
            vector_operations,
            file_operations,
            log,
            nonlinear_operations,
            linear_operator,
            linear_system,
            solution_storage_);
    }

    solver_bundle(const solver_bundle&) = delete;
    solver_bundle& operator=(const solver_bundle&) = delete;

    linear_operator_type* linear_operator() const { return linear_operator_.get(); }
    preconditioner_type* preconditioner() const { return preconditioner_.get(); }
    linear_system_type* linear_system() const { return linear_system_.get(); }
    convergence_type* convergence() const { return convergence_.get(); }
    system_operator_type* system_operator() const { return system_operator_.get(); }
    newton_type* newton() const { return newton_.get(); }
    knots_type* knots() const { return knots_.get(); }
    solution_storage_type* solution_storage() const { return solution_storage_; }
    continuation_type* continuation() const { return continuation_.get(); }
    analytical_continuation_type* analytical_continuation() const
    {
        return analytical_continuation_.get();
    }
    diagram_type* diagram() const { return diagram_.get(); }
    deflation_type* deflation() const { return deflation_.get(); }
    bool owns_solution_storage() const
    {
        return owned_solution_storage_ != nullptr;
    }

private:
    std::unique_ptr<linear_operator_type> linear_operator_;
    std::unique_ptr<preconditioner_type> preconditioner_;
    std::unique_ptr<linear_system_type> linear_system_;
    std::unique_ptr<convergence_type> convergence_;
    std::unique_ptr<system_operator_type> system_operator_;
    std::unique_ptr<newton_type> newton_;
    std::unique_ptr<knots_type> knots_;
    std::unique_ptr<solution_storage_type> owned_solution_storage_;
    solution_storage_type* solution_storage_ = nullptr;
    std::unique_ptr<continuation_type> continuation_;
    std::unique_ptr<analytical_continuation_type> analytical_continuation_;
    std::unique_ptr<diagram_type> diagram_;
    std::unique_ptr<deflation_type> deflation_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__
