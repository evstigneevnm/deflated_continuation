#ifndef __MAIN_STABILITY_CONTINUATION_SOLVER_BUNDLE_H__
#define __MAIN_STABILITY_CONTINUATION_SOLVER_BUNDLE_H__

#include <memory>
#include <stdexcept>
#include <utility>

namespace main_classes
{
namespace stability_continuation_detail
{

template <class Types>
class solver_bundle
{
public:
    using vector_operations_type    = typename Types::vector_operations_type;
    using nonlinear_operations_type = typename Types::nonlinear_operations_type;
    using log_type                  = typename Types::log_type;
    using linear_operator_type      = typename Types::linear_operator_type;
    using preconditioner_type       = typename Types::preconditioner_type;
    using linear_solver_type        = typename Types::linear_solver_type;
    using convergence_type          = typename Types::convergence_type;
    using system_operator_type      = typename Types::system_operator_type;
    using newton_type               = typename Types::newton_type;

    solver_bundle(
        std::shared_ptr<vector_operations_type> operations, log_type *log, log_type *linear_solver_log,
        std::shared_ptr<nonlinear_operations_type> problem
    )
        : solver_bundle( operations.get(), log, linear_solver_log, problem.get() )
    {
        operations_owner_ = std::move( operations );
        problem_owner_    = std::move( problem );
    }

    // Borrowed construction remains for callers not migrated yet.
    solver_bundle(
        vector_operations_type *operations, log_type *log, log_type *linear_solver_log,
        nonlinear_operations_type *problem
    )
    {
        if ( !operations || !problem )
            throw std::invalid_argument( "stability solver_bundle: missing computational dependency" );

        linear_operator_ = std::make_shared<linear_operator_type>( problem );
        preconditioner_  = std::make_shared<preconditioner_type>( problem );
        linear_solver_   = std::make_shared<linear_solver_type>( operations, linear_solver_log );
        linear_solver_->set_preconditioner( preconditioner_.get() );
        auto *linear_operator = linear_operator_.get();
        auto *linear_solver   = linear_solver_.get();
        convergence_          = std::make_unique<convergence_type>( operations, log );
        auto *convergence     = convergence_.get();
        system_operator_      = std::make_unique<system_operator_type>( operations, linear_operator, linear_solver );
        auto *system_operator = system_operator_.get();
        newton_               = std::make_unique<newton_type>( operations, system_operator, convergence );
    }

    solver_bundle( const solver_bundle & )            = delete;
    solver_bundle &operator=( const solver_bundle & ) = delete;

    linear_operator_type *linear_operator() const
    {
        return linear_operator_.get();
    }
    preconditioner_type *preconditioner() const
    {
        return preconditioner_.get();
    }
    linear_solver_type *linear_solver() const
    {
        return linear_solver_.get();
    }
    convergence_type *convergence() const
    {
        return convergence_.get();
    }
    system_operator_type *system_operator() const
    {
        return system_operator_.get();
    }
    newton_type *newton() const
    {
        return newton_.get();
    }

private:
    std::shared_ptr<vector_operations_type>    operations_owner_;
    std::shared_ptr<nonlinear_operations_type> problem_owner_;
    std::shared_ptr<linear_operator_type>      linear_operator_;
    std::shared_ptr<preconditioner_type>       preconditioner_;
    std::shared_ptr<linear_solver_type>        linear_solver_;
    std::unique_ptr<convergence_type>          convergence_;
    std::unique_ptr<system_operator_type>      system_operator_;
    std::unique_ptr<newton_type>               newton_;
};

} // namespace stability_continuation_detail
} // namespace main_classes

#endif
