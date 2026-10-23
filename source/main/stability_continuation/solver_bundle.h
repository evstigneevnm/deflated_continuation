#ifndef __MAIN_STABILITY_CONTINUATION_SOLVER_BUNDLE_H__
#define __MAIN_STABILITY_CONTINUATION_SOLVER_BUNDLE_H__

#include <memory>
#include <stdexcept>
#include <utility>
#include <main/solver_construction.h>

namespace main_classes
{
namespace stability_continuation_detail
{

template <class Types, class NewtonTypes = void>
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
        : operations_owner_( std::move( operations ) ), problem_owner_( std::move( problem ) )
    {
        if ( !operations_owner_ || !problem_owner_ )
            throw std::invalid_argument( "stability solver_bundle: missing computational dependency" );
        linear_operator_ = detail::make_problem_component<linear_operator_type>( problem_owner_ );
        preconditioner_  = detail::make_problem_component<preconditioner_type>( problem_owner_ );
        linear_solver_   = detail::make_linear_solver<linear_solver_type>(
            operations_owner_, linear_operator_, preconditioner_, linear_solver_log
        );
        initialize( operations_owner_.get(), log );
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
        initialize( operations, log );
    }

    // Retain an already configured solver, e.g. the transition GMRES fallback.
    solver_bundle(
        std::shared_ptr<vector_operations_type> operations, log_type *log,
        std::shared_ptr<nonlinear_operations_type> problem, std::shared_ptr<linear_operator_type> op,
        std::shared_ptr<preconditioner_type> prec, std::shared_ptr<linear_solver_type> solver
    )
        : operations_owner_( std::move( operations ) ), problem_owner_( std::move( problem ) ),
          linear_operator_( std::move( op ) ), preconditioner_( std::move( prec ) ),
          linear_solver_( std::move( solver ) )
    {
        if ( !operations_owner_ || !problem_owner_ || !linear_operator_ || !preconditioner_ || !linear_solver_ )
            throw std::invalid_argument( "stability solver_bundle: missing computational dependency" );
        initialize( operations_owner_.get(), log );
    }

private:
    void initialize( vector_operations_type *operations, log_type *log )
    {
        if constexpr ( !std::is_void_v<NewtonTypes> )
        {
            system_operator_ = std::make_shared<system_operator_type>(
                problem_owner_, linear_operator_, typename vector_operations_type::scalar_type( 0 )
            );
            newton_ = NewtonTypes::fixed_method::make(
                operations_owner_, problem_owner_, system_operator_, operations_owner_, linear_solver_, log,
                std::make_shared<typename NewtonTypes::projector_type>( problem_owner_ )
            );
        }
        else
        {
            auto *linear_operator = linear_operator_.get();
            auto *linear_solver   = linear_solver_.get();
            convergence_          = std::make_shared<convergence_type>( operations, log );
            auto *convergence     = convergence_.get();
            system_operator_ = std::make_shared<system_operator_type>( operations, linear_operator, linear_solver );
            auto *system_operator = system_operator_.get();
            newton_               = std::make_shared<newton_type>( operations, system_operator, convergence );
        }
    }

public:
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
        if constexpr ( std::is_void_v<NewtonTypes> )
            return convergence_.get();
        else
            return newton_->get_convergence_strategy_handle();
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
    std::shared_ptr<convergence_type>          convergence_;
    std::shared_ptr<system_operator_type>      system_operator_;
    std::shared_ptr<newton_type>               newton_;
};

} // namespace stability_continuation_detail
} // namespace main_classes

#endif
