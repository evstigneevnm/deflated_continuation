#ifndef __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__
#define __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <main/solver_construction.h>
#include <main/deflation_continuation/parameter_application.h>

namespace main_classes
{
namespace deflation_continuation_detail
{

template <class Types, class NewtonTypes = void>
class solver_bundle
{
public:
    using vector_operations_type       = typename Types::vector_operations_type;
    using vector_file_operations_type  = typename Types::vector_file_operations_type;
    using log_type                     = typename Types::log_type;
    using nonlinear_operations_type    = typename Types::nonlinear_operations_type;
    using linear_operator_type         = typename Types::linear_operator_type;
    using preconditioner_type          = typename Types::preconditioner_type;
    using linear_system_type           = typename Types::linear_system_type;
    using convergence_type             = typename Types::convergence_type;
    using system_operator_type         = typename Types::system_operator_type;
    using newton_type                  = typename Types::newton_type;
    using knots_type                   = typename Types::knots_type;
    using solution_storage_type        = typename Types::solution_storage_type;
    using continuation_type            = typename Types::continuation_type;
    using analytical_continuation_type = typename Types::analytical_continuation_type;
    using diagram_type                 = typename Types::diagram_type;
    using deflation_type               = typename Types::deflation_type;
    using scalar_type                  = typename vector_operations_type::scalar_type;

    solver_bundle(
        std::shared_ptr<vector_operations_type>      vector_operations,
        std::shared_ptr<vector_file_operations_type> file_operations, log_type *log, log_type *linear_solver_log,
        std::shared_ptr<nonlinear_operations_type> nonlinear_operations, const std::string &project_directory,
        const unsigned int skip_files, std::shared_ptr<solution_storage_type> external_solution_storage = nullptr
    )
        : vector_operations_owner_( std::move( vector_operations ) ),
          file_operations_owner_( std::move( file_operations ) ),
          nonlinear_operations_owner_( std::move( nonlinear_operations ) ),
          external_solution_storage_owner_( std::move( external_solution_storage ) )
    {
        if ( !vector_operations_owner_ || !file_operations_owner_ || !nonlinear_operations_owner_ )
            throw std::invalid_argument( "solver_bundle: missing computational dependency" );
        linear_operator_ = detail::make_problem_component<linear_operator_type>( nonlinear_operations_owner_ );
        preconditioner_  = detail::make_problem_component<preconditioner_type>( nonlinear_operations_owner_ );
        linear_system_   = detail::make_linear_system<linear_system_type>(
            vector_operations_owner_, linear_operator_, preconditioner_, linear_solver_log
        );
        if constexpr ( std::is_void_v<NewtonTypes> )
            initialize(
                vector_operations_owner_.get(), file_operations_owner_.get(), log, nonlinear_operations_owner_.get(),
                project_directory, skip_files, external_solution_storage_owner_.get()
            );
        else
            initialize_nmfd( log, linear_solver_log, project_directory, skip_files );
    }

    // Borrowed construction remains for callers not migrated yet.
    solver_bundle(
        vector_operations_type *vector_operations, vector_file_operations_type *file_operations, log_type *log,
        log_type *linear_solver_log, nonlinear_operations_type *nonlinear_operations,
        const std::string &project_directory, const unsigned int skip_files,
        solution_storage_type *external_solution_storage = nullptr
    )
    {
        if ( !vector_operations || !file_operations || !nonlinear_operations )
            throw std::invalid_argument( "solver_bundle: missing computational dependency" );

        linear_operator_                    = std::make_shared<linear_operator_type>( nonlinear_operations );
        preconditioner_                     = std::make_shared<preconditioner_type>( nonlinear_operations );
        preconditioner_type *preconditioner = preconditioner_.get();
        linear_system_ = std::make_shared<linear_system_type>( preconditioner, vector_operations, linear_solver_log );
        initialize(
            vector_operations, file_operations, log, nonlinear_operations, project_directory, skip_files,
            external_solution_storage
        );
    }

private:
    void initialize_nmfd(
        log_type *log, log_type *linear_solver_log, const std::string &project_directory, unsigned int skip_files
    )
    {
        using fixed_method    = typename NewtonTypes::fixed_method;
        using original_solver = typename linear_system_type::linear_solver_original_type;
        auto original =
            std::shared_ptr<original_solver>( linear_system_, linear_system_->get_linsolver_handle_original() );
        system_operator_ =
            std::make_shared<system_operator_type>( nonlinear_operations_owner_, linear_operator_, scalar_type( 0 ) );
        newton_ = fixed_method::make(
            vector_operations_owner_, nonlinear_operations_owner_, system_operator_, vector_operations_owner_, original,
            log, std::make_shared<typename NewtonTypes::projector_type>( nonlinear_operations_owner_ )
        );
        knots_ = std::make_unique<knots_type>();
        if ( external_solution_storage_owner_ )
            storage_owner_ = external_solution_storage_owner_;
        else
            storage_owner_ = std::make_shared<solution_storage_type>(
                vector_operations_owner_.get(), 50, scalar_type( vector_operations_owner_->get_vector_size() ),
                scalar_type( 2 ), log
            );
        solution_storage_        = storage_owner_.get();
        nmfd_.corrector_operator = std::make_shared<typename NewtonTypes::corrector_operator_type>(
            vector_operations_owner_, nonlinear_operations_owner_, linear_operator_
        );
        nmfd_.bordered = NewtonTypes::make_bordered(
            nmfd_.corrector_operator->space(), linear_system_, preconditioner_, linear_solver_log
        );
        nmfd_.tangent   = NewtonTypes::make_tangent( nmfd_.corrector_operator, nmfd_.bordered );
        nmfd_.corrector = NewtonTypes::continuation_method::make(
            vector_operations_owner_, nonlinear_operations_owner_, nmfd_.corrector_operator,
            nmfd_.corrector_operator->space(), nmfd_.bordered, log,
            std::make_shared<typename NewtonTypes::corrector_operator_type::trial_projector>( nmfd_.corrector_operator )
        );
        nmfd_.deflated_operator = std::make_shared<typename NewtonTypes::deflated_operator_type>(
            vector_operations_owner_, nonlinear_operations_owner_, linear_operator_, storage_owner_, scalar_type( 0 ),
            scalar_type( 0 )
        );
        nmfd_.deflated = NewtonTypes::deflation_method::make(
            vector_operations_owner_, nonlinear_operations_owner_, nmfd_.deflated_operator, vector_operations_owner_,
            std::make_shared<typename NewtonTypes::rank_one_type>( linear_system_ ), log,
            std::make_shared<typename NewtonTypes::deflated_operator_type::trial_projector>(
                nonlinear_operations_owner_
            )
        );
        auto *ops     = vector_operations_owner_.get();
        auto *files   = file_operations_owner_.get();
        auto *problem = nonlinear_operations_owner_.get();
        auto *op      = linear_operator_.get();
        auto *system  = linear_system_.get();
        auto *knots   = knots_.get();
        auto *fixed   = newton_.get();
        continuation_ = std::make_unique<continuation_type>(
            ops, files, log, problem, op, knots, system, fixed, nmfd_.corrector, nmfd_.tangent
        );
        analytical_continuation_ = std::make_unique<analytical_continuation_type>(
            ops, files, log, problem, op, knots, system, fixed, nmfd_.corrector, nmfd_.tangent
        );
        diagram_ = std::make_unique<diagram_type>( ops, files, log, problem, fixed, project_directory, skip_files );
        deflation_ =
            std::make_unique<deflation_type>( ops, files, log, problem, op, system, solution_storage_, nmfd_.deflated );
    }

    void initialize(
        vector_operations_type *vector_operations, vector_file_operations_type *file_operations, log_type *log,
        nonlinear_operations_type *nonlinear_operations, const std::string &project_directory, unsigned int skip_files,
        solution_storage_type *external_solution_storage
    )
    {
        linear_operator_type *linear_operator = linear_operator_.get();
        linear_system_type   *linear_system   = linear_system_.get();
        convergence_                          = std::make_shared<convergence_type>( vector_operations, log );
        convergence_type *convergence         = convergence_.get();
        system_operator_ = std::make_shared<system_operator_type>( vector_operations, linear_operator, linear_system );
        system_operator_type *system_operator = system_operator_.get();
        newton_             = std::make_shared<newton_type>( vector_operations, system_operator, convergence );
        newton_type *newton = newton_.get();
        knots_              = std::make_unique<knots_type>();
        knots_type *knots   = knots_.get();

        if ( external_solution_storage != nullptr )
        {
            solution_storage_ = external_solution_storage;
        }
        else
        {
            owned_solution_storage_ = std::make_unique<solution_storage_type>(
                vector_operations, 50, vector_operations->get_l2_size(), scalar_type( 2 ), log
            );
            solution_storage_ = owned_solution_storage_.get();
        }

        continuation_ = std::make_unique<continuation_type>(
            vector_operations, file_operations, log, nonlinear_operations, linear_operator, knots, linear_system, newton
        );
        analytical_continuation_ = std::make_unique<analytical_continuation_type>(
            vector_operations, file_operations, log, nonlinear_operations, linear_operator, knots, linear_system, newton
        );
        diagram_ = std::make_unique<diagram_type>(
            vector_operations, file_operations, log, nonlinear_operations, newton, project_directory, skip_files
        );
        deflation_ = std::make_unique<deflation_type>(
            vector_operations, file_operations, log, nonlinear_operations, linear_operator, linear_system,
            solution_storage_
        );
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
    linear_system_type *linear_system() const
    {
        return linear_system_.get();
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
    knots_type *knots() const
    {
        return knots_.get();
    }
    solution_storage_type *solution_storage() const
    {
        return solution_storage_;
    }
    continuation_type *continuation() const
    {
        return continuation_.get();
    }
    analytical_continuation_type *analytical_continuation() const
    {
        return analytical_continuation_.get();
    }
    diagram_type *diagram() const
    {
        return diagram_.get();
    }
    deflation_type *deflation() const
    {
        return deflation_.get();
    }
    bool owns_solution_storage() const
    {
        return owned_solution_storage_ != nullptr || ( storage_owner_ && !external_solution_storage_owner_ );
    }

    template <class Parameters>
    void configure_augmented( const Parameters &parameters )
    {
        if constexpr ( !std::is_void_v<NewtonTypes> )
            if constexpr ( !std::is_void_v<typename NewtonTypes::tangent_solver_type> )
                configure_linear_solver<scalar_type>( &nmfd_.bordered->krylov_solver(), parameters );
    }

    template <class Parameters>
    void configure_continuation( const Parameters &parameters )
    {
        NewtonTypes::configure_continuation( nmfd_.corrector->get_convergence_strategy_handle(), parameters );
    }
    template <class Parameters>
    void configure_deflation( const Parameters &parameters )
    {
        NewtonTypes::configure_deflation( nmfd_.deflated->get_convergence_strategy_handle(), parameters );
        nmfd_.deflated_operator->set_exclusion_distance( scalar_type( 100 ) * parameters.tolerance );
    }

private:
    // Declared first so borrowed consumers are destroyed before their dependencies.
    std::shared_ptr<vector_operations_type>       vector_operations_owner_;
    std::shared_ptr<vector_file_operations_type>  file_operations_owner_;
    std::shared_ptr<nonlinear_operations_type>    nonlinear_operations_owner_;
    std::shared_ptr<solution_storage_type>        external_solution_storage_owner_;
    std::shared_ptr<linear_operator_type>         linear_operator_;
    std::shared_ptr<preconditioner_type>          preconditioner_;
    std::shared_ptr<linear_system_type>           linear_system_;
    std::shared_ptr<convergence_type>             convergence_;
    std::shared_ptr<system_operator_type>         system_operator_;
    std::shared_ptr<newton_type>                  newton_;
    std::unique_ptr<knots_type>                   knots_;
    std::unique_ptr<solution_storage_type>        owned_solution_storage_;
    solution_storage_type                        *solution_storage_ = nullptr;
    std::shared_ptr<solution_storage_type>        storage_owner_;
    detail::nonlinear_solver_state<NewtonTypes>   nmfd_;
    std::unique_ptr<continuation_type>            continuation_;
    std::unique_ptr<analytical_continuation_type> analytical_continuation_;
    std::unique_ptr<diagram_type>                 diagram_;
    std::unique_ptr<deflation_type>               deflation_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_SOLVER_BUNDLE_H__
