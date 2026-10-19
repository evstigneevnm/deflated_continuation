#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined( TEST_WORKFLOW_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_WORKFLOW_OMP )
#    include <scfd/backend/omp.h>
#else
#    include <scfd/backend/serial_cpu.h>
#endif
#include <scfd/utils/log.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <common/gpu_file_operations.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/system_operator.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <main/parameters.hpp>
#include <main/deflation_continuation.hpp>
#include <main/stability_continuation.hpp>
#include <stability/eigensolvers/direct_scalar_eigensolver.h>
#include <stability/eigensolvers/transformations/scaled_real_operator.h>

namespace
{

#if defined( TEST_WORKFLOW_CUDA )
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_WORKFLOW_OMP )
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif

int checks = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

class vector_space : public nmfd::operations::scfd_vector_space<backend_type, double>
{
    using parent_type = nmfd::operations::scfd_vector_space<backend_type, double>;

public:
    using typename parent_type::vector_type;
    inline static bool invalid_cleanup = false;
    mutable int        allocations     = 0;
    mutable int        frees           = 0;
    mutable int        activations     = 0;
    int                fail_activation = -1;

    vector_space() : parent_type( 1 )
    {
    }
    ~vector_space()
    {
        invalid_cleanup = invalid_cleanup || allocations != frees;
    }
    void init_vector( vector_type &v ) const
    {
        parent_type::init_vector( v );
        ++allocations;
    }
    void free_vector( vector_type &v ) const
    {
        parent_type::free_vector( v );
        ++frees;
    }
    void start_use_vector( vector_type &v ) const
    {
        if ( ++activations == fail_activation )
            throw std::runtime_error( "injected workspace activation failure" );
        parent_type::start_use_vector( v );
    }
};

using file_operations_type = gpu_file_operations<vector_space>;
using log_type             = scfd::utils::log_std;
using problem_type         = nonlinear_operators::circle<vector_space, 64>;
using linear_operator_type = nonlinear_operators::linear_operator_circle<vector_space, problem_type>;
using preconditioner_type =
    nonlinear_operators::preconditioner_circle<vector_space, problem_type, linear_operator_type>;
using monitor_type    = numerical_algos::lin_solvers::default_monitor<vector_space, log_type>;
using parameters_type = main_classes::parameters<double>;
using scaled_operator_type =
    stability::eigensolvers::transformations::scaled_real_operator<vector_space, linear_operator_type>;
using scalar_eigensolver_type = stability::eigensolvers::direct_scalar_eigensolver<vector_space, scaled_operator_type>;

static_assert(
    std::is_constructible<problem_type, double, std::size_t, std::shared_ptr<vector_space>>::value,
    "circle construction must retain shared vector operations"
);
static_assert(
    !std::is_constructible<problem_type, double, std::size_t, vector_space *>::value,
    "circle construction must not accept borrowed vector operations"
);

struct eigensolver
{
    using result_type = stability::eigensolvers::eigensolver_result<double>;
    result_type execute( const vector_space::vector_type & ) const
    {
        result_type result;
        result.status            = stability::eigensolvers::eigensolver_status::success;
        result.coverage_complete = true;
        return result;
    }
};

struct linearization_provider
{
    std::shared_ptr<problem_type> problem;
    explicit linearization_provider( std::shared_ptr<problem_type> problem ) : problem( std::move( problem ) )
    {
    }
    void set_linearization_point( const vector_space::vector_type &state, double parameter )
    {
        problem->set_linearization_point( state, parameter );
    }
};

using deflation_workflow = main_classes::deflation_continuation<
    vector_space, file_operations_type, log_type, monitor_type, problem_type, linear_operator_type, preconditioner_type,
    numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_type>;
using stability_workflow = main_classes::stability_continuation<
    vector_space, file_operations_type, log_type, monitor_type, problem_type, linear_operator_type, preconditioner_type,
    numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_type, eigensolver>;
using custom_stability_workflow = main_classes::stability_continuation<
    vector_space, file_operations_type, log_type, monitor_type, problem_type, linear_operator_type, preconditioner_type,
    numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_type, eigensolver,
    linearization_provider>;

struct fixture
{
    log_type                              log;
    std::shared_ptr<vector_space>         operations = std::make_shared<vector_space>();
    std::shared_ptr<file_operations_type> files      = std::make_shared<file_operations_type>( operations.get() );
    std::shared_ptr<problem_type>         problem    = std::make_shared<problem_type>( 1.0, 1, operations );
    std::shared_ptr<parameters_type>      parameters = std::make_shared<parameters_type>();
    std::shared_ptr<eigensolver>          adapter    = std::make_shared<eigensolver>();

    fixture()
    {
        log.set_verbosity( 0 );
        parameters->set_default();
        parameters->stability_continuation.classification_uncertainty_registry.enabled = false;
        const auto id = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        parameters->path_to_project =
            ( std::filesystem::temp_directory_path() / ( "nmfd_workflow_ownership_" + std::to_string( id ) ) ).string();
    }

    void release()
    {
        adapter.reset();
        parameters.reset();
        problem.reset();
        files.reset();
        operations.reset();
    }
};

void test_shared_workflows()
{
    fixture                             f;
    std::weak_ptr<vector_space>         operations = f.operations;
    std::weak_ptr<file_operations_type> files      = f.files;
    std::weak_ptr<problem_type>         problem    = f.problem;
    std::weak_ptr<parameters_type>      parameters = f.parameters;
    std::weak_ptr<eigensolver>          adapter    = f.adapter;
    auto                                deflation =
        std::make_unique<deflation_workflow>( f.operations, f.files, &f.log, &f.log, f.problem, f.parameters );
    auto stability = std::make_unique<stability_workflow>(
        f.operations, f.files, &f.log, &f.log, f.problem, f.parameters, f.adapter
    );
    f.release();
    require( !operations.expired() && !files.expired() && !problem.expired(), "shared workflow resources retained" );
    require( !parameters.expired() && !adapter.expired(), "configuration and eigensolver retained" );
    deflation->set_parameters();
    stability->set_parameters();
    deflation.reset();
    require( !operations.expired() && !problem.expired(), "second workflow keeps common dependencies alive" );
    stability->set_parameters();
    stability.reset();
    require( operations.expired() && files.expired() && problem.expired(), "common dependencies released" );
    require( parameters.expired() && adapter.expired(), "no workflow ownership cycles" );
}

void test_custom_provider()
{
    fixture                               f;
    auto                                  provider = std::make_shared<linearization_provider>( f.problem );
    std::weak_ptr<linearization_provider> observed = provider;
    auto                                  workflow = std::make_unique<custom_stability_workflow>(
        f.operations, f.files, &f.log, &f.log, f.problem, f.parameters, f.adapter, provider
    );
    provider.reset();
    f.release();
    require( !observed.expired(), "custom linearization provider retained" );
    workflow->set_parameters();
    workflow.reset();
    require( observed.expired(), "custom linearization provider released" );
}

void test_workspace_failure()
{
    fixture   f;
    const int active = f.operations->allocations - f.operations->frees;
    const int before = f.operations->activations;
    {
        stability_workflow workflow( f.operations, f.files, &f.log, &f.log, f.problem, f.parameters, f.adapter );
    }
    const int activations = f.operations->activations - before;
    require( f.operations->allocations - f.operations->frees == active, "successful workflow workspace cleanup" );
    // The final two activations belong to the persistent traversal states.
    for ( int offset : { 1, 0 } )
    {
        f.operations->fail_activation = f.operations->activations + activations - offset;
        bool failed                   = false;
        try
        {
            stability_workflow workflow( f.operations, f.files, &f.log, &f.log, f.problem, f.parameters, f.adapter );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        f.operations->fail_activation = -1;
        require( failed, "workspace activation failure propagated" );
        require( f.operations->allocations - f.operations->frees == active, "failed workflow workspace cleanup" );
    }
}

void test_circle_stability_chain()
{
    auto                        operations          = std::make_shared<vector_space>();
    auto                        problem             = std::make_shared<problem_type>( 1.0, 1, operations );
    std::weak_ptr<vector_space> observed_operations = operations;
    operations.reset();
    require( !observed_operations.expired(), "circle retains its vector space independently of the eigensolver" );
    operations                                  = observed_operations.lock();
    auto                                linear  = std::make_shared<linear_operator_type>( problem );
    auto                                scaled  = std::make_shared<scaled_operator_type>( operations, linear, -1.0 );
    auto                                adapter = std::make_shared<scalar_eigensolver_type>( operations, scaled );
    std::weak_ptr<problem_type>         observed_problem = problem;
    std::weak_ptr<linear_operator_type> observed_linear  = linear;
    std::weak_ptr<scaled_operator_type> observed_scaled  = scaled;
    auto                               *space            = operations.get();
    auto                               *model            = problem.get();

    operations.reset();
    problem.reset();
    linear.reset();
    scaled.reset();
    require( !observed_operations.expired() && !observed_problem.expired(), "eigensolver retains circle resources" );
    require( !observed_linear.expired() && !observed_scaled.expired(), "eigensolver retains its operator chain" );
    {
        nmfd::detail::vector_wrap<vector_space, false, false> state( *space );
        nmfd::detail::vector_wrap<vector_space, false, false> residual( *space );
        state.init();
        state.start_use();
        residual.init();
        residual.start_use();
        space->assign_scalar( 0.6, *state );
        model->F( *state, 0.8, *residual );
        require( space->norm_l2( *residual ) < 1e-14, "retained circle residual remains usable" );
        model->set_linearization_point( *state, 0.8 );
        for ( int iteration = 0; iteration < 3; ++iteration )
        {
            const auto result = adapter->execute( *state );
            require(
                result.coverage_complete && result.eigenpairs.size() == 1 && result.eigenpairs.front().converged &&
                    std::abs( result.eigenpairs.front().value.real() + 1.2 ) < 1e-14,
                "retained eigensolver computes the analytical circle eigenvalue"
            );
        }
    }
    adapter.reset();
    require(
        observed_operations.expired() && observed_problem.expired() && observed_linear.expired() &&
            observed_scaled.expired(),
        "circle stability chain released without ownership cycles"
    );
}

void test_circle_construction_failure()
{
    // The problem owns one workspace; the scalar eigensolver owns three.
    for ( int activation = 1; activation <= 4; ++activation )
    {
        auto operations             = std::make_shared<vector_space>();
        operations->fail_activation = activation;
        bool failed                 = false;
        try
        {
            auto problem = std::make_shared<problem_type>( 1.0, 1, operations );
            auto linear  = std::make_shared<linear_operator_type>( problem );
            auto scaled  = std::make_shared<scaled_operator_type>( operations, linear, -1.0 );
            auto adapter = std::make_shared<scalar_eigensolver_type>( operations, scaled );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        require( failed, "circle stability construction failure propagated" );
        require( operations->allocations == operations->frees, "circle stability construction releases all storage" );
    }
}

void test_circle_invalid_construction()
{
    bool failed = false;
    try
    {
        problem_type problem( 1.0, 1, std::shared_ptr<vector_space>{} );
    }
    catch ( const std::invalid_argument & )
    {
        failed = true;
    }
    require( failed, "null circle vector operations rejected" );

    auto operations = std::make_shared<vector_space>();
    failed          = false;
    try
    {
        problem_type problem( 1.0, 2, operations );
    }
    catch ( const std::runtime_error & )
    {
        failed = true;
    }
    require( failed, "invalid circle dimension rejected" );
    require( operations->allocations == 0, "invalid circle construction does not allocate scratch" );
    require( operations.use_count() == 1, "invalid circle construction releases its shared dependency" );
}

} // namespace

int main()
{
    try
    {
        backend_type::init_device();
        test_shared_workflows();
        test_custom_provider();
        test_workspace_failure();
        test_circle_stability_chain();
        test_circle_construction_failure();
        test_circle_invalid_construction();
        require( !vector_space::invalid_cleanup, "all vector allocations released before their space" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << checks << " workflow ownership checks passed\n";
    return 0;
}
