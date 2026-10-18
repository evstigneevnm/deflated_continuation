#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <time_stepper/integration/time_integrator.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>

void require( bool passed, const char *message )
{
    if ( !passed )
        throw std::runtime_error( message );
}

struct operations_type : nmfd::operations::scfd_vector_space<backend_type, double>
{
    using base_type = nmfd::operations::scfd_vector_space<backend_type, double>;
    using base_type::base_type;
    mutable unsigned int allocations = 0, live = 0;
    void                 start_use_vector( vector_type &x ) const override
    {
        if ( x.is_free() )
        {
            ++allocations;
            ++live;
        }
        base_type::start_use_vector( x );
    }
    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
            --live;
        base_type::free_vector( x );
    }
};

using vector_type = operations_type::vector_type;
struct problem_type
{
    std::shared_ptr<operations_type> operations;
    void                             apply( const vector_type &, vector_type &out )
    {
        operations->assign_scalar( 1., out );
    }
};
using adaptation_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
using step_type =
    nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, problem_type, adaptation_type, true>;

struct external_manager
{
    std::shared_ptr<step_type> step;
    unsigned int               calls = 0;
    void                       set_time_interval( const double &, double & )
    {
    }
    bool apply( const vector_type &, vector_type & )
    {
        require(
            step->get_status() == nmfd::time_steppers::single_step_status::converged,
            "external manager sees the accepted trial"
        );
        ++calls;
        return false;
    }
    nmfd::time_steppers::integration_status get_status() const
    {
        return nmfd::time_steppers::integration_status::running;
    }
};
using integrator_type = nmfd::time_steppers::integration::time_integrator<operations_type, step_type, external_manager>;
using matlab_type     = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
using scipy_type      = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;

static_assert(
    std::is_constructible_v<
        step_type, std::shared_ptr<operations_type>, std::shared_ptr<problem_type>, std::shared_ptr<adaptation_type>>
);
static_assert( !std::is_constructible_v<step_type, operations_type &, problem_type &, adaptation_type &> );
static_assert( std::is_constructible_v<
               integrator_type, std::shared_ptr<operations_type>, std::shared_ptr<step_type>, integrator_type::params,
               std::shared_ptr<external_manager>> );
static_assert( !std::is_constructible_v<
               integrator_type, operations_type &, step_type &, integrator_type::params, external_manager *> );
static_assert( std::is_constructible_v<matlab_type, std::shared_ptr<operations_type>> );
static_assert( std::is_constructible_v<scipy_type, std::shared_ptr<operations_type>> );
static_assert( !std::is_constructible_v<matlab_type, operations_type &> );
static_assert( !std::is_constructible_v<scipy_type, operations_type &> );

void check_null_dependencies()
{
    auto       operations = std::make_shared<operations_type>( 1 );
    auto       problem    = std::make_shared<problem_type>( problem_type{ operations } );
    auto       adaptation = std::make_shared<adaptation_type>();
    const auto before     = operations->allocations;
    for ( unsigned int missing = 0; missing < 3; ++missing )
    {
        bool rejected = false;
        try
        {
            step_type invalid(
                missing == 0 ? nullptr : operations, missing == 1 ? nullptr : problem,
                missing == 2 ? nullptr : adaptation
            );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected && operations->allocations == before, "RK null dependency fails before allocation" );
    }
    auto       step      = std::make_shared<step_type>( operations, problem, adaptation );
    const auto allocated = operations->allocations;
    for ( unsigned int missing = 0; missing < 2; ++missing )
    {
        bool rejected = false;
        try
        {
            integrator_type invalid( missing == 0 ? nullptr : operations, missing == 1 ? nullptr : step );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require(
            rejected && operations->allocations == allocated, "integration null dependency fails before allocation"
        );
    }
    for ( bool scipy : { false, true } )
    {
        bool rejected = false;
        try
        {
            if ( scipy )
                nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type> invalid( nullptr );
            else
                nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type> invalid( nullptr );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "adaptation requires an owned vector space" );
    }
}

void check_retained_graph()
{
    auto operations = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<problem_type>( problem_type{ operations } );
    auto adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ .1 } );
    auto step = std::make_shared<step_type>( operations, problem, adaptation, step_type::params{ "RK45", 32, true } );
    auto external                                    = std::make_shared<external_manager>( external_manager{ step } );
    std::weak_ptr<operations_type>  space_lifetime   = operations;
    std::weak_ptr<problem_type>     problem_lifetime = problem;
    std::weak_ptr<adaptation_type>  adaptation_lifetime = adaptation;
    std::weak_ptr<step_type>        step_lifetime       = step;
    std::weak_ptr<external_manager> external_lifetime   = external;
    auto integrator = std::make_unique<integrator_type>( operations, step, integrator_type::params{}, external );
    integrator->set_time_interval( 0, .25 );
    operations.reset();
    problem.reset();
    adaptation.reset();
    step.reset();
    external.reset();
    require(
        !space_lifetime.expired() && !problem_lifetime.expired() && !adaptation_lifetime.expired() &&
            !step_lifetime.expired() && !external_lifetime.expired(),
        "integrator retains the complete dependency graph"
    );
    {
        auto                                       space = space_lifetime.lock();
        nmfd::detail::vector_wrap<operations_type> initial( *space ), result( *space );
        initial.start_use();
        result.start_use();
        space->assign_scalar( 0., *initial );
        const auto allocated = space->allocations;
        for ( unsigned int run = 0; run < 2; ++run )
        {
            integrator->apply( *initial, *result );
            require(
                integrator->get_status() == nmfd::time_steppers::integration_status::completed &&
                    std::abs( space->get_value_at_point( 0, *result ) - .25 ) < 1e-14,
                "retained dependencies support integration and restart"
            );
        }
        require(
            space->allocations == allocated && external_lifetime.lock()->calls == 6,
            "integration and callbacks allocate no additional vectors"
        );
    }
    auto retained_external = external_lifetime.lock();
    integrator.reset();
    require(
        !space_lifetime.expired() && !problem_lifetime.expired() && !adaptation_lifetime.expired() &&
            !step_lifetime.expired(),
        "a retained external manager keeps its step and dependencies alive"
    );
    {
        auto                                       space = space_lifetime.lock();
        nmfd::detail::vector_wrap<operations_type> initial( *space ), result( *space ), sample( *space );
        initial.start_use();
        result.start_use();
        sample.start_use();
        space->assign_scalar( .25, *initial );
        const auto allocated = space->allocations;
        retained_external->step->set_time( .25 );
        retained_external->step->set_target_time( .35 );
        retained_external->step->apply( *initial, *result );
        retained_external->apply( *initial, *result );
        const auto dense = retained_external->step->get_continuous_integration();
        dense.evaluate( .5, *sample );
        require(
            std::abs( space->get_value_at_point( 0, *sample ) - .3 ) < 1e-14 && space->allocations == allocated,
            "a retained manager can resume stepping and interpolation without reallocating"
        );
        retained_external->step->finalize( nmfd::time_steppers::adaptation_status::accepted, .35, *result );
    }
    retained_external.reset();
    require(
        space_lifetime.expired() && problem_lifetime.expired() && adaptation_lifetime.expired() &&
            step_lifetime.expired() && external_lifetime.expired(),
        "all dependencies are released without cycles"
    );
}

template <class Adaptation>
void check_adaptation_lifetime()
{
    auto                           operations = std::make_shared<operations_type>( 1 );
    std::weak_ptr<operations_type> lifetime   = operations;
    {
        Adaptation adaptation( operations );
        require( operations->allocations == 0, "adaptation ownership adds no trajectory vectors" );
        operations.reset();
        require( !lifetime.expired(), "standalone adaptation retains its vector operations" );
    }
    require( lifetime.expired(), "standalone adaptation releases its vector operations" );
}

int main()
try
{
    backend_type::init_device();
    check_null_dependencies();
    check_retained_graph();
    check_adaptation_lifetime<nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>>();
    check_adaptation_lifetime<nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>>();
    std::cout << "Shared timestepper dependencies, null guards, lifetime, restart and allocation tests: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
