#include <memory>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
#include <time_stepper/runge_kutta/explicit_time_step.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_integrator.h>

using operations_type    = nmfd::operations::scfd_vector_space<backend_type, double>;
using vector_type        = operations_type::vector_type;
using fixed_type         = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
using adaptive_type      = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
using adaptation_status  = nmfd::time_steppers::adaptation_status;
using integration_status = nmfd::time_steppers::integration_status;

void require( bool ok, const char *message )
{
    if ( !ok )
    {
        throw std::runtime_error( message );
    }
}

double exact( double t )
{
    return 2 * std::exp( t ) - t - 1;
}

double read( const operations_type &ops, const vector_type &x )
{
    double result;
    ops.get( x, &result );
    return result;
}

struct growth
{
    std::shared_ptr<operations_type> ops;
    double                           time  = 0;
    unsigned int                     calls = 0, fail_call = 0;

    void set_time( double t )
    {
        time = t;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        ++calls;
        ops->assign( in, out );
        ops->add_mul_scalar( time, 1., out );
        if ( calls == fail_call )
        {
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
        }
    }
};

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0;

    void start_use_vector( vector_type &v ) const override
    {
        if ( v.is_free() )
        {
            ++allocations;
            ++live;
        }
        operations_type::start_use_vector( v );
    }

    void free_vector( vector_type &v ) const override
    {
        if ( !v.is_free() )
        {
            --live;
        }
        operations_type::free_vector( v );
    }
};

template <class Step, class = void>
struct has_dense : std::false_type
{
};

template <class Step>
struct has_dense<Step, std::void_t<decltype( std::declval<Step &>().get_continuous_integration() )>> : std::true_type
{
};

using disabled_step = nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, growth, fixed_type>;
using dense_step    = nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, growth, fixed_type, true>;
static_assert( !has_dense<disabled_step>::value && has_dense<dense_step>::value );

void check_methods( std::shared_ptr<operations_type> ops_owner, vector_type &in, vector_type &out, vector_type &sample )
{
    auto &ops = *ops_owner;
    for ( const auto *method : { "EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK4", "RK64SSP", "RK45" } )
    {
        for ( const double direction : { 1., -1. } )
        {
            double       errors[3]{};
            unsigned int order = 0;
            for ( int level = 0; level < 3; ++level )
            {
                auto  counted_owner = std::make_shared<counting_operations>( 1 );
                auto &counted       = *counted_owner;
                auto  problem_owner = std::make_shared<growth>( growth{ counted_owner } );
                auto &problem       = *problem_owner;
                auto  fixed_owner =
                    std::make_shared<fixed_type>( typename fixed_type::params{ .2 / std::pow( 2., level ) } );
                auto &fixed = *fixed_owner;
                using step_type =
                    nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, growth, fixed_type, true>;
                {
                    auto step_owner = std::make_shared<step_type>(
                        counted_owner, problem_owner, fixed_owner, typename step_type::params{ method }
                    );
                    auto &step = *step_owner;
                    order      = step.dense_output_order();
                    if ( std::string( method ) == "RK4" )
                    {
                        require( order == 3, "RK4 uses third-order Hermite dense output" );
                    }
                    const auto expected = step.table().size() + 1 + step.table().is_embedded() +
                                          ( step.table().has_dense_output() ? 1 : 2 );
                    require( counted.live == expected, "Dense vector storage budget" );
                    const auto allocated = counted.allocations;
                    ops.assign_scalar( exact( .4 ), in );
                    step.set_time( .4 );
                    step.set_target_time( .4 + direction * .3 );
                    step.apply( in, out );
                    require(
                        step.get_status() == nmfd::time_steppers::single_step_status::converged, "Dense step failed"
                    );
                    require(
                        problem.calls == step.table().size() + ( step.table().has_dense_output() ? 0 : 1 ),
                        "Dense RHS budget"
                    );
                    const auto dense = step.get_continuous_integration();
                    const auto calls = problem.calls;
                    for ( const double theta : { 0., .13, .37, .73, 1. } )
                    {
                        const auto time = dense.evaluate( theta, sample );
                        require( time == .4 + theta * step.get_dt(), "Dense physical time" );
                        if ( theta == 0 )
                        {
                            require( read( ops, sample ) == read( ops, in ), "Dense initial endpoint" );
                        }
                        else if ( theta == 1 )
                        {
                            require( read( ops, sample ) == read( ops, out ), "Dense final endpoint" );
                        }
                        else
                        {
                            errors[level] = std::max( errors[level], std::abs( read( ops, sample ) - exact( time ) ) );
                        }
                    }
                    require(
                        problem.calls == calls && counted.allocations == allocated,
                        "Evaluation must not allocate or evaluate RHS"
                    );
                    step.finalize( adaptation_status::accepted, .4 + step.get_dt(), out );
                    bool refused = false;
                    try
                    {
                        dense.evaluate( .5, sample );
                    }
                    catch ( const std::logic_error & )
                    {
                        refused = true;
                    }
                    require( refused, "Finalization must invalidate dense output" );
                }
                require( counted.live == 0, "Dense storage release" );
            }
            require(
                errors[0] / errors[1] > std::pow( 2., order + .2 ) &&
                    errors[1] / errors[2] > std::pow( 2., order + .2 ),
                "Interior local convergence order"
            );
            std::cout << method << " dense order " << order << " direction " << direction << " PASS\n";
        }
    }
}

void check_lifecycle(
    std::shared_ptr<operations_type> ops_owner, vector_type &in, vector_type &out, vector_type &sample
)
{
    auto                 &ops           = *ops_owner;
    auto                  problem_owner = std::make_shared<growth>( growth{ ops_owner } );
    auto                 &problem       = *problem_owner;
    adaptive_type::params p;
    p.initial_step        = 1;
    p.relative_tolerance  = 1e-11;
    p.absolute_tolerance  = 1e-13;
    auto adaptation_owner = std::make_shared<adaptive_type>( ops_owner, p );
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, growth, adaptive_type, true>;
    auto  step_owner = std::make_shared<step_type>( ops_owner, problem_owner, adaptation_owner );
    auto &step       = *step_owner;
    require( step.get_status() == nmfd::time_steppers::single_step_status::idle, "New dense step is idle" );
    bool refused = false;
    try
    {
        step.get_continuous_integration();
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    require( refused, "No dense output before apply" );
    ops.assign_scalar( 1., in );
    step.apply( in, in );
    require( step.get_attempts() > 1, "Exercise rejection before dense output" );
    const auto old = step.get_continuous_integration();
    old.evaluate( 0, sample );
    require( read( ops, sample ) == 1, "Aliased apply preserves dense initial state" );
    old.evaluate( 1, sample );
    require( read( ops, sample ) == read( ops, in ), "Aliased apply preserves dense endpoint" );
    for ( const double theta :
          { -.01, 1.01, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
    {
        refused = false;
        ops.assign_scalar( 77., sample );
        try
        {
            old.evaluate( theta, sample );
        }
        catch ( const std::invalid_argument & )
        {
            refused = true;
        }
        require( refused && read( ops, sample ) == 77, "Invalid theta must not modify output" );
    }
    step.reset();
    require(
        step.get_status() == nmfd::time_steppers::single_step_status::idle && step.get_dt() == 0 &&
            step.get_attempts() == 0,
        "Reset clears the pending dense step"
    );
    ops.assign_scalar( 1., in );
    step.apply( in, out );
    refused = false;
    try
    {
        old.evaluate( .5, sample );
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    require( refused, "Old dense handle cannot observe a new step after reset" );
    const auto current = step.get_continuous_integration();
    step.finalize( adaptation_status::failed, 0, in );
    step.apply( in, out );
    refused = false;
    try
    {
        current.evaluate( .5, sample );
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    require( refused, "Old dense handle cannot observe a new pending step" );

    // SSP33's stages succeed but the extra endpoint derivative fails.
    auto  fixed_owner = std::make_shared<fixed_type>( typename fixed_type::params{ .1 } );
    auto &fixed       = *fixed_owner;
    auto  bad_owner   = std::make_shared<growth>( growth{ ops_owner, 0, 0, 4 } );
    auto  failed_owner =
        std::make_shared<dense_step>( ops_owner, bad_owner, fixed_owner, typename dense_step::params{ "RK33SSP" } );
    auto &failed = *failed_owner;
    ops.assign_scalar( 77., out );
    failed.apply( in, out );
    require(
        failed.get_status() == nmfd::time_steppers::single_step_status::failed_nonfinite && read( ops, out ) == 77,
        "Hermite preparation failure must not publish a candidate"
    );
    refused = false;
    try
    {
        failed.get_continuous_integration();
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    require( refused, "No dense output after failed preparation" );
}

struct section_manager
{
    std::shared_ptr<operations_type> ops;
    std::shared_ptr<dense_step>      step;
    double                           start  = 0;
    double                          *end    = nullptr;
    integration_status               status = integration_status::running;

    void set_time_interval( const double &t, double &next )
    {
        start = t;
        end   = &next;
    }

    bool apply( const vector_type &in, vector_type &out )
    {
        const auto threshold = exact( .35 );
        if ( read( *ops, in ) > threshold || read( *ops, out ) < threshold )
        {
            return false;
        }
        const auto dense = step->get_continuous_integration();
        double     left = 0, right = 1;
        for ( int i = 0; i < 45; ++i )
        {
            const auto theta = ( left + right ) / 2;
            dense.evaluate( theta, out );
            if ( read( *ops, out ) < threshold )
            {
                left = theta;
            }
            else
            {
                right = theta;
            }
        }
        *end   = dense.evaluate( ( left + right ) / 2, out );
        status = integration_status::stopped_by_external_operation;
        return true;
    }

    integration_status get_status() const
    {
        return status;
    }
};

int main()
try
{
    backend_type::init_device();
    auto                                       ops_owner = std::make_shared<operations_type>( 1 );
    auto                                      &ops       = *ops_owner;
    nmfd::detail::vector_wrap<operations_type> in( ops ), out( ops ), sample( ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    check_methods( ops_owner, *in, *out, *sample );
    check_lifecycle( ops_owner, *in, *out, *sample );
    auto  problem_owner  = std::make_shared<growth>( growth{ ops_owner } );
    auto &problem        = *problem_owner;
    auto  fixed_owner    = std::make_shared<fixed_type>( typename fixed_type::params{ .1 } );
    auto &fixed          = *fixed_owner;
    auto  step_owner     = std::make_shared<dense_step>( ops_owner, problem_owner, fixed_owner );
    auto &step           = *step_owner;
    auto  external_owner = std::make_shared<section_manager>( section_manager{ ops_owner, step_owner } );
    auto &external       = *external_owner;
    nmfd::time_steppers::integration::time_integrator<operations_type, dense_step, section_manager> integrator(
        ops_owner, step_owner, {}, external_owner
    );
    ops.assign_scalar( 1., *in );
    integrator.set_time_interval( 0, 1 );
    integrator.apply( *in, *out );
    require(
        integrator.get_status() == integration_status::stopped_by_external_operation &&
            std::abs( integrator.get_final_time() - .35 ) < 1e-8 &&
            std::abs( read( ops, *out ) - exact( .35 ) ) < 1e-12,
        "External section refinement must commit matching time and state"
    );
    std::cout << "Dense output lifecycle, memory, rejection, and external section tests PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
