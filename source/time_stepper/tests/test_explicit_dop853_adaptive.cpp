#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
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

using operations_type    = nmfd::operations::scfd_vector_space<backend_type, double>;
using vector_type        = operations_type::vector_type;
using adaptation_type    = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;
using adaptation_status  = nmfd::time_steppers::adaptation_status;
using step_status        = nmfd::time_steppers::single_step_status;
using integration_status = nmfd::time_steppers::integration_status;

void require( bool condition, const char *message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

double read( const operations_type &operations, const vector_type &state )
{
    const auto view = operations.view( state );
    return view( 0 );
}

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0;

    void start_use_vector( vector_type &state ) const override
    {
        if ( state.is_free() )
        {
            ++allocations;
            ++live;
        }
        operations_type::start_use_vector( state );
    }

    void free_vector( vector_type &state ) const override
    {
        if ( !state.is_free() )
        {
            --live;
        }
        operations_type::free_vector( state );
    }
};

struct forced_growth
{
    std::shared_ptr<operations_type> operations;
    unsigned int                     calls = 0, fail_call = 0;
    double                           time = 0;

    static double exact( double time )
    {
        return 2 * std::exp( time ) - time - 1;
    }

    void set_time( double value )
    {
        time = value;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        ++calls;
        operations->assign( in, out );
        operations->add_mul_scalar( time, 1., out );
        if ( calls == fail_call )
        {
            operations->assign_scalar( std::numeric_limits<double>::infinity(), out );
        }
    }
};

struct quadratic_growth
{
    std::shared_ptr<operations_type> operations;

    static double exact( double time )
    {
        return 1 / ( 1 - time );
    }

    void apply( const vector_type &in, vector_type &out )
    {
        operations->mul_pointwise( 1., in, 1., in, out );
    }
};

void check_estimator( std::shared_ptr<operations_type> operations_owner, vector_type &initial, vector_type &result )
{
    auto &operations = *operations_owner;
    // Independent SciPy rk_step at h=+/-0.5, y(0)=1, f(t,y)=y+t.
    const double candidates[] = { 1.7974425410584143, .71306131960354979 };
    const double errors5[]    = { -4.1973993792493447e-7, -4.2002260698063149e-7 };
    const double errors3[]    = { 6.6083434716700945e-4, 4.1883651205208185e-4 };
    for ( int direction = 0; direction < 2; ++direction )
    {
        adaptation_type::params p;
        p.initial_step         = .5;
        p.maximum_step         = 2;
        p.relative_tolerance   = 1e-5;
        p.absolute_tolerance   = 1e-7;
        auto  adaptation_owner = std::make_shared<adaptation_type>( operations_owner, p );
        auto &adaptation       = *adaptation_owner;
        auto  problem_owner    = std::make_shared<forced_growth>( forced_growth{ operations_owner } );
        auto &problem          = *problem_owner;
        using step_type =
            nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptation_type>;
        auto step_owner = std::make_shared<step_type>(
            operations_owner, problem_owner, adaptation_owner, typename step_type::params{ "DOP853" }
        );
        auto &step = *step_owner;
        step.set_target_time( direction == 0 ? .5 : -.5 );
        step.apply( initial, result );
        require(
            step.get_status() == step_status::converged && step.get_attempts() == 1 && problem.calls == 13,
            "Combined estimator uses the ordinary derivatives without extra RHS calls"
        );
        require(
            std::abs( read( operations, result ) - candidates[direction] ) < 2e-14 &&
                std::abs( read( operations, step.error_estimate( 0 ) ) - errors5[direction] ) < 2e-14 &&
                std::abs( read( operations, step.error_estimate( 1 ) ) - errors3[direction] ) < 2e-14,
            "DOP853 E5/E3 state defects must agree with SciPy, including the signed step factor"
        );
        const auto scale =
            p.absolute_tolerance + p.relative_tolerance * std::max( 1., std::abs( candidates[direction] ) );
        const auto rms5 = std::abs( errors5[direction] ) / scale, rms3 = std::abs( errors3[direction] ) / scale;
        const auto ratio    = rms5 * ( rms5 / std::hypot( rms5, .1 * rms3 ) );
        const auto expected = .5 * .9 * std::pow( ratio, -1. / 8 );
        require(
            std::abs( adaptation.get_dt() - expected ) < 2e-8,
            "DOP853 adaptation must use the combined error and exponent eight, not E5 alone"
        );
        bool refused = false;
        try
        {
            step.error_estimate();
        }
        catch ( const std::logic_error & )
        {
            refused = true;
        }
        require( refused, "DOP853 must not expose E5 as a standalone combined error vector" );
    }
}

template <bool DenseOutput>
void check_storage_and_retry(
    std::shared_ptr<operations_type> operations_owner, vector_type &initial, vector_type &result
)
{
    auto                   &operations    = *operations_owner;
    auto                    counted_owner = std::make_shared<counting_operations>( 1 );
    auto                   &counted       = *counted_owner;
    auto                    problem_owner = std::make_shared<forced_growth>( forced_growth{ counted_owner } );
    auto                   &problem       = *problem_owner;
    adaptation_type::params p;
    p.initial_step         = 1;
    p.relative_tolerance   = 1e-10;
    p.absolute_tolerance   = 1e-12;
    auto  adaptation_owner = std::make_shared<adaptation_type>( counted_owner, p );
    auto &adaptation       = *adaptation_owner;
    using step_type        = nmfd::time_steppers::runge_kutta::explicit_time_step<
               counting_operations, forced_growth, adaptation_type, DenseOutput>;
    {
        auto step_owner = std::make_shared<step_type>(
            counted_owner, problem_owner, adaptation_owner, typename step_type::params{ "DOP853" }
        );
        auto &step = *step_owner;
        require(
            counted.live == ( DenseOutput ? 20u : 16u ),
            "Adaptive DOP853 owns two error vectors independently of optional dense storage"
        );
        const auto allocated = counted.allocations;
        step.apply( initial, result );
        require(
            step.get_status() == step_status::converged && step.get_attempts() > 1 &&
                problem.calls == 13 * step.get_attempts() + ( DenseOutput ? 3u : 0u ),
            "Rejected trials must not evaluate dense-only stages"
        );
        require(
            std::abs( read( operations, result ) - forced_growth::exact( step.get_dt() ) ) < 2e-8,
            "Adaptive DOP853 candidate accuracy"
        );
        if constexpr ( DenseOutput )
        {
            const auto time = step.get_continuous_integration().evaluate( .37, result );
            require(
                std::abs( read( operations, result ) - forced_growth::exact( time ) ) < 2e-8,
                "Native dense output follows accepted adaptive stages"
            );
            step.finalize( adaptation_status::accepted_modified, time, result );
        }
        else
        {
            step.finalize( adaptation_status::accepted, step.get_dt(), result );
        }
        require( counted.allocations == allocated, "Adaptive attempts and interpolation must reuse allocations" );
    }
    require( counted.live == 0, "Adaptive error buffers must release their storage" );
    using fixed_type  = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
    auto  fixed_owner = std::make_shared<fixed_type>( typename fixed_type::params{ .1 } );
    auto &fixed       = *fixed_owner;
    {
        nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, fixed_type> step(
            counted_owner, problem_owner, fixed_owner, { "DOP853" }
        );
        require( counted.live == 14, "Supporting adaptive DOP853 must not add error storage to fixed DOP853" );
    }
    {
        nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, forced_growth, adaptation_type> step(
            counted_owner, problem_owner, adaptation_owner, { "RK45" }
        );
        require( counted.live == 9, "Ordinary embedded RK must retain its single error vector" );
    }
    require( counted.live == 0, "Other configurations must release their storage" );
}

template <class Problem>
void check_integration( std::shared_ptr<operations_type> operations_owner, double end_time )
{
    auto                                      &operations = *operations_owner;
    nmfd::detail::vector_wrap<operations_type> initial( operations ), result( operations );
    initial.start_use();
    result.start_use();
    operations.assign_scalar( 1., *initial );
    double previous_error = 1;
    for ( const double tolerance : { 1e-5, 1e-8, 1e-11 } )
    {
        auto                    problem_owner = std::make_shared<Problem>( Problem{ operations_owner } );
        auto                   &problem       = *problem_owner;
        adaptation_type::params p;
        p.initial_step         = 1;
        p.relative_tolerance   = tolerance;
        p.absolute_tolerance   = tolerance / 100;
        auto  adaptation_owner = std::make_shared<adaptation_type>( operations_owner, p );
        auto &adaptation       = *adaptation_owner;
        using step_type =
            nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, Problem, adaptation_type>;
        auto step_owner = std::make_shared<step_type>(
            operations_owner, problem_owner, adaptation_owner, typename step_type::params{ "DOP853" }
        );
        auto                                                                         &step = *step_owner;
        nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator(
            operations_owner, step_owner
        );
        integrator.set_time_interval( 0, end_time );
        for ( int run = 0; run < 2; ++run )
        {
            integrator.apply( *initial, *result );
            const auto error = std::abs( read( operations, *result ) - Problem::exact( end_time ) );
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == end_time &&
                    error < 10 * tolerance * Problem::exact( end_time ),
                "Adaptive DOP853 analytical accuracy and fresh-trajectory restart"
            );
            if ( run == 0 )
            {
                require( error < previous_error / 2, "Tighter DOP853 tolerances must improve analytical accuracy" );
                previous_error = error;
                std::cout << "Adaptive DOP853 tolerance " << tolerance << ": error " << error << ", steps "
                          << integrator.get_steps() << '\n';
            }
        }
        operations.assign_scalar( Problem::exact( end_time ), *result );
        integrator.set_time_interval( end_time, 0 );
        integrator.apply( *result, *result );
        require(
            integrator.get_status() == integration_status::completed && integrator.get_final_time() == 0 &&
                std::abs( read( operations, *result ) - 1 ) < 100 * tolerance,
            "Adaptive DOP853 backward integration supports aliased states"
        );
    }
}

void check_failures( std::shared_ptr<operations_type> operations_owner, vector_type &initial, vector_type &result )
{
    auto                   &operations    = *operations_owner;
    auto                    problem_owner = std::make_shared<forced_growth>( forced_growth{ operations_owner } );
    auto                   &problem       = *problem_owner;
    adaptation_type::params p;
    p.initial_step         = 1;
    p.relative_tolerance   = 1e-10;
    p.absolute_tolerance   = 1e-12;
    auto  adaptation_owner = std::make_shared<adaptation_type>( operations_owner, p );
    auto &adaptation       = *adaptation_owner;
    using step_type =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptation_type>;
    auto budgeted_owner = std::make_shared<step_type>(
        operations_owner, problem_owner, adaptation_owner, typename step_type::params{ "DOP853", 1 }
    );
    auto &budgeted = *budgeted_owner;
    operations.assign_scalar( 77., result );
    budgeted.apply( initial, result );
    require(
        budgeted.get_status() == step_status::attempt_limit_reached && read( operations, result ) == 77 &&
            read( operations, initial ) == 1,
        "Rejected adaptive candidates must respect attempt limits and preserve caller state"
    );
    p.minimum_step       = .5;
    p.relative_tolerance = 1e-14;
    p.absolute_tolerance = 1e-16;
    auto limited_owner   = std::make_shared<adaptation_type>( operations_owner, p );
    auto failed_owner    = std::make_shared<step_type>(
        operations_owner, problem_owner, limited_owner, typename step_type::params{ "DOP853" }
    );
    auto &failed = *failed_owner;
    failed.apply( initial, result );
    require(
        failed.get_status() == step_status::failed_minimum_dt && read( operations, result ) == 77,
        "Minimum-step failure must not publish an inaccurate candidate"
    );

    using matlab_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
    auto matlab_owner = std::make_shared<matlab_type>( operations_owner );
    nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, matlab_type> unsupported(
        operations_owner, problem_owner, matlab_owner, { "DOP853" }
    );
    const auto calls = problem.calls;
    unsupported.apply( initial, result );
    require(
        unsupported.get_status() == step_status::error_estimate_unavailable && problem.calls == calls &&
            read( operations, result ) == 77,
        "Controllers without combined assessment must fail before evaluating DOP853"
    );

    p                     = {};
    p.initial_step        = .5;
    p.relative_tolerance  = .01;
    p.absolute_tolerance  = .001;
    auto  recovery_owner  = std::make_shared<adaptation_type>( operations_owner, p );
    auto  transient_owner = std::make_shared<forced_growth>( forced_growth{ operations_owner, 0, 14 } );
    auto &transient       = *transient_owner;
    nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, forced_growth, adaptation_type, true>
        recovered( operations_owner, transient_owner, recovery_owner, { "DOP853" } );
    recovered.apply( initial, result );
    require(
        recovered.get_status() == step_status::converged && recovered.get_attempts() == 2 && transient.calls == 30 &&
            recovered.get_dt() == .1 && std::abs( read( operations, result ) - forced_growth::exact( .1 ) ) < 1e-10,
        "Adaptive dense-stage failure must retry from the original state with the attempted step reduced"
    );
}

int main()
try
{
    backend_type::init_device();
    auto                                       operations_owner = std::make_shared<operations_type>( 1 );
    auto                                      &operations       = *operations_owner;
    nmfd::detail::vector_wrap<operations_type> initial( operations ), result( operations );
    initial.start_use();
    result.start_use();
    operations.assign_scalar( 1., *initial );
    check_estimator( operations_owner, *initial, *result );
    check_storage_and_retry<false>( operations_owner, *initial, *result );
    check_storage_and_retry<true>( operations_owner, *initial, *result );
    check_integration<forced_growth>( operations_owner, 2 );
    check_integration<quadratic_growth>( operations_owner, .5 );
    check_failures( operations_owner, *initial, *result );
    std::cout << "Adaptive DOP853 estimator, optional storage, tolerance, restart and failure tests: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
