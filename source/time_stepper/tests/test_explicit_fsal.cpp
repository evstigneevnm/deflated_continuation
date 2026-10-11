#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu;
#endif
#include <common/scfd_vector_operations.h>
#include <time_stepper/runge_kutta/explicit_time_step.h>
#include <time_stepper/integration/time_step_adaptation_constant.h>
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/integration/time_integrator.h>

using operations_type    = scfd_vector_operations<backend_type, double>;
using vector_type        = operations_type::vector_type;
using step_status        = nmfd::time_steppers::single_step_status;
using adaptation_status  = nmfd::time_steppers::adaptation_status;
using integration_status = nmfd::time_steppers::integration_status;

void require( bool condition, const char *message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

struct counting_operations : operations_type
{
    using operations_type::assign;
    using operations_type::operations_type;
    mutable std::size_t allocations = 0, live = 0, copies = 0;

    void start_use_vector( vector_type &x ) const override
    {
        if ( x.is_free() )
        {
            ++allocations;
            ++live;
        }
        operations_type::start_use_vector( x );
    }

    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
        {
            --live;
        }
        operations_type::free_vector( x );
    }

    void assign( const vector_type &in, vector_type &out ) const
    {
        ++copies;
        operations_type::assign( in, out );
    }
};

// Autonomous u'=u or non-autonomous u'=u+t, with one injected nonfinite evaluation.
struct growth
{
    counting_operations &operations;
    bool                 forced = false, quadratic = false;
    double               time = 0, rate = 1;
    std::size_t          calls = 0, initial_calls = 0, fail_call = 0;

    void set_time( double t )
    {
        time = t;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        ++calls;
        initial_calls += time == 0;
        if ( calls == fail_call )
        {
            operations.assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
            return;
        }
        if ( quadratic )
        {
            operations.mul_pointwise( rate, in, 1., in, out );
        }
        else
        {
            operations.assign_mul( rate, in, out );
        }
        if ( forced )
        {
            operations.add_mul_scalar( time, 1., out );
        }
    }
};

using fixed_type    = nmfd::time_steppers::integration::time_step_adaptation_constant<counting_operations>;
using adaptive_type = nmfd::time_steppers::integration::time_step_adaptation_scipy<counting_operations>;
template <class Adaptation, bool DenseOutput = false>
using step_type =
    nmfd::time_steppers::runge_kutta::explicit_time_step<counting_operations, growth, Adaptation, DenseOutput>;

double read( counting_operations &operations, const vector_type &x )
{
    double value = 0;
    operations.get( x, &value );
    return value;
}

template <bool DenseOutput>
void check_trajectory(
    counting_operations &operations, vector_type &x, vector_type &reference, vector_type &sample,
    vector_type &reference_sample, const std::string &method, bool forced, bool quadratic, double direction
)
{
    operations.assign_scalar( 1., x );
    operations.assign_scalar( 1., reference );
    growth     problem{ operations, forced, quadratic }, reference_problem{ operations, forced, quadratic };
    fixed_type adaptation( { .05 } ), reference_adaptation( { .05 } );
    using method_type  = step_type<fixed_type, DenseOutput>;
    const auto  before = operations.allocations;
    method_type baseline( operations, reference_problem, reference_adaptation, { method } );
    const auto  baseline_storage = operations.allocations - before;
    method_type reused( operations, problem, adaptation, { method, 32, true } );
    require( operations.allocations - before == 2 * baseline_storage, "FSAL adds no trajectory vectors" );
    const auto allocated = operations.allocations;
    double     time      = 0;
    for ( unsigned int i = 0; i < 5; ++i )
    {
        reused.set_time( time );
        baseline.set_time( time );
        reused.set_target_time( direction );
        baseline.set_target_time( direction );
        reused.apply( x, x );
        baseline.apply( reference, reference );
        require(
            reused.get_status() == step_status::converged && baseline.get_status() == step_status::converged,
            "FSAL trajectory must converge"
        );
        require( read( operations, x ) == read( operations, reference ), "FSAL endpoint matches recomputation" );
        if ( reused.table().is_embedded() )
        {
            require(
                read( operations, reused.error_estimate() ) == read( operations, baseline.error_estimate() ),
                "FSAL embedded error matches recomputation"
            );
        }
        if constexpr ( DenseOutput )
        {
            const auto dense           = reused.get_continuous_integration();
            const auto reference_dense = baseline.get_continuous_integration();
            for ( const double theta : { 0., .3, .75, 1. } )
            {
                require(
                    dense.evaluate( theta, sample ) == reference_dense.evaluate( theta, reference_sample ),
                    "FSAL dense times match"
                );
                require(
                    read( operations, sample ) == read( operations, reference_sample ),
                    "FSAL stage mapping preserves dense output"
                );
            }
        }
        time += reused.get_dt();
        const auto copies = operations.copies;
        reused.finalize( adaptation_status::accepted, time, x );
        require( operations.copies == copies, "FSAL finalization must not copy vectors" );
        baseline.finalize( adaptation_status::accepted, time, reference );
        if constexpr ( DenseOutput )
        {
            bool refused = false;
            try
            {
                reused.get_continuous_integration();
            }
            catch ( const std::logic_error & )
            {
                refused = true;
            }
            require( refused, "FSAL finalization expires dense output" );
        }
    }
    const auto stages = reused.table().size();
    const auto extras =
        DenseOutput ? ( reused.table().has_dense_output() ? reused.table().dense_outout_stage_count() - stages : 1 )
                    : 0;
    const auto savings = reused.table().has_fsal() ? 4 : 0;
    require(
        reference_problem.calls == 5 * ( stages + extras ) && problem.calls == reference_problem.calls - savings,
        "FSAL saves one RHS evaluation per continued step"
    );
    require( operations.allocations == allocated, "FSAL must not allocate during stepping or interpolation" );
}

void check_invalidation(
    counting_operations &operations, vector_type &x, vector_type &reference, vector_type &out,
    vector_type &reference_out, const std::string &method
)
{
    using method_type = step_type<fixed_type, true>;
    growth      problem{ operations, true }, reference_problem{ operations, true };
    fixed_type  adaptation( { .05 } ), reference_adaptation( { .05 } );
    method_type reused( operations, problem, adaptation, { method, 32, true } );
    method_type baseline( operations, reference_problem, reference_adaptation, { method } );
    operations.assign_scalar( 1., x );
    operations.assign_scalar( 1., reference );
    double time   = 0;
    bool   cached = false;
    for ( unsigned int action = 0; action < 11; ++action )
    {
        if ( action == 2 || action == 9 )
        {
            problem.rate = reference_problem.rate = 1.2;
            if ( action == 9 )
            {
                operations.assign_scalar( 2., x );
                operations.assign_scalar( 2., reference );
            }
            reused.invalidate_rhs_cache();
            cached = false;
        }
        if ( action == 7 || action == 10 )
        {
            time += .0123;
            cached = false;
        }
        if ( action == 8 )
        {
            reused.reset();
            baseline.reset();
            time = 0;
            operations.assign_scalar( 1., x );
            operations.assign_scalar( 1., reference );
            require( reused.get_status() == step_status::idle, "FSAL reset reports idle" );
            cached = false;
        }
        reused.set_time( time );
        baseline.set_time( time );
        const auto calls = problem.calls;
        reused.apply( x, out );
        baseline.apply( reference, reference_out );
        require(
            reused.get_status() == step_status::converged && baseline.get_status() == step_status::converged,
            "FSAL invalidation replay must converge"
        );
        require(
            problem.calls - calls == reused.table().dense_outout_stage_count() - ( cached ? 1 : 0 ),
            "Invalidation must recompute the first derivative"
        );
        require(
            read( operations, out ) == read( operations, reference_out ), "Invalidation replay matches an uncached step"
        );
        auto outcome        = adaptation_status::accepted;
        auto committed_time = time + reused.get_dt();
        cached              = true;
        if ( action == 3 )
        {
            // Invalidation while pending must also block promotion during finalize().
            problem.rate = reference_problem.rate = .8;
            reused.invalidate_rhs_cache();
            cached = false;
        }
        if ( action == 4 )
        {
            operations.assign_scalar( 1.5, out );
            operations.assign_scalar( 1.5, reference_out );
            outcome = adaptation_status::accepted_modified;
            cached  = false;
        }
        if ( action == 5 )
        {
            committed_time = reused.get_continuous_integration().evaluate( .5, out );
            baseline.get_continuous_integration().evaluate( .5, reference_out );
            cached = false;
        }
        if ( action == 6 )
        {
            outcome        = adaptation_status::failed;
            committed_time = time;
            cached         = false;
        }
        reused.finalize( outcome, committed_time, outcome == adaptation_status::failed ? x : out );
        baseline.finalize( outcome, committed_time, outcome == adaptation_status::failed ? reference : reference_out );
        if ( outcome != adaptation_status::failed )
        {
            operations.assign( out, x );
            operations.assign( reference_out, reference );
        }
        time = committed_time;
    }
}

template <bool DenseOutput>
void check_retries(
    counting_operations &operations, vector_type &x, vector_type &reference, vector_type &out,
    vector_type &reference_out, const std::string &method
)
{
    using method_type = step_type<adaptive_type, DenseOutput>;
    growth                problem{ operations, true }, reference_problem{ operations, true };
    adaptive_type::params p;
    p.initial_step       = 1;
    p.relative_tolerance = 1e-12;
    p.absolute_tolerance = 1e-14;
    adaptive_type adaptation( operations, p ), reference_adaptation( operations, p );
    method_type   reused( operations, problem, adaptation, { method, 32, true } );
    method_type   baseline( operations, reference_problem, reference_adaptation, { method } );
    operations.assign_scalar( 1., x );
    operations.assign_scalar( 1., reference );
    double time = 0;
    for ( unsigned int i = 0; i < 2; ++i )
    {
        reused.set_time( time );
        baseline.set_time( time );
        const auto calls = problem.calls;
        reused.apply( x, out );
        baseline.apply( reference, reference_out );
        require(
            reused.get_status() == step_status::converged && baseline.get_status() == step_status::converged &&
                reused.get_dt() == baseline.get_dt() && reused.get_attempts() == baseline.get_attempts(),
            "FSAL preserves adaptive assessment and retries"
        );
        require( read( operations, out ) == read( operations, reference_out ), "FSAL retry endpoint matches" );
        const auto stages = reused.table().size();
        const auto extras = DenseOutput ? reused.table().dense_outout_stage_count() - stages : 0;
        require(
            problem.calls - calls == ( i == 0 ? 1 : 0 ) + reused.get_attempts() * ( stages - 1 ) + extras,
            "Retries retain the original first derivative, not a rejected endpoint"
        );
        require( problem.initial_calls == 1, "Rejected trials must not reevaluate the finite starting derivative" );
        if ( i == 0 )
        {
            require( reused.get_attempts() > 1, "Exercise adaptive rejection with FSAL" );
        }
        time += reused.get_dt();
        reused.finalize( adaptation_status::accepted, time, out );
        baseline.finalize( adaptation_status::accepted, time, reference_out );
        const auto next_dt = adaptation.get_dt();
        if ( i == 1 )
        {
            reused.invalidate_rhs_cache();
            require( adaptation.get_dt() == next_dt, "Explicit cache invalidation preserves adaptation history" );
        }
        operations.assign( out, x );
        operations.assign( reference_out, reference );
    }
}

void check_failures( counting_operations &operations, vector_type &in, vector_type &out )
{
    operations.assign_scalar( 1., in );
    for ( const auto *method : { "RK23", "RK45", "DOP853" } )
    {
        for ( const std::size_t fail_call : { 1u, 2u } )
        {
            growth problem{ operations, true };
            problem.fail_call = fail_call;
            adaptive_type::params p;
            p.initial_step       = .5;
            p.relative_tolerance = .01;
            p.absolute_tolerance = .001;
            adaptive_type            adaptation( operations, p );
            step_type<adaptive_type> recovered( operations, problem, adaptation, { method, 32, true } );
            recovered.apply( in, out );
            require(
                recovered.get_status() == step_status::converged && recovered.get_attempts() == 2 &&
                    recovered.get_dt() == .1 && problem.calls == recovered.table().size() + 1,
                "Nonfinite stages must retry from the original state"
            );
            require(
                problem.initial_calls == ( fail_call == 1 ? 2u : 1u ),
                "A nonfinite first derivative must never be reused"
            );
        }
        growth problem{ operations, true };
        problem.fail_call = 2;
        fixed_type            fixed( { .1 } );
        step_type<fixed_type> failed( operations, problem, fixed, { method, 32, true } );
        operations.assign_scalar( 77., out );
        failed.apply( in, out );
        require(
            failed.get_status() == step_status::failed_nonfinite && read( operations, out ) == 77,
            "Failed FSAL attempt preserves caller output"
        );
        const auto calls = problem.calls;
        failed.apply( in, out );
        require(
            failed.get_status() == step_status::converged && problem.calls - calls == failed.table().size(),
            "A terminal failure must discard the derivative cache"
        );
    }
    growth problem{ operations, true };
    problem.fail_call = 14;
    adaptive_type::params p;
    p.initial_step       = .5;
    p.relative_tolerance = .01;
    p.absolute_tolerance = .001;
    adaptive_type                  adaptation( operations, p );
    step_type<adaptive_type, true> dense_retry( operations, problem, adaptation, { "DOP853", 32, true } );
    dense_retry.apply( in, out );
    require(
        dense_retry.get_status() == step_status::converged && dense_retry.get_attempts() == 2 &&
            dense_retry.get_dt() == .1 && problem.calls == 29 && problem.initial_calls == 1,
        "Dense-stage rejection retains the base derivative, not the rejected endpoint"
    );
}

struct section_manager
{
    step_type<fixed_type, true> &step;
    double                      *end = nullptr;

    void set_time_interval( double, double &finish )
    {
        end = &finish;
    }

    bool apply( const vector_type &, vector_type &out )
    {
        *end = step.get_continuous_integration().evaluate( .5, out );
        return true;
    }

    integration_status get_status() const
    {
        return integration_status::stopped_by_external_operation;
    }
};

void check_external_manager( counting_operations &operations, vector_type &in, vector_type &out )
{
    operations.assign_scalar( 1., in );
    growth     problem{ operations, true };
    fixed_type fixed( { .1 } );
    using method_type = step_type<fixed_type, true>;
    method_type     step( operations, problem, fixed, { "RK45", 32, true } );
    section_manager section{ step };
    nmfd::time_steppers::integration::time_integrator<counting_operations, method_type, section_manager> integrator(
        operations, step, {}, &section
    );
    for ( unsigned int run = 0; run < 2; ++run )
    {
        integrator.apply( in, out );
        require(
            integrator.get_status() == integration_status::stopped_by_external_operation &&
                integrator.get_final_time() == .05,
            "External section refinement commits matching state and time"
        );
        const auto calls = problem.calls;
        step.set_time( integrator.get_final_time() );
        step.apply( out, out );
        require(
            step.get_status() == step_status::converged && problem.calls - calls == step.table().size(),
            "External section refinement invalidates endpoint reuse"
        );
        step.finalize( adaptation_status::accepted, .05 + step.get_dt(), out );
    }
}

int main()
try
{
    backend_type::init_device();
    counting_operations                            operations( 1 );
    nmfd::detail::vector_wrap<counting_operations> x( operations ), reference( operations ), sample( operations ),
        reference_sample( operations );
    x.start_use();
    reference.start_use();
    sample.start_use();
    reference_sample.start_use();
    for ( const auto *method : { "RK23", "RK45", "DOP853", "RK4", "HE", "RK33SSP" } )
    {
        for ( const bool forced : { false, true } )
        {
            for ( const double direction : { 1., -1. } )
            {
                check_trajectory<false>(
                    operations, *x, *reference, *sample, *reference_sample, method, forced, false, direction
                );
                check_trajectory<true>(
                    operations, *x, *reference, *sample, *reference_sample, method, forced, false, direction
                );
            }
        }
        check_trajectory<true>( operations, *x, *reference, *sample, *reference_sample, method, false, true, 1. );
    }
    for ( const auto *method : { "RK23", "RK45", "DOP853" } )
    {
        check_invalidation( operations, *x, *reference, *sample, *reference_sample, method );
        check_retries<false>( operations, *x, *reference, *sample, *reference_sample, method );
        check_retries<true>( operations, *x, *reference, *sample, *reference_sample, method );
    }
    check_failures( operations, *x, *sample );
    check_external_manager( operations, *x, *sample );
    require( operations.live == 4, "FSAL releases all internal vectors" );
    std::cout << "FSAL stage reuse, allocation, invalidation, retry, dense output and external section tests: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
