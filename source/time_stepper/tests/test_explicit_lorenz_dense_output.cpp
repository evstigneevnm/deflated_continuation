#include "explicit_lorenz_problem.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <scfd/static_vec/vec.h>
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
#include <time_stepper/integration/time_step_adaptation_matlab.h>
#include <time_stepper/integration/time_step_adaptation_scipy.h>
#include <time_stepper/integration/time_integrator.h>

using operations_type           = scfd_vector_operations<backend_type, double>;
using vector_type               = operations_type::vector_type;
using integration_status        = nmfd::time_steppers::integration_status;
constexpr double duration       = .1;
const double     sample_times[] = { .13 * duration, .37 * duration, .71 * duration, .93 * duration, duration };
// Independent 40-digit Decimal RK4, h=T/10000 and T/20000; agreement < 6e-16.
const scfd::static_vec::vec<double, 3> reference[] = {
    { 5.6889203524481688, 31.378119553659648, 3.9886645769545415 },
    { 11.448080327261721, 34.900648934834862, 10.429758871378498 },
    { 18.289905843889304, 38.214512775374551, 27.813392419621337 },
    { 21.07648244236281, 34.348143264744735, 41.820033838432565 },
    { 21.527600219626681, 31.740750381438446, 45.939726836031014 }
};

void require( bool ok, const char *message )
{
    if ( !ok )
    {
        throw std::runtime_error( message );
    }
}

// The manager borrows the step; the integrator needs no dense-output-specific hooks.
template <class Step>
struct external_manager
{
    operations_type                           &ops;
    Step                                      &step;
    nmfd::detail::vector_wrap<operations_type> sample;
    double                                     start = 0, *end = nullptr, error = 0;
    unsigned int                               samples = 0, interior_samples = 0;
    bool                                       stop   = false;
    integration_status                         status = integration_status::running;

    external_manager( operations_type &operations, Step &method )
        : ops( operations ), step( method ), sample( operations )
    {
        sample.start_use();
    }

    void reset( bool early_stop = false )
    {
        samples = interior_samples = 0;
        error                      = 0;
        stop                       = early_stop;
        status                     = integration_status::running;
    }

    void set_time_interval( const double &t, double &next )
    {
        start = t;
        end   = &next;
    }

    bool apply( const vector_type &, vector_type &out )
    {
        const auto dense = step.get_continuous_integration();
        while ( samples < 5 && sample_times[samples] <= *end )
        {
            require( sample_times[samples] >= start, "Skipped sample time" );
            const auto theta = std::clamp( ( sample_times[samples] - start ) / step.get_dt(), 0., 1. );
            const auto time  = dense.evaluate( theta, *sample );
            require( std::abs( time - sample_times[samples] ) < 1e-14, "Dense sample time" );
            if ( theta > 0 && theta < 1 )
            {
                ++interior_samples;
            }
            scfd::static_vec::vec<double, 3> values;
            ops.get( *sample, values.d );
            for ( int i = 0; i < 3; ++i )
            {
                error = std::max( error, std::abs( values[i] - reference[samples][i] ) );
            }
            ++samples;
            if ( stop && samples == 4 )
            {
                ops.assign( *sample, out );
                *end   = time;
                status = integration_status::stopped_by_external_operation;
                return true;
            }
        }
        return false;
    }

    integration_status get_status() const
    {
        return status;
    }
};

template <class Adaptation, class Problem>
void check_adaptive(
    operations_type &ops, Problem &problem, const vector_type &initial, vector_type &result, const char *method
)
{
    typename Adaptation::params p;
    p.initial_step       = duration;
    p.relative_tolerance = 1e-11;
    p.absolute_tolerance = 1e-13;
    Adaptation adaptation( ops, p );
    using step_type = nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, Problem, Adaptation, true>;
    step_type                   step( ops, problem, adaptation, { method } );
    external_manager<step_type> external( ops, step );
    nmfd::time_steppers::integration::time_integrator<operations_type, step_type, decltype( external )> integrator(
        ops, step, {}, &external
    );
    integrator.set_time_interval( 0, duration );
    for ( int run = 0; run < 3; ++run )
    {
        external.reset( run == 2 );
        integrator.apply( initial, result );
        require( external.error < 2e-8 && external.interior_samples >= 4, "Independent dense reference mismatch" );
        require( external.samples == ( run == 2 ? 4u : 5u ), "Restart must sample each point exactly once" );
        require(
            integrator.get_status() ==
                ( run == 2 ? integration_status::stopped_by_external_operation : integration_status::completed ),
            "Dense integration status"
        );
        require(
            std::abs( integrator.get_final_time() - ( run == 2 ? sample_times[3] : duration ) ) < 1e-14,
            "Dense manager must commit matching time"
        );
        scfd::static_vec::vec<double, 3> final_state;
        ops.get( result, final_state.d );
        for ( int i = 0; i < 3; ++i )
        {
            require(
                std::abs( final_state[i] - reference[run == 2 ? 3 : 4][i] ) < 2e-8, "Dense manager committed state"
            );
        }
    }
}

int main()
try
{
    backend_type::init_device();
    operations_type                            ops( 3 );
    nmfd::detail::vector_wrap<operations_type> initial( ops ), result( ops );
    initial.start_use();
    result.start_use();
    const scfd::static_vec::vec<double, 3> values{ 2.2, 30.5, 2.5 };
    ops.set( values.d, *initial );
    nmfd::time_steppers::tests::explicit_lorenz_problem<operations_type> problem;
    // The stored references are for the damped model, independent of the example default.
    problem.epsilon  = .0055;
    using fixed_type = nmfd::time_steppers::integration::time_step_adaptation_constant<operations_type>;
    using fixed_step =
        nmfd::time_steppers::runge_kutta::explicit_time_step<operations_type, decltype( problem ), fixed_type, true>;
    for ( const auto *method : { "EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK64SSP", "RK45", "DOP853" } )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            fixed_type                   fixed( { duration / ( 10 * ( level + 1 ) ) } );
            fixed_step                   step( ops, problem, fixed, { method } );
            external_manager<fixed_step> external( ops, step );
            nmfd::time_steppers::integration::time_integrator<operations_type, fixed_step, decltype( external )>
                integrator( ops, step, {}, &external );
            integrator.set_time_interval( 0, duration );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed && external.samples == 5 &&
                    external.interior_samples >= 4,
                "Fixed dense output sampling"
            );
            errors[level] = external.error;
            if ( step.table().order() == 8 )
            {
                external.reset( true );
                integrator.apply( *initial, *result );
                require(
                    integrator.get_status() == integration_status::stopped_by_external_operation &&
                        external.samples == 4 && external.error < 2e-8 &&
                        std::abs( integrator.get_final_time() - sample_times[3] ) < 1e-14,
                    "DOP853 external manager must commit a shortened dense endpoint"
                );
                scfd::static_vec::vec<double, 3> endpoint;
                ops.get( *result, endpoint.d );
                for ( int i = 0; i < 3; ++i )
                {
                    require( std::abs( endpoint[i] - reference[3][i] ) < 2e-8, "DOP853 external committed state" );
                }
            }
        }
        require( errors[1] < std::max( 1e-11, .8 * errors[0] ), "Fixed dense refinement must improve accuracy" );
    }
    using matlab_type = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>;
    for ( const auto *method : { "RK45", "RK23" } )
    {
        check_adaptive<matlab_type>( ops, problem, *initial, *result, method );
    }
    using scipy_type = nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>;
    check_adaptive<scipy_type>( ops, problem, *initial, *result, "DOP853" );
    std::cout << "lorenz: dense output, eight fixed methods, adaptive restart and external endpoint refinement PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
