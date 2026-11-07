#ifndef PERIODIC_ORBIT_SECTION_TEST_SUPPORT_H
#define PERIODIC_ORBIT_SECTION_TEST_SUPPORT_H

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda_common.h>
using native_backend = scfd::backend::cuda_common<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <scfd/backend/hip_common.h>
using native_backend = scfd::backend::hip_common<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using native_backend = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using native_backend = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_constant.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <periodic_orbit/sections/automatic_layout.h>
#include <periodic_orbit/flows/recording_stepper.h>

namespace section_tests
{
inline std::size_t checks = 0;
inline void        require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}
template <class Exception, class Action>
void require_throws( Action action, const char *message )
{
    bool caught = false;
    try
    {
        action();
    }
    catch ( const Exception & )
    {
        caught = true;
    }
    require( caught, message );
}

// All vector storage uses this memory adapter. Scalar reductions/point reads are
// permitted; a vector-sized host transfer fails the test on every backend.
struct guarded_memory : native_backend::memory_type
{
    using parent                               = native_backend::memory_type;
    inline static std::size_t host_bytes       = 0;
    inline static std::size_t allocation_calls = 0, live_allocations = 0;
    static void               malloc( void **pointer, std::size_t bytes )
    {
        parent::malloc( pointer, bytes );
        if ( bytes )
        {
            ++allocation_calls;
            ++live_allocations;
        }
    }
    static void free( void *pointer )
    {
        if ( pointer )
            --live_allocations;
        parent::free( pointer );
    }
    static void copy_to_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "Vector-sized device-to-host copy" );
        host_bytes += bytes;
        parent::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "Vector-sized host-to-device copy" );
        host_bytes += bytes;
        parent::copy_from_host( bytes, src, dst );
    }
};
struct backend : native_backend
{
    using memory_type = guarded_memory;
};
using operations       = nmfd::operations::scfd_vector_space<backend, double>;
using X                = operations::vector_type;
using wrap             = nmfd::detail::vector_wrap<operations, false, false>;
using plane            = periodic_orbit::affine_section<operations>;
using policy           = periodic_orbit::identity_state_policy<operations>;
using fixed            = nmfd::time_steppers::integration::time_step_adaptation_constant<operations>;
using adaptive         = nmfd::time_steppers::integration::time_step_adaptation_matlab<operations>;
using status           = periodic_orbit::section_status;
inline const double pi = std::acos( -1. );

inline wrap buffer( const std::shared_ptr<operations> &ops )
{
    wrap v( *ops );
    v.init();
    v.start_use();
    return v;
}
struct fill_kernel
{
    double             *out;
    double              a, b, c;
    int                 components;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out[i] = i % components == 0 ? a : i % components == 1 ? b : c;
    }
};
inline void fill( const operations &ops, X &out, double a, double b, double c = 0, int components = 2 )
{
    backend::for_each_type{}( fill_kernel{ out.raw_ptr(), a, b, c, components }, out.size() );
}

// Unit harmonic oscillator: x'=-y, y'=x. Exact period 2*pi, transverse
// x=0 and y=0 crossings. Independent copies permit large backend-only tests.
struct harmonic
{
    __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double ) const
    {
        return i % 2 ? u[i - 1] : -u[i + 1];
    }
};
template <class Model>
struct rhs_kernel
{
    const double       *in;
    double             *out;
    double              time;
    Model               model;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out[i] = model.rhs( i, in, time );
    }
};
template <class Model>
struct problem
{
    std::shared_ptr<operations> ops;
    Model                       model;
    double                      time    = 0;
    bool                        invalid = false;
    void                        set_time( double t )
    {
        time = t;
    }
    bool evaluate_rate( const X &in, X &out )
    {
        if ( invalid )
        {
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
            return true;
        }
        backend::for_each_type{}( rhs_kernel<Model>{ in.raw_ptr(), out.raw_ptr(), time, model }, out.size() );
        return true;
    }
};
// An independent provider avoids changing the timestepper's prepared problem context.
template <class Model>
struct rate_provider
{
    Model model{};
    bool  failure = false;
    bool  evaluate( double t, const X &in, X &out )
    {
        if ( failure )
            return false;
        backend::for_each_type{}( rhs_kernel<Model>{ in.raw_ptr(), out.raw_ptr(), t, model }, out.size() );
        return true;
    }
};
template <class Model, class Controller = fixed>
class step : public nmfd::time_steppers::runge_kutta::explicit_time_step<operations, problem<Model>, Controller, true>
{
    using base = nmfd::time_steppers::runge_kutta::explicit_time_step<operations, problem<Model>, Controller, true>;

public:
    using base::base;
    std::size_t modified = 0, failed = 0;
    void        finalize( nmfd::time_steppers::adaptation_status state, double t, const X &x )
    {
        modified += state == nmfd::time_steppers::adaptation_status::accepted_modified;
        failed += state == nmfd::time_steppers::adaptation_status::failed;
        base::finalize( state, t, x );
    }
};
template <class Model>
auto make_step( const std::shared_ptr<operations> &ops, double dt = .05, Model model = {} )
{
    auto rhs = std::make_shared<problem<Model>>( problem<Model>{ ops, model } );
    return std::make_shared<step<Model>>(
        ops, rhs, std::make_shared<fixed>( fixed::params{ dt } ), typename step<Model>::params{ "RK45" }
    );
}
template <class Model>
struct explicit_step_assembly
{
    using step_type   = step<Model>;
    using record_type = periodic_orbit::flight_step_record<typename step_type::scalar_type>;
    static auto make_stepper( const std::shared_ptr<operations> &ops, double h, const std::string &method )
    {
        return std::make_shared<step_type>(
            ops, std::make_shared<problem<Model>>( problem<Model>{ ops, {} } ),
            std::make_shared<fixed>( fixed::params{ h } ), typename step_type::params{ method }
        );
    }
};
inline std::shared_ptr<const plane>
make_plane( const std::shared_ptr<operations> &ops, double cx, double cy, double nx, double ny )
{
    auto center = buffer( ops ), normal = buffer( ops );
    fill( *ops, *center, cx, cy );
    fill( *ops, *normal, nx, ny );
    return std::make_shared<plane>( ops, *center, *normal );
}
inline void near_state( const std::shared_ptr<operations> &ops, const X &state, double x, double y, double tol )
{
    auto expected = buffer( ops ), diff = buffer( ops );
    fill( *ops, *expected, x, y );
    ops->assign_lin_comb( 1, state, -1, *expected, *diff );
    require( ops->norm_l2( *diff ) < tol, "Incorrect event state" );
}
template <class Test>
int run( Test test )
{
    try
    {
        test();
        std::cout << checks << " checks passed; guarded host bytes=" << guarded_memory::host_bytes << '\n';
        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
}
#endif
