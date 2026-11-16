#include "implicit_projection_problem.h"
#include "projection_transfer_guard.h"
#include "manufactured_flow.h"
#include <nonlinear_operators/stokes/periodic_stokes_imex.h>
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace projection_tests;
using namespace stokes_validation;
using json       = nlohmann::json;
using space      = counted_space<guarded_backend>;
using vector     = space::vector_type;
using modes      = modes_type<space>;
namespace stokes = nonlinear_operators::stokes;

#if defined( TEST_VECTOR_BACKEND_CUDA )
constexpr auto backend_name = "cuda";
#elif defined( TEST_VECTOR_BACKEND_HIP )
constexpr auto backend_name = "hip";
#elif defined( TEST_VECTOR_BACKEND_OMP )
constexpr auto backend_name = "cpu_omp";
#else
constexpr auto backend_name = "cpu";
#endif

// Parseval weights for normalized R2C coefficients. Norms are physical vector
// RMS over [0,2*pi]^2, not RMS over the allocated spectral storage slots.
struct weighted_mapping
{
    vector              state, reference, out;
    std::size_t         nx, my;
    bool                difference, tail;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t index ) const
    {
        const auto ix = std::size_t( index ) / my, iy = std::size_t( index ) % my;
        const bool active = ( ix == 0 && iy == 0 ) || ( ( ix == 1 || ix == nx - 1 ) && iy == 1 ) ||
                            ( ( ix == 3 || ix == nx - 3 ) && iy == 2 );
        const double weight = ( iy == 0 || iy == my - 1 ) ? 1 : sqrt( 2. );
        for ( int j = 0; j < 4; ++j )
            out( 4 * index + j ) =
                tail && active ? 0
                               : weight * ( state( 4 * index + j ) - ( difference ? reference( 4 * index + j ) : 0 ) );
    }
};
double physical_norm(
    const space &ops, const modes &grid, const vector &state, const vector &reference, vector &scratch,
    bool difference = false, bool tail = false
)
{
    space::for_each_type each;
    each(
        weighted_mapping{
            state, reference, scratch, grid.index_space().nx(), grid.index_space().my(), difference, tail
        },
        grid.index_space().complex_size()
    );
    each.wait();
    return ops.norm( scratch );
}

// Export only five known active modes, through bounded scalar reads. A backend
// tail norm independently verifies that this representation loses no solution.
json snapshot(
    const space &ops, const modes &grid, const vector &state, const vector &reference, vector &scratch, double time
)
{
    const double tail = physical_norm( ops, grid, state, reference, scratch, false, true );
    if ( !std::isfinite( tail ) || tail > 1e-11 )
        throw std::runtime_error( "Unexported Fourier modes are not negligible" );
    json coefficients = json::array();
    for ( const auto &mode : { std::array<int, 2>{ 0, 0 }, { 1, 1 }, { -1, 1 }, { 3, 2 }, { -3, 2 } } )
    {
        const auto ix     = mode[0] < 0 ? grid.index_space().nx() + mode[0] : std::size_t( mode[0] );
        const auto index  = 4 * grid.index_space().flat_index( ix, mode[1] );
        json       values = json::array();
        for ( int j = 0; j < 4; ++j )
            values.push_back( ops.get_value_at_point( index + j, state ) );
        coefficients.push_back( { { "kx", mode[0] }, { "ky", mode[1] }, { "velocity", values } } );
    }
    return { { "time", time }, { "coefficients", coefficients }, { "omitted_rms", tail } };
}

template <bool Split>
struct benchmark_types
{
    using full_problem  = stokes::periodic_stokes<space, manufactured_flow>;
    using split_problem = stokes::periodic_stokes_imex<space, manufactured_flow>;
    using problem       = std::conditional_t<Split, split_problem, full_problem>;
    using implicit      = std::conditional_t<Split, typename split_problem::implicit_type, full_problem>;
    using solver        = typename implicit_newton<space, implicit>::solver;
    using control       = fixed_control<space>;
    using projector     = stokes::periodic_projector<space>;
    using step          = std::conditional_t<
                 Split, runge_kutta::imex_time_step<space, problem, control, solver, false, projector>,
                 runge_kutta::implicit_time_step<space, problem, control, solver, false, projector>>;
    static auto make_problem( std::shared_ptr<space> ops, std::shared_ptr<modes> grid, manufactured_flow flow )
    {
        if constexpr ( Split )
            return std::make_shared<problem>( ops, grid, .15, flow, std::array<double, 2>{ flow.cx, flow.cy } );
        else
            return std::make_shared<problem>( ops, grid, .15, flow );
    }
};

template <bool Split>
json execute( const json &config )
{
    using types         = benchmark_types<Split>;
    const auto   method = config.at( "method" ).get<std::string>();
    const auto   n      = config.at( "grid" ).get<std::size_t>();
    const auto   levels = config.at( "levels" ).get<unsigned int>();
    const auto   base   = config.at( "base_steps" ).get<std::size_t>();
    const double begin = config.at( "start" ).get<double>(), end = config.at( "end" ).get<double>();
    if ( config.at( "backend" ) != backend_name || n < 8 || n > 512 || n % 2 || base < 4 || base > 512 || base % 2 ||
         levels < 3 || levels > 8 || !std::isfinite( begin ) || !std::isfinite( end ) || end <= begin ||
         begin + ( end - begin ) / ( base << ( levels - 1 ) ) == begin )
        throw std::invalid_argument( "Invalid Stokes benchmark configuration" );
    const auto system = config.at( "system" ).get<std::string>();
    if ( system != "stokes" && system != "oseen" )
        throw std::invalid_argument( "Unknown Stokes benchmark system" );
    if ( !Split && system == "oseen" )
        throw std::invalid_argument( "DIRK validation covers the existing Stokes formulation only" );
    manufactured_flow flow;
    if ( system == "oseen" )
    {
        flow.cx = .7;
        flow.cy = -.3;
    }
    guarded_memory::check_transfers = false;
    auto grid                       = make_modes<space>( n );
    guarded_memory::check_transfers = true;
    auto ops                        = std::make_shared<space>( 4 * grid->index_space().complex_size() );
    auto projector                  = std::make_shared<typename types::projector>( ops, grid );
    json refinements = json::array(), history = json::array(), snapshots = json::array();
    for ( unsigned int level = 0; level < levels; ++level )
    {
        const auto   count   = base << level;
        const double dt      = ( end - begin ) / count;
        auto         p       = types::make_problem( ops, grid, flow );
        auto         control = std::make_shared<typename types::control>( typename types::control::params{ dt } );
        typename types::step step(
            ops, p, control, implicit_newton<space, typename types::implicit>::make( ops ),
            typename types::step::params{ method }, projector
        );
        buffer<space> state( *ops ), out( *ops ), reference( *ops ), scratch( *ops ), divergence( *ops );
        exact_state<space>( *grid, *state, begin, flow );
        const bool capture = level + 1 == levels;
        if ( capture )
            snapshots.push_back( snapshot( *ops, *grid, *state, *reference, *scratch, begin ) );
        double      time = begin, max_divergence = 0, error = 0;
        std::size_t warmed = 0;
        for ( std::size_t j = 0; j < count; ++j )
        {
            // Exactly count uniform steps; endpoint clipping handles the final
            // floating-point remainder without introducing an extra tiny step.
            const double target = j + 1 == count ? end : begin + ( j + 1 ) * dt;
            step.set_time( time );
            step.set_target_time( target );
            step.apply( *state, *out );
            if ( step.get_status() != single_step_status::converged )
                throw std::runtime_error( "Stokes benchmark stage solve failed" );
            const double next = step.get_dt() == target - time ? target : time + step.get_dt();
            step.finalize( adaptation_status::accepted, next, *out );
            ops->assign( *out, *state );
            time = next;
            exact_state<space>( *grid, *reference, time, flow );
            error = physical_norm( *ops, *grid, *state, *reference, *scratch, true );
            projector->divergence( *state, *divergence );
            const double div = physical_norm( *ops, *grid, *divergence, *reference, *scratch );
            if ( !std::isfinite( error ) || !std::isfinite( div ) )
                throw std::runtime_error( "Nonfinite benchmark diagnostic" );
            max_divergence = std::max( max_divergence, div );
            if ( j == 0 )
                warmed = guarded_memory::allocation_count;
            if ( guarded_memory::allocation_count != warmed )
                throw std::runtime_error( "Benchmark allocated backend vectors after warmup" );
            if ( capture )
            {
                history.push_back( { { "time", time }, { "error_rms", error }, { "divergence_rms", div } } );
                if ( j + 1 == count / 2 || j + 1 == count )
                    snapshots.push_back( snapshot( *ops, *grid, *state, *reference, *scratch, time ) );
            }
        }
        if ( max_divergence > 1e-11 )
            throw std::runtime_error( "Stokes benchmark divergence exceeded its bound" );
        refinements.push_back(
            { { "steps", count },
              { "step", dt },
              { "error_rms", error },
              { "maximum_divergence_rms", max_divergence },
              { "backend_allocations_after_warmup", guarded_memory::allocation_count - warmed } }
        );
    }
    unsigned int order = 0;
    if constexpr ( Split )
        order = runge_kutta::make_imex_butcher_table( method ).order;
    else
        order = runge_kutta::make_butcher_table( method ).order();
    return {
        { "format", "stokes.timestepper.validation.v1" },
        { "configuration", config },
        { "passed", true },
        { "expected_order", order },
        { "refinements", refinements },
        { "history", history },
        { "snapshots", snapshots },
        { "norm", "physical_velocity_rms" },
        { "fit_error_floor", 1e-11 },
        { "bulk_host_transfer_bytes", 0 },
        { "exported_scalar_values", snapshots.size() * 20 },
        { "pressure_reconstructed", false }
    };
}

int main( int argc, char **argv )
try
{
    if ( argc != 3 )
        throw std::invalid_argument( "Usage: stokes_benchmark config.json report.json" );
    std::ifstream input( argv[1] );
    const auto    config = json::parse( input );
    backend_type::init_device();
    const auto    method = config.at( "method" ).get<std::string>();
    const auto    report = method.rfind( "IMEX_", 0 ) == 0 ? execute<true>( config ) : execute<false>( config );
    std::ofstream output( argv[2] );
    output.exceptions( std::ios::badbit | std::ios::failbit );
    output << report.dump( 2 ) << '\n';
    if ( !guarded_memory::allocations.empty() )
        throw std::runtime_error( "Stokes benchmark backend allocation leak" );
    std::cout << config.at( "system" ) << ' ' << method << ": exported " << report.at( "snapshots" ).size()
              << " solution snapshots\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
