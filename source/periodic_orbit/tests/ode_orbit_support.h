#ifndef PERIODIC_ORBIT_ODE_ORBIT_SUPPORT_H
#define PERIODIC_ORBIT_ODE_ORBIT_SUPPORT_H

#include "variational_test_support.h"
#include "newton_test_support.h"
#include <periodic_orbit/matching/multiple_section_problem.h>

namespace ode_orbit_tests
{
using namespace section_tests;
using chart = periodic_orbit::affine_chart<operations>;

template <class Model>
struct fixture
{
    using map           = periodic_orbit::recorded_section_map<operations, step<Model>, plane, rate_provider<Model>>;
    using orbit_problem = periodic_orbit::multiple_section_problem<operations, operations, map, tangent_rate<Model>>;
    using residual_type = typename orbit_problem::residual_type;
    using blocks        = typename orbit_problem::space_type;
    std::shared_ptr<operations>                                physical    = std::make_shared<operations>( 3 );
    std::shared_ptr<operations>                                coordinates = std::make_shared<operations>( 2 );
    std::shared_ptr<blocks>                                    space;
    std::shared_ptr<const typename orbit_problem::layout_type> layout;
    std::shared_ptr<tangent_rate<Model>>                       rate = std::make_shared<tangent_rate<Model>>();
    std::shared_ptr<orbit_problem>                             problem;
    double                                                     dt, start;
    std::size_t                                                maximum_steps;
    std::string                                                method;

    fixture( std::size_t count, double h, std::string scheme = "DOP853", double t = .37, std::size_t budget = 20000 )
        : space( std::make_shared<blocks>( coordinates, count ) ), dt( h ), start( t ), maximum_steps( budget ),
          method( std::move( scheme ) )
    {
        auto initial = buffer( physical );
        fill( *physical, *initial, Model::seed[0], Model::seed[1], Model::seed[2], 3 );
        periodic_orbit::automatic_layout<operations, step<Model>, rate_provider<Model>> builder(
            physical, make_stepper( dt ), std::make_shared<rate_provider<Model>>(),
            std::make_shared<policy>( physical ),
            std::make_shared<periodic_orbit::euclidean_section_metric<operations>>( physical )
        );
        periodic_orbit::placement_params<double> p;
        p.sections                  = count;
        p.maximum_pilot_steps       = maximum_steps;
        p.maximum_endpoint_distance = 1e-4;
        auto result                 = builder.build( *initial, { start, start + Model::period, nullptr }, p );
        if ( !result.succeeded() )
            throw std::runtime_error(
                "ODE pilot layout failed at section " + std::to_string( result.failed_section ) +
                ", M=" + std::to_string( count ) + ", status=" + std::to_string( int( result.status ) ) +
                ", flight=" + std::to_string( result.flight.flight_time )
            );
        layout  = std::move( result.layout );
        problem = std::make_shared<orbit_problem>( physical, space, layout, make_segments( dt ), rate );
    }
    auto make_stepper( double h ) const
    {
        auto rhs = std::make_shared<section_tests::problem<Model>>( section_tests::problem<Model>{ physical, {} } );
        return std::make_shared<step<Model>>(
            physical, rhs, std::make_shared<fixed>( fixed::params{ h } ), typename step<Model>::params{ method }
        );
    }
    auto block_buffer() const
    {
        nmfd::detail::vector_wrap<blocks, false, false> x( *space );
        x.init();
        x.start_use();
        return x;
    }
    auto make_segments( double h ) const
    {
        std::vector<typename orbit_problem::segment> result;
        typename map::map_type::params               p;
        p.maximum_steps = maximum_steps;
        for ( std::size_t i = 0; i < layout->size(); ++i )
            result.push_back(
                { std::make_shared<map>( physical, make_stepper( h ), std::make_shared<rate_provider<Model>>(), p ),
                  std::make_shared<chart>( physical, coordinates, layout->at( i ).section ) }
            );
        return result;
    }
    void perturb( double amplitude, typename blocks::vector_type &out ) const
    {
        for ( std::size_t i = 0; i < layout->size(); ++i )
            fill( *coordinates, out.comp( i ), amplitude * ( i + 1 ) / layout->size(), -.5 * amplitude );
    }
    double closure( const typename blocks::vector_type &state, double period, double h ) const
    {
        auto  initial = buffer( physical ), endpoint = buffer( physical );
        chart section( physical, coordinates, layout->at( 0 ).section );
        section.lift( state.comp( 0 ), *initial );
        using integrator = nmfd::time_steppers::integration::time_integrator<operations, step<Model>>;
        integrator verify( physical, make_stepper( h ), typename integrator::params{ maximum_steps } );
        verify.set_time_interval( start, start + period );
        verify.apply( *initial, *endpoint );
        if ( verify.get_status() != nmfd::time_steppers::integration_status::completed )
            throw std::runtime_error( "Independent ODE orbit integration failed" );
        physical->add_lin_comb( -1, *initial, *endpoint );
        return physical->norm( *endpoint );
    }
};
}
#endif
