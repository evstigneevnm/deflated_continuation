#ifndef PERIODIC_ORBIT_BENCHMARK_ODE_CONTINUATION_FIXTURE_H
#define PERIODIC_ORBIT_BENCHMARK_ODE_CONTINUATION_FIXTURE_H

#include "../ode_orbit_support.h"
#include "branch_solver.h"
#include <periodic_orbit/matching/rechart_orbit.h>

namespace ode_orbit_tests
{
template <class Model>
struct parameter_rate : tangent_rate<Model>
{
    struct source
    {
        Model                 model;
        __DEVICE_TAG__ double rhs( std::ptrdiff_t i, const double *u, double t ) const
        {
            return model.parameter_rhs( i, u, t );
        }
    };
    bool parameter_derivative( double t, const X &u, X &out ) const
    {
        backend::for_each_type{}( rhs_kernel<source>{ u.raw_ptr(), out.raw_ptr(), t, { this->model } }, out.size() );
        return true;
    }
};

template <class Model>
struct continuation_fixture
{
    static constexpr auto dimension = Model::seed.size();
    using rk_type                   = step<Model>;
    using map      = periodic_orbit::recorded_section_map<operations, rk_type, plane, rate_provider<Model>>;
    using matching = periodic_orbit::multiple_section_problem<operations, operations, map, parameter_rate<Model>>;
    using space    = periodic_orbit::orbit_continuation_space<typename matching::space_type>;
    using state    = typename space::vector_type;
    struct binding
    {
        Model                                 anchor;
        std::shared_ptr<operations>           physical;
        std::shared_ptr<problem<Model>>       rhs;
        std::shared_ptr<rate_provider<Model>> crossing;
        void                                  set_trial_parameter( double p )
        {
            rhs->model = crossing->model = anchor.with_parameter( p );
        }
        typename matching::factory_type make_factory( double p ) const
        {
            auto snapshot   = std::make_shared<parameter_rate<Model>>();
            snapshot->model = anchor.with_parameter( p );
            return { physical, snapshot };
        }
    };
    using orbit                             = periodic_orbit::parameterized_multiple_section_problem<matching, binding>;
    std::shared_ptr<operations> physical    = std::make_shared<operations>( dimension ),
                                coordinates = std::make_shared<operations>( dimension - 1 );
    std::shared_ptr<space>                                coords;
    std::shared_ptr<orbit>                                op;
    std::shared_ptr<chart>                                first;
    std::shared_ptr<const typename matching::layout_type> layout;
    std::vector<std::shared_ptr<chart>>                   charts;
    std::vector<std::size_t>                              pilot_indices;
    const Model                                           model;
    const double                                          period;
    const double                                          pilot_parameter;
    continuation_fixture(
        const Model &anchor, const std::array<double, dimension> &pilot_state, double pilot_period, std::size_t m,
        double h, double t, std::size_t budget
    )
        : coords( std::make_shared<space>( coordinates, m ) ), model( anchor ), period( pilot_period ),
          pilot_parameter( Model::anchor_parameter )
    {
        layout = anchor_layout( physical, make_step( physical, h, model ), model, pilot_state, period, m, t, budget );
        assemble( h, budget );
    }
    continuation_fixture(
        const continuation_fixture &old, const X &seed, double p, double new_period, double h, double start,
        std::size_t budget
    )
        : physical( old.physical ), coordinates( old.coordinates ), coords( old.coords ),
          model( old.model.with_parameter( p ) ), period( new_period ), pilot_parameter( p )
    {
        layout = backend_layout(
            physical, make_step( physical, h, model ), model, seed, period, coords->component_count(), start, budget
        );
        assemble( h, budget );
    }
    void assemble( double h, std::size_t budget )
    {
        binding b{
            model, physical, std::make_shared<problem<Model>>( problem<Model>{ physical, model } ),
            std::make_shared<rate_provider<Model>>()
        };
        b.crossing->model = model;
        std::vector<typename matching::segment> segments;
        typename map::map_type::params          event;
        event.maximum_steps = budget;
        for ( std::size_t i = 0; i < layout->size(); ++i )
        {
            auto rk = std::make_shared<rk_type>(
                physical, b.rhs, std::make_shared<fixed>( fixed::params{ h } ), typename rk_type::params{ "RK45" }
            );
            auto section = std::make_shared<chart>( physical, coordinates, layout->at( i ).section );
            if ( i == 0 )
                first = section;
            charts.push_back( section );
            pilot_indices.push_back( layout->at( i ).pilot_step );
            segments.push_back( { std::make_shared<map>( physical, rk, b.crossing, event ), section } );
        }
        auto rate    = std::make_shared<parameter_rate<Model>>();
        rate->model  = model;
        auto matcher = std::make_shared<matching>( physical, coords, layout, std::move( segments ), rate );
        op           = std::make_shared<orbit>( matcher, b, coords );
    }
    auto workspace() const
    {
        nmfd::detail::vector_wrap<space, false, false> x( *coords );
        x.init();
        x.start_use();
        return x;
    }
    bool
    rechart_previous( X &physical_seed, double p, double old_period, double h, std::size_t budget, state &out ) const
    {
        using flight_type = periodic_orbit::section_map<operations, rk_type, plane, policy, rate_provider<Model>>;
        auto rate         = std::make_shared<rate_provider<Model>>();
        rate->model       = model.with_parameter( p );
        typename flight_type::params params;
        params.maximum_steps = budget;
        flight_type flight(
            physical, make_step( physical, h, rate->model ), std::make_shared<policy>( physical ), params, rate
        );
        return periodic_orbit::rechart_orbit( flight, *layout, charts, physical_seed, old_period, out );
    }
};
}
#endif
