#ifndef PERIODIC_ORBIT_TESTS_KS1D_BRANCH_FLIGHTS_H
#define PERIODIC_ORBIT_TESTS_KS1D_BRANCH_FLIGHTS_H
#include "ks1d_orbit_support.h"
#include <periodic_orbit/matching/parameterized_multiple_section_problem.h>
#include <periodic_orbit/matching/rechart_orbit.h>

namespace ks1d_orbit
{
struct binding
{
    std::shared_ptr<operations> physical;
    std::shared_ptr<rate>       trial, replay;

    void set_trial_parameter( double p ) const
    {
        trial->set_parameter( p );
    }
    matching::factory_type make_factory( double p ) const
    {
        return { physical, std::make_shared<const rate>( replay->with_parameter( p ) ) };
    }
};

struct branch_flights
{
    static constexpr auto        dimension        = seed.size();
    static constexpr const char *observable_label = "l2_norm";
    using space                                   = periodic_orbit::orbit_continuation_space<matching::space_type>;
    using state                                   = space::vector_type;
    using orbit = periodic_orbit::parameterized_multiple_section_problem<matching, binding>;
    std::shared_ptr<operations> physical, coordinates;
    std::shared_ptr<space>      coords;
    double                      pilot_parameter, period;
    binding rates{ physical, provider( physical, pilot_parameter ), provider( physical, pilot_parameter ) };
    std::shared_ptr<rate>                        verification = provider( physical, pilot_parameter );
    std::shared_ptr<rk>                          fine;
    std::shared_ptr<const matching::layout_type> layout;
    std::vector<std::shared_ptr<chart>>          charts;
    std::vector<std::size_t>                     pilot_indices;
    std::shared_ptr<chart>                       first;
    std::shared_ptr<orbit>                       op;

    branch_flights( std::size_t count, double h, double start, std::size_t maximum_steps )
        : physical( std::make_shared<operations>( dimension ) ),
          coordinates( std::make_shared<operations>( dimension - 1 ) ),
          coords( std::make_shared<space>( coordinates, count ) ), pilot_parameter( 68. ), period( ks1d_orbit::period )
    {
        auto initial = buffer( physical );
        fill_profile( *initial, seed );
        assemble( *initial, h, start, maximum_steps );
    }
    branch_flights( const branch_flights &old, const X &initial, double p, double T, double h, double t, std::size_t n )
        : physical( old.physical ), coordinates( old.coordinates ), coords( old.coords ), pilot_parameter( p ),
          period( T )
    {
        assemble( initial, h, t, n );
    }
    void assemble( const X &initial, double h, double t, std::size_t n )
    {
        layout        = make_layout( physical, rates.trial, initial, period, coords->component_count(), h, t, n );
        assemble_layout( h, n );
    }
    branch_flights( std::shared_ptr<operations> phys, std::shared_ptr<const matching::layout_type> saved,
                    double p, double T, double h, std::size_t n )
        : physical( std::move( phys ) ), coordinates( std::make_shared<operations>( dimension - 1 ) ),
          coords( std::make_shared<space>( coordinates, saved->size() ) ), pilot_parameter( p ), period( T ),
          layout( std::move( saved ) )
    {
        assemble_layout( h, n );
    }
    void assemble_layout( double h, std::size_t n )
    {
        fine = stepper( physical, verification, h / 4 );
        auto segments = make_segments( physical, coordinates, layout, rates.trial, h, n );
        for ( std::size_t i = 0; i < segments.size(); ++i )
        {
            charts.push_back( segments[i].chart );
            pilot_indices.push_back( layout->at( i ).pilot_step );
        }
        first        = charts.front();
        auto matcher = std::make_shared<matching>( physical, coords, layout, std::move( segments ), rates.replay );
        op           = std::make_shared<orbit>( matcher, rates, coords );
    }
    auto workspace() const
    {
        return nmfd::detail::vector_wrap<space, true, true>( *coords );
    }
    // The KS diagram's l2_norm column, without downloading unused low-mode columns.
    double observable( const X &x ) const
    {
        return physical->norm_l2( x );
    }
    auto verification_step( double p ) const
    {
        verification->set_parameter( p );
        return fine;
    }
    bool rechart_previous( X &initial, double p, double T, double h, std::size_t n, state &out ) const
    {
        using flight = periodic_orbit::section_map<operations, rk, plane, policy, rate>;
        rates.set_trial_parameter( p );
        flight::params params;
        params.maximum_steps = n;
        flight f(
            physical, stepper( physical, rates.trial, h ), std::make_shared<policy>( physical ), params, rates.trial
        );
        return periodic_orbit::rechart_orbit( f, *layout, charts, initial, T, out );
    }
};
}
#endif
