#ifndef PERIODIC_ORBIT_PREPARED_EXPLICIT_SECTION_H
#define PERIODIC_ORBIT_PREPARED_EXPLICIT_SECTION_H

#include <limits>
#include <nmfd/time_steppers/runge_kutta/butcher_table.h>
#include <periodic_orbit/flows/recording_stepper.h>
#include <periodic_orbit/flows/prepared_rk_section.h>

namespace periodic_orbit
{
// One reusable backend workspace for a serial sequence of prepared flights.
// The provider evaluates the physical rate and its Jv at the supplied stage state.
template <class Space, class RateProvider>
class explicit_flight_replay
{
public:
    using T          = typename Space::scalar_type;
    using X          = typename Space::vector_type;
    using table_type = nmfd::time_steppers::runge_kutta::butcher_table;
    using wrap       = nmfd::detail::vector_wrap<Space, false, false>;
    static bool supports( const table_type &table )
    {
        return table.type() == table_type::scheme_type::explicit_rk && table.has_dense_output();
    }
    explicit_flight_replay( std::shared_ptr<Space> space, std::shared_ptr<const RateProvider> rate, std::size_t stages )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing replay space" ) ),
          rate_( rate ? std::move( rate ) : throw std::invalid_argument( "Missing replay rate/Jv provider" ) ),
          state_( *space_ ), tangent_( *space_ ), stage_state_( *space_ ), stage_tangent_( *space_ )
    {
        for ( auto *x : { &state_, &tangent_, &stage_state_, &stage_tangent_ } )
        {
            x->init();
            x->start_use();
        }
        rates_.reserve( stages );
        tangents_.reserve( stages );
        for ( std::size_t i = 0; i < stages; ++i )
        {
            rates_.emplace_back( *space_ );
            rates_.back().init();
            rates_.back().start_use();
            tangents_.emplace_back( *space_ );
            tangents_.back().init();
            tangents_.back().start_use();
        }
    }

    // No controller is invoked: both the base and tangent use the frozen mesh.
    void
    run( const table_type &table, const std::vector<flight_step_record<T>> &steps, const X &initial, const X *direction,
         X &endpoint, X *derivative, X *time_rate )
    {
        if ( table.type() != table_type::scheme_type::explicit_rk || !table.has_dense_output() || steps.empty() ||
             table.dense_outout_stage_count() > rates_.size() || bool( direction ) != bool( derivative ) ||
             &endpoint == derivative || &endpoint == time_rate || ( derivative && derivative == time_rate ) )
            throw std::invalid_argument( "Invalid flight replay workspace" );
        space_->assign( initial, *state_ );
        if ( direction )
            space_->assign( *direction, *tangent_ );
        for ( std::size_t step = 0; step < steps.size(); ++step )
        {
            const auto &r     = steps[step];
            const bool  last  = step + 1 == steps.size();
            const auto  count = last ? table.dense_outout_stage_count() : table.size();
            for ( std::size_t i = 0; i < count; ++i )
            {
                space_->assign( *state_, *stage_state_ );
                if ( direction )
                    space_->assign( *tangent_, *stage_tangent_ );
                for ( std::size_t j = 0; j < i; ++j )
                {
                    const T a = static_cast<T>( table.dense_outout_a( i, j ) );
                    if ( a == T( 0 ) )
                        continue;
                    space_->add_lin_comb( r.dt * a, *rates_[j], *stage_state_ );
                    if ( direction )
                        space_->add_lin_comb( r.dt * a, *tangents_[j], *stage_tangent_ );
                }
                const T time = r.time + r.dt * static_cast<T>( table.dense_outout_c( i ) );
                if ( !rate_->evaluate( time, *stage_state_, *rates_[i] ) || !space_->is_valid_number( *rates_[i] ) )
                    throw std::runtime_error( "Prepared flight rate evaluation failed" );
                if ( direction && ( !rate_->apply_jacobian( time, *stage_state_, *stage_tangent_, *tangents_[i] ) ||
                                    !space_->is_valid_number( *tangents_[i] ) ) )
                    throw std::runtime_error( "Prepared flight tangent evaluation failed" );
            }
            if ( last && time_rate )
            {
                space_->assign_scalar( T( 0 ), *time_rate );
                for ( std::size_t i = 0; i < count; ++i )
                {
                    // Differentiate the same dense polynomial, not the physical
                    // RHS at an interpolated endpoint. dt cancels with dtheta/dt.
                    long double weight = 0;
                    for ( auto j = table.dense_degree(); j > 0; --j )
                        weight = weight * r.theta + j * table.dense_coefficient( i, j - 1 );
                    if ( weight != 0 )
                        space_->add_lin_comb( static_cast<T>( weight ), *rates_[i], *time_rate );
                }
            }
            for ( std::size_t i = 0; i < count; ++i )
            {
                const T b = last && r.theta != T( 1 ) ? static_cast<T>( table.dense_b( i, r.theta ) )
                                                      : ( i < table.size() ? static_cast<T>( table.b( i ) ) : T( 0 ) );
                if ( b == T( 0 ) )
                    continue;
                space_->add_lin_comb( r.dt * b, *rates_[i], *state_ );
                if ( direction )
                    space_->add_lin_comb( r.dt * b, *tangents_[i], *tangent_ );
            }
            if ( !space_->is_valid_number( *state_ ) || ( direction && !space_->is_valid_number( *tangent_ ) ) )
                throw std::runtime_error( "Nonfinite prepared flight replay" );
        }
        space_->assign( *state_, endpoint );
        if ( derivative )
            space_->assign( *tangent_, *derivative );
    }

private:
    std::shared_ptr<Space>              space_;
    std::shared_ptr<const RateProvider> rate_;
    wrap                                state_, tangent_, stage_state_, stage_tangent_;
    std::vector<wrap>                   rates_, tangents_;
};

template <class Space, class RateProvider, class Section>
using prepared_explicit_section = prepared_rk_section<Space, explicit_flight_replay<Space, RateProvider>, Section>;

template <class Segments>
std::size_t maximum_rk_stages( const Segments &segments )
{
    std::size_t stages = 0;
    for ( const auto &s : segments )
        stages = std::max( stages, s.flight->table().dense_outout_stage_count() );
    return stages;
}

template <class Space, class RateProvider, class Section>
class explicit_flight_factory
{
public:
    using flow_type = prepared_explicit_section<Space, RateProvider, Section>;
    explicit_flight_factory( std::shared_ptr<Space> space, std::shared_ptr<const RateProvider> rate )
        : space_( std::move( space ) ), rate_( std::move( rate ) )
    {
        if ( !space_ || !rate_ )
            throw std::invalid_argument( "Missing explicit replay dependencies" );
    }
    template <class Segments>
    auto make_workspace( const Segments &segments ) const
    {
        return std::make_shared<typename flow_type::replay_type>( space_, rate_, maximum_rk_stages( segments ) );
    }
    template <class Map>
    flow_type prepare(
        std::shared_ptr<typename flow_type::replay_type> workspace, const Map &flight,
        const typename Space::vector_type &initial, const typename Space::vector_type &arrival,
        std::shared_ptr<const Section> target, typename Space::scalar_type speed, typename Space::scalar_type tolerance
    ) const
    {
        return { space_,  std::move( workspace ), flight.table(), flight.records(), initial,
                 arrival, std::move( target ),    speed,          tolerance };
    }

private:
    std::shared_ptr<Space>              space_;
    std::shared_ptr<const RateProvider> rate_;
};
}
#endif
