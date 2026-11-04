#ifndef PERIODIC_ORBIT_PREPARED_EXPLICIT_SECTION_H
#define PERIODIC_ORBIT_PREPARED_EXPLICIT_SECTION_H

#include <limits>
#include <nmfd/time_steppers/runge_kutta/butcher_table.h>
#include <periodic_orbit/flows/recording_stepper.h>

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

// Initial state + bounded scalar schedule form a one-checkpoint replay policy.
// Immutable preparation survives trial evaluations; scratch is not concurrent.
template <class Space, class RateProvider, class Section>
class prepared_explicit_section
{
public:
    using T           = typename Space::scalar_type;
    using X           = typename Space::vector_type;
    using traits      = scfd::utils::scalar_traits<T>;
    using replay_type = explicit_flight_replay<Space, RateProvider>;
    using table_type  = typename replay_type::table_type;
    using wrap        = nmfd::detail::vector_wrap<Space, false, false>;
    prepared_explicit_section(
        std::shared_ptr<Space> space, std::shared_ptr<replay_type> replay, const table_type &table,
        const std::vector<flight_step_record<T>> &steps, const X &initial, const X &arrival,
        std::shared_ptr<const Section> target, T minimum_speed, T replay_tolerance
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing prepared flight space" ) ),
          replay_( std::move( replay ) ), table_( table ), steps_( steps ), target_( std::move( target ) ),
          initial_( *space_ ), time_rate_( *space_ ), endpoint_( *space_ ), pending_( *space_ ), work_( space_ )
    {
        if ( !replay_ || !target_ || table.type() != table_type::scheme_type::explicit_rk ||
             !table.has_dense_output() || steps.empty() || !traits::is_finite( minimum_speed ) ||
             minimum_speed <= T( 0 ) || !traits::is_finite( replay_tolerance ) || replay_tolerance <= T( 0 ) )
            throw std::invalid_argument(
                "Prepared sections require an explicit RK dense table and transverse identity-state flights"
            );
        if ( space_->get_size( initial ) != space_->get_vector_size() ||
             space_->get_size( arrival ) != space_->get_vector_size() || !space_->is_valid_number( initial ) ||
             !space_->is_valid_number( arrival ) )
            throw std::invalid_argument( "Invalid prepared flight states" );
        for ( std::size_t i = 0; i < steps.size(); ++i )
        {
            const auto &r = steps[i];
            if ( !traits::is_finite( r.time ) || !traits::is_finite( r.dt ) || !traits::is_finite( r.end ) ||
                 !traits::is_finite( r.theta ) || r.dt == T( 0 ) || r.theta <= T( 0 ) || r.theta > T( 1 ) ||
                 ( i && r.time != steps[i - 1].end ) || ( i + 1 < steps.size() && r.theta != T( 1 ) ) ||
                 r.end == r.time || ( r.end > r.time ) != ( r.dt > T( 0 ) ) ||
                 ( r.dt > T( 0 ) ) != ( steps.front().dt > T( 0 ) ) ||
                 traits::abs( r.end - ( r.time + r.theta * r.dt ) ) >
                     T( 64 ) * std::numeric_limits<T>::epsilon() *
                         std::max( T( 1 ), std::max( traits::abs( r.time ), traits::abs( r.end ) ) ) )
                throw std::invalid_argument( "Invalid frozen flight schedule" );
        }
        for ( auto *x : { &initial_, &time_rate_, &endpoint_, &pending_ } )
        {
            x->init();
            x->start_use();
        }
        space_->assign( initial, *initial_ );
        replay_->run( table_, steps_, *initial_, nullptr, *endpoint_, nullptr, &*time_rate_ );
        space_->assign_lin_comb( 1, *endpoint_, -1, arrival, *pending_ );
        if ( space_->norm( *pending_ ) > replay_tolerance * ( T( 1 ) + space_->norm( arrival ) ) ||
             !space_->is_valid_number( *time_rate_ ) )
            throw std::runtime_error( "Frozen replay disagrees with the prepared section flight" );
        denominator_ = target_->bind( work_ ).state_derivative( steps_.back().end, *endpoint_, *time_rate_ );
        if ( !traits::is_finite( denominator_ ) || traits::abs( denominator_ ) <= minimum_speed )
            throw std::runtime_error( "Grazing dense-output derivative" );
    }

    void apply( const X &in, X &out ) const
    {
        if ( space_->get_size( in ) != space_->get_vector_size() ||
             space_->get_size( out ) != space_->get_vector_size() )
            throw std::invalid_argument( "Prepared flight direction shape mismatch" );
        if ( !space_->is_valid_number( in ) )
            throw std::runtime_error( "Nonfinite flight direction" );
        replay_->run( table_, steps_, *initial_, &in, *endpoint_, &*pending_, nullptr );
        const T dt =
            -target_->bind( work_ ).state_derivative( steps_.back().end, *endpoint_, *pending_ ) / denominator_;
        space_->add_lin_comb( dt, *time_rate_, *pending_ );
        if ( !traits::is_finite( dt ) || !space_->is_valid_number( *pending_ ) )
            throw std::runtime_error( "Nonfinite section derivative" );
        space_->assign( *pending_, out );
        last_time_increment_ = dt;
    }
    T last_time_increment() const
    {
        return last_time_increment_;
    }
    const auto &records() const
    {
        return steps_;
    }

private:
    std::shared_ptr<Space>                         space_;
    std::shared_ptr<replay_type>                   replay_;
    table_type                                     table_;
    std::vector<flight_step_record<T>>             steps_;
    std::shared_ptr<const Section>                 target_;
    wrap                                           initial_, time_rate_;
    mutable wrap                                   endpoint_, pending_;
    mutable typename Section::evaluation_workspace work_;
    T                                              denominator_{};
    mutable T                                      last_time_increment_{};
};
}
#endif
