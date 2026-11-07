#ifndef PERIODIC_ORBIT_PREPARED_SECTION_H
#define PERIODIC_ORBIT_PREPARED_SECTION_H

#include <limits>
#include <periodic_orbit/flows/recording_stepper.h>

namespace periodic_orbit
{
// One checkpoint and a bounded scalar schedule; the supplied replay owns serial
// scratch. Only stationary sections and identity-state reconstruction are implied.
template <
    class Space, class Replay, class Section, class Configuration = typename Replay::table_type,
    class Record = flight_step_record<typename Space::scalar_type>>
class prepared_section
{
public:
    using T           = typename Space::scalar_type;
    using X           = typename Space::vector_type;
    using traits      = scfd::utils::scalar_traits<T>;
    using replay_type = Replay;
    using table_type  = Configuration;
    using wrap        = nmfd::detail::vector_wrap<Space, false, false>;
    prepared_section(
        std::shared_ptr<Space> space, std::shared_ptr<Replay> replay, const table_type &table,
        const std::vector<Record> &steps, const X &initial, const X &arrival, std::shared_ptr<const Section> target,
        T minimum_speed, T replay_tolerance
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing prepared flight space" ) ),
          replay_( std::move( replay ) ), table_( table ), steps_( steps ), target_( std::move( target ) ),
          initial_( *space_ ), time_rate_( *space_ ), endpoint_( *space_ ), pending_( *space_ ), work_( space_ )
    {
        if ( !replay_ || !target_ || !Replay::supports( table ) || steps.empty() ||
             !traits::is_finite( minimum_speed ) || minimum_speed <= T( 0 ) || !traits::is_finite( replay_tolerance ) ||
             replay_tolerance <= T( 0 ) )
            throw std::invalid_argument( "Unsupported prepared section configuration" );
        if ( space_->get_size( initial ) != space_->get_vector_size() ||
             space_->get_size( arrival ) != space_->get_vector_size() || !space_->is_valid_number( initial ) ||
             !space_->is_valid_number( arrival ) )
            throw std::invalid_argument( "Invalid prepared flight states" );
        detail::validate_flight_schedule( steps, std::numeric_limits<std::size_t>::max() );
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
    std::shared_ptr<Replay>                        replay_;
    table_type                                     table_;
    std::vector<Record>                            steps_;
    std::shared_ptr<const Section>                 target_;
    wrap                                           initial_, time_rate_;
    mutable wrap                                   endpoint_, pending_;
    mutable typename Section::evaluation_workspace work_;
    T                                              denominator_{};
    mutable T                                      last_time_increment_{};
};
}
#endif
