#ifndef PERIODIC_ORBIT_SECTION_MANAGER_H
#define PERIODIC_ORBIT_SECTION_MANAGER_H

#include <type_traits>
#include <nmfd/time_steppers/detail/status.h>
#include <periodic_orbit/events/event_locator.h>

namespace periodic_orbit
{
// One manager owns one flight's arming state and scratch. Geometry stays immutable.
template <
    class Space, class Stepper, class Event, class StatePolicy = identity_state_policy<Space>,
    class RateProvider = no_section_rate>
class section_manager
{
public:
    using T                  = typename Space::norm_type;
    using X                  = typename Space::vector_type;
    using traits             = scfd::utils::scalar_traits<T>;
    using request_type       = flight_request<T, Event>;
    using integration_status = nmfd::time_steppers::integration_status;
    using storage_type       = nmfd::detail::vector_wrap<Space, false, false>;

    section_manager(
        std::shared_ptr<Space> space, std::shared_ptr<Stepper> step, std::shared_ptr<StatePolicy> policy,
        const event_params<T> &params = {}, std::shared_ptr<RateProvider> rate = nullptr
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing section-map space" ) ),
          step_( step ? std::move( step ) : throw std::invalid_argument( "Missing section-map stepper" ) ),
          policy_( policy ? std::move( policy ) : throw std::invalid_argument( "Missing state policy" ) ),
          rate_( std::move( rate ) ), params_( params ), work_( space_ ), locator_( space_, params ),
          sample_( *space_ ), located_( *space_ ), rate_buffer_( *space_ )
    {
        sample_.init();
        sample_.start_use();
        located_.init();
        located_.start_use();
        if constexpr ( !std::is_same_v<RateProvider, no_section_rate> )
        {
            if ( !rate_ )
                throw std::invalid_argument( "Missing section rate provider" );
            rate_buffer_.init();
            rate_buffer_.start_use();
        }
    }

    void reset( const request_type &request )
    {
        if ( !request.target || !traits::is_finite( request.start_time ) || !traits::is_finite( request.end_time ) ||
             !traits::is_finite( request.end_time - request.start_time ) || request.start_time == request.end_time )
            throw std::invalid_argument( "Invalid section flight interval/target" );
        request.rule.validate( params_.value_tolerance );
        if constexpr ( std::is_same_v<RateProvider, no_section_rate> )
            if ( request.rule.minimum_crossing_speed > T( 0 ) )
                throw std::invalid_argument( "Crossing-speed validation requires a rate provider" );
        request_ = request;
        report_  = {};
        status_  = integration_status::running;
        armed_   = false;
        end_     = nullptr;
    }

    void set_time_interval( const T &start, T &end )
    {
        start_ = start;
        end_   = &end;
    }

    // The returned bool denotes endpoint modification, not event success.
    bool apply( const X &, X &candidate )
    {
        if ( !end_ || !request_.target )
            throw std::logic_error( "Section manager was not prepared" );
        T &endpoint      = *end_;
        end_             = nullptr;
        const T    dt    = endpoint - start_;
        const auto dense = step_->get_continuous_integration();
        auto       event = request_.target->bind( work_ );
        T          previous{};
        if ( !sample( dense, event, T( 0 ), previous ) )
            return false;
        if ( traits::abs( previous ) > request_.rule.arming_band )
            armed_ = true;
        for ( std::size_t j = 0; j < params_.subdivisions; ++j )
        {
            const T lower = T( j ) / T( params_.subdivisions ), upper = T( j + 1 ) / T( params_.subdivisions );
            T       value{};
            if ( !sample( dense, event, upper, value ) )
                return false;
            if ( armed_ && crosses( previous, value, dt ) )
            {
                report_ = locator_.locate(
                    dense, event_bracket<T>{ start_, dt, lower, upper, previous, value }, event, *policy_, *located_
                );
                if ( !report_.succeeded() )
                {
                    status_ = integration_status::external_operation_failure;
                    return false;
                }
                const T elapsed = traits::abs( report_.time - request_.start_time );
                // A root at the initial endpoint cannot be committed as a zero-length step.
                const bool advances = dt > T( 0 ) ? report_.time > start_ : report_.time < start_;
                if ( elapsed >= request_.rule.minimum_flight_time && advances )
                {
                    if constexpr ( !std::is_same_v<RateProvider, no_section_rate> )
                    {
                        if ( !rate_->evaluate( report_.time, *located_, *rate_buffer_ ) ||
                             !space_->is_valid_number( *rate_buffer_ ) )
                            return fail( section_status::rate_failure );
                        report_.crossing_speed = event.state_derivative( report_.time, *located_, *rate_buffer_ );
                        if ( !traits::is_finite( report_.crossing_speed ) )
                            return fail( section_status::nonfinite_event );
                        if ( traits::abs( report_.crossing_speed ) <= request_.rule.minimum_crossing_speed ||
                             !oriented_speed( report_.crossing_speed ) )
                            return fail( section_status::grazing );
                    }
                    space_->assign( *located_, candidate );
                    endpoint = report_.time;
                    status_  = integration_status::stopped_by_external_operation;
                    return true;
                }
                report_.status = section_status::not_found;
            }
            if ( !armed_ && traits::abs( value ) > request_.rule.arming_band )
                armed_ = true;
            previous = value;
        }
        return false;
    }

    integration_status get_status() const
    {
        return status_;
    }
    const event_report<T> &report() const
    {
        return report_;
    }

private:
    bool fail( section_status reason )
    {
        report_.status = reason;
        status_        = integration_status::external_operation_failure;
        return false;
    }
    bool oriented_speed( T speed ) const
    {
        return request_.rule.direction == crossing_direction::either ||
               ( request_.rule.direction == crossing_direction::increasing ? speed > T( 0 ) : speed < T( 0 ) );
    }
    bool crosses( T from, T to, T dt ) const
    {
        if ( dt < T( 0 ) )
            std::swap( from, to );
        const bool up   = ( from < T( 0 ) && to >= T( 0 ) ) || ( from <= T( 0 ) && to > T( 0 ) );
        const bool down = ( from > T( 0 ) && to <= T( 0 ) ) || ( from >= T( 0 ) && to < T( 0 ) );
        return request_.rule.direction == crossing_direction::either       ? up || down
               : request_.rule.direction == crossing_direction::increasing ? up
                                                                           : down;
    }
    template <class Dense, class Evaluator>
    bool sample( const Dense &dense, Evaluator &event, T theta, T &value )
    {
        const T time = dense.evaluate( theta, *sample_ );
        if ( !traits::is_finite( time ) || !space_->is_valid_number( *sample_ ) )
            return fail( section_status::invalid_state );
        if ( !policy_->reconstruct( time, *sample_ ) )
            return fail( section_status::reconstruction_failure );
        if ( !space_->is_valid_number( *sample_ ) )
            return fail( section_status::invalid_state );
        value = event.value( time, *sample_ );
        return traits::is_finite( value ) || fail( section_status::nonfinite_event );
    }
    std::shared_ptr<Space>               space_;
    std::shared_ptr<Stepper>             step_;
    std::shared_ptr<StatePolicy>         policy_;
    std::shared_ptr<RateProvider>        rate_;
    event_params<T>                      params_;
    typename Event::evaluation_workspace work_;
    event_locator<Space>                 locator_;
    storage_type                         sample_, located_, rate_buffer_;
    request_type                         request_;
    event_report<T>                      report_;
    integration_status                   status_ = integration_status::running;
    bool                                 armed_  = false;
    T                                    start_{};
    T                                   *end_ = nullptr;
};
}
#endif
