#ifndef PERIODIC_ORBIT_AUTOMATIC_LAYOUT_H
#define PERIODIC_ORBIT_AUTOMATIC_LAYOUT_H

#include <algorithm>
#include <limits>
#include <periodic_orbit/sections/affine_section.h>
#include <periodic_orbit/sections/section_layout.h>
#include <periodic_orbit/section_map.h>

namespace periodic_orbit
{
// Pilot observation never changes pending endpoints or the adaptive step sequence.
// Only scalar step times and bounded selected backend states are retained.
template <
    class Space, class Stepper, class RateProvider, class StatePolicy = identity_state_policy<Space>,
    class Metric = euclidean_section_metric<Space>>
class automatic_layout
{
public:
    using T            = typename Space::norm_type;
    using X            = typename Space::vector_type;
    using traits       = scfd::utils::scalar_traits<T>;
    using section_type = affine_section<Space, Metric>;
    using layout_type  = section_layout<section_type>;
    using storage_type = nmfd::detail::vector_wrap<Space, false, false>;
    using map_type     = section_map<Space, Stepper, section_type, StatePolicy, RateProvider>;

    automatic_layout(
        std::shared_ptr<Space> space, std::shared_ptr<Stepper> step, std::shared_ptr<RateProvider> rate,
        std::shared_ptr<StatePolicy> policy, std::shared_ptr<Metric> metric, const event_params<T> &event = {}
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing layout space" ) ),
          step_( step ? std::move( step ) : throw std::invalid_argument( "Missing pilot stepper" ) ),
          rate_( rate ? std::move( rate ) : throw std::invalid_argument( "Missing pilot rate" ) ),
          policy_( policy ? std::move( policy ) : throw std::invalid_argument( "Missing pilot state policy" ) ),
          metric_( metric ? std::move( metric ) : throw std::invalid_argument( "Missing pilot metric" ) ),
          event_( event ), trajectory_( *space_ ), rate_buffer_( *space_ )
    {
        event_.validate();
        if ( !step_->dense_output_order() )
            throw std::invalid_argument( "Pilot sampling requires dense output" );
        trajectory_.init();
        trajectory_.start_use();
        rate_buffer_.init();
        rate_buffer_.start_use();
    }

    layout_result<section_type>
    build( const X &initial, const pilot_request<T, section_type> &request, const placement_params<T> &p )
    {
        validate( request, p );
        layout_result<section_type> result;
        if ( space_->get_size( initial ) != space_->get_vector_size() )
        {
            result.status = section_status::invalid_input;
            return result;
        }
        if ( !space_->is_valid_number( initial ) || !policy_->validate_initial( request.start_time, initial ) )
        {
            result.status = section_status::invalid_state;
            return result;
        }
        auto observer    = std::make_shared<pilot_observer>( space_, step_, policy_, p.maximum_pilot_steps );
        using integrator = nmfd::time_steppers::integration::time_integrator<Space, Stepper, pilot_observer>;
        integrator pilot( space_, step_, typename integrator::params{ p.maximum_pilot_steps }, observer );
        pilot.set_time_interval( request.start_time, request.end_time );
        try
        {
            pilot.apply( initial, *trajectory_ );
        }
        catch ( ... )
        {
            step_->reset();
            throw;
        }
        if ( pilot.get_status() != nmfd::time_steppers::integration_status::completed )
        {
            result.status = section_status::step_failure;
            return result;
        }
        const std::size_t count = observer->times.size() - 1;
        result.pilot_steps      = count;
        if ( p.sections > count / p.minimum_steps_per_segment )
        {
            result.status = section_status::insufficient_steps;
            return result;
        }

        std::vector<std::size_t> quantiles;
        std::size_t              middle = 0, remainder = 0;
        for ( std::size_t i = 0; i < p.sections; ++i )
        {
            quantiles.push_back( middle );
            middle += count / p.sections;
            const auto extra = count % p.sections;
            if ( remainder >= p.sections - extra )
            {
                remainder -= p.sections - extra;
                ++middle;
            }
            else
                remainder += extra;
        }
        std::vector<std::size_t> indices;
        for ( std::size_t i = 0; i < p.sections; ++i )
        {
            const auto middle = quantiles[i];
            const auto lo     = i ? middle - std::min( middle, p.neighbor_steps ) : 0;
            const auto hi     = i ? middle + std::min( count - 1 - middle, p.neighbor_steps ) : 0;
            for ( auto index = lo;; ++index )
            {
                indices.push_back( index );
                if ( index == hi )
                    break;
            }
        }
        std::sort( indices.begin(), indices.end() );
        indices.erase( std::unique( indices.begin(), indices.end() ), indices.end() );
        observer->prepare_samples( indices );
        // A second deterministic pass replays the same intervals; it does not clip to samples.
        try
        {
            pilot.apply( initial, *trajectory_ );
        }
        catch ( ... )
        {
            step_->reset();
            throw;
        }
        if ( pilot.get_status() != nmfd::time_steppers::integration_status::completed ||
             observer->next != indices.size() )
        {
            result.status = observer->failure;
            return result;
        }

        std::vector<typename layout_type::node> nodes;
        nodes.reserve( p.sections );
        for ( std::size_t i = 0; i < p.sections; ++i )
        {
            result.failed_section           = i;
            const auto               middle = quantiles[i];
            std::vector<std::size_t> options;
            const auto               lo = i ? middle - std::min( middle, p.neighbor_steps ) : 0;
            const auto               hi = i ? middle + std::min( count - 1 - middle, p.neighbor_steps ) : 0;
            for ( auto candidate = std::lower_bound( indices.begin(), indices.end(), lo );
                  candidate != indices.end() && *candidate <= hi; ++candidate )
                options.push_back( static_cast<std::size_t>( candidate - indices.begin() ) );
            std::stable_sort( options.begin(), options.end(), [&]( auto a, auto b ) {
                return distance( indices[a], middle ) < distance( indices[b], middle );
            } );
            bool selected = false;
            for ( auto j : options )
            {
                const auto index = indices[j];
                if ( ( i && index < nodes.back().pilot_step + p.minimum_steps_per_segment ) ||
                     count - index < ( p.sections - i ) * p.minimum_steps_per_segment )
                    continue;
                const T     time  = observer->times[index];
                const auto &state = *observer->states[j];
                if ( !rate_->evaluate( time, state, *rate_buffer_ ) || !space_->is_valid_number( *rate_buffer_ ) )
                    continue;
                const T speed = metric_->norm( *rate_buffer_ );
                if ( !traits::is_finite( speed ) || speed <= p.minimum_speed )
                    continue;
                nodes.push_back(
                    { std::make_shared<section_type>( space_, state, *rate_buffer_, metric_ ), time, T( 0 ), index, 0 }
                );
                selected = true;
                break;
            }
            if ( !selected )
            {
                result.status = section_status::placement_failure;
                return result;
            }
        }
        const auto                closing = request.closing_target ? request.closing_target : nodes.front().section;
        typename map_type::params map_params;
        map_params.maximum_steps = p.maximum_pilot_steps;
        map_params.event         = event_;
        map_type                                    flight( space_, step_, policy_, map_params, rate_ );
        typename section_type::evaluation_workspace work( space_ );
        for ( std::size_t i = 0; i < nodes.size(); ++i )
        {
            auto      &node      = nodes[i];
            const auto next      = i + 1;
            const T    end       = next < nodes.size() ? nodes[next].pilot_time : request.end_time;
            node.expected_flight = end - node.pilot_time;
            node.expected_steps  = ( next < nodes.size() ? nodes[next].pilot_step : count ) - node.pilot_step;
            const auto                      target = next < nodes.size() ? nodes[next].section : closing;
            flight_request<T, section_type> journey;
            journey.start_time = node.pilot_time;
            journey.end_time   = node.pilot_time + p.flight_limit_factor * node.expected_flight;
            if ( !traits::is_finite( journey.end_time ) )
            {
                result.status = section_status::placement_failure;
                return result;
            }
            journey.target                      = target;
            journey.rule.minimum_crossing_speed = p.minimum_speed;
            result.flight                       = flight.evaluate( node.section->center(), journey, *trajectory_ );
            result.failed_section               = i;
            if ( !result.flight.succeeded() )
            {
                result.status = result.flight.status;
                return result;
            }
            const T error = target->bind( work ).distance( *trajectory_ );
            // Matching geometry alone can mistake a different number of cycles
            // for the pilot itinerary. Convert its spatial allowance to time.
            const T arrival_tolerance = event_.time_tolerance + event_.relative_time_tolerance * traits::abs( end ) +
                                        p.maximum_endpoint_distance / traits::abs( result.flight.crossing_speed );
            if ( !traits::is_finite( error ) || error > p.maximum_endpoint_distance ||
                 traits::abs( result.flight.time - end ) > arrival_tolerance )
            {
                result.status = section_status::placement_failure;
                return result;
            }
        }
        result.layout = std::make_shared<layout_type>( std::move( nodes ), closing );
        result.status = section_status::located;
        return result;
    }

private:
    static std::size_t distance( std::size_t a, std::size_t b )
    {
        return a > b ? a - b : b - a;
    }
    void validate( const pilot_request<T, section_type> &r, const placement_params<T> &p ) const
    {
        if ( !traits::is_finite( r.start_time ) || !traits::is_finite( r.end_time ) || r.start_time == r.end_time ||
             !traits::is_finite( r.end_time - r.start_time ) || !p.sections || !p.minimum_steps_per_segment ||
             !p.maximum_pilot_steps || p.maximum_pilot_steps == std::numeric_limits<std::size_t>::max() ||
             p.neighbor_steps > 32 || !traits::is_finite( p.minimum_speed ) || p.minimum_speed <= T( 0 ) ||
             !traits::is_finite( p.maximum_endpoint_distance ) || p.maximum_endpoint_distance < T( 0 ) ||
             !traits::is_finite( p.flight_limit_factor ) || p.flight_limit_factor <= T( 1 ) )
            throw std::invalid_argument( "Invalid pilot/section placement parameters" );
    }
    class pilot_observer
    {
        using status = nmfd::time_steppers::integration_status;
        std::shared_ptr<Space>       space;
        std::shared_ptr<Stepper>     step;
        std::shared_ptr<StatePolicy> policy;
        std::size_t                  budget, cursor = 0;
        T                            start{}, end{};
        bool                         replay  = false;
        status                       status_ = status::running;
        std::vector<std::size_t>     indices;

    public:
        std::vector<T>            times;
        std::vector<storage_type> states;
        std::size_t               next    = 0;
        section_status            failure = section_status::step_failure;
        pilot_observer(
            std::shared_ptr<Space> s, std::shared_ptr<Stepper> st, std::shared_ptr<StatePolicy> pol, std::size_t b
        )
            : space( std::move( s ) ), step( std::move( st ) ), policy( std::move( pol ) ), budget( b )
        {
        }
        void prepare_samples( const std::vector<std::size_t> &selection )
        {
            indices = selection;
            replay  = true;
            cursor = next = 0;
            status_       = status::running;
            states.reserve( indices.size() );
            for ( std::size_t i = 0; i < indices.size(); ++i )
            {
                states.emplace_back( *space );
                states.back().init();
                states.back().start_use();
            }
        }
        void set_time_interval( const T &a, T &b )
        {
            start = a;
            end   = b;
        }
        bool apply( const X &, X & )
        {
            if ( !replay )
            {
                if ( times.empty() )
                    times.push_back( start );
                if ( cursor == budget )
                {
                    status_ = status::external_operation_failure;
                    return false;
                }
                times.push_back( end );
            }
            else
            {
                if ( cursor + 1 >= times.size() || start != times[cursor] || end != times[cursor + 1] )
                {
                    status_ = status::external_operation_failure;
                    return false;
                }
                while ( next < indices.size() && indices[next] <= cursor + 1 )
                {
                    const T theta = indices[next] == cursor ? T( 0 ) : T( 1 );
                    const T time  = step->get_continuous_integration().evaluate( theta, *states[next] );
                    if ( !space->is_valid_number( *states[next] ) )
                    {
                        failure = section_status::invalid_state;
                        status_ = status::external_operation_failure;
                        return false;
                    }
                    if ( !policy->reconstruct( time, *states[next] ) )
                    {
                        failure = section_status::reconstruction_failure;
                        status_ = status::external_operation_failure;
                        return false;
                    }
                    if ( !space->is_valid_number( *states[next] ) )
                    {
                        failure = section_status::invalid_state;
                        status_ = status::external_operation_failure;
                        return false;
                    }
                    ++next;
                }
            }
            ++cursor;
            return false;
        }
        status get_status() const
        {
            return status_;
        }
    };
    std::shared_ptr<Space>        space_;
    std::shared_ptr<Stepper>      step_;
    std::shared_ptr<RateProvider> rate_;
    std::shared_ptr<StatePolicy>  policy_;
    std::shared_ptr<Metric>       metric_;
    event_params<T>               event_;
    storage_type                  trajectory_, rate_buffer_;
};
}
#endif
