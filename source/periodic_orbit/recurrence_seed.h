#ifndef PERIODIC_ORBIT_RECURRENCE_SEED_H
#define PERIODIC_ORBIT_RECURRENCE_SEED_H

#include <optional>
#include <nmfd/detail/vector_wrap.h>
#include <periodic_orbit/events/event_types.h>

namespace periodic_orbit
{
enum class seed_origin { supplied, section_recurrence };

// A trial for an autonomous periodic orbit, not a verified orbit or restart.
// The caller owns the backend state throughout layout construction.
template <class Space>
struct orbit_seed
{
    const typename Space::vector_type &state;
    typename Space::norm_type          parameter, period;
    seed_origin                        origin;
};

template <class Space>
class recurrence_seed_search
{
public:
    using T = typename Space::norm_type;
    using X = typename Space::vector_type;
    struct params
    {
        T minimum_period, maximum_period, distance_tolerance;
        std::size_t maximum_crossings;
    };
    struct report
    {
        std::optional<orbit_seed<Space>> candidate;
        map_report<T>                    last_flight;
        std::size_t                      crossings = 0;
        std::optional<T>                 last_distance;
    };

    explicit recurrence_seed_search( std::shared_ptr<Space> space )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing recurrence space" ) ),
          a_( *space_ ), b_( *space_ )
    {
        a_.init(); a_.start_use();
        b_.init(); b_.start_use();
    }

    // flight is bound to parameter. end_time is a deadline for the whole search,
    // not a new allowance per crossing. Publish to out only on success.
    template <class Flight>
    report find( Flight &flight, typename Flight::request_type request, const X &initial, T parameter,
                 const params &p, X &out )
    {
        using traits = scfd::utils::scalar_traits<T>;
        if ( !traits::is_finite( parameter ) || !traits::is_finite( request.start_time ) ||
             !traits::is_finite( request.end_time ) || !(request.start_time < request.end_time) ||
             !traits::is_finite( p.minimum_period ) || !traits::is_finite( p.maximum_period ) ||
             p.minimum_period <= T(0) || p.maximum_period < p.minimum_period ||
             !traits::is_finite( p.distance_tolerance ) || p.distance_tolerance <= T(0) ||
            !p.maximum_crossings || request.rule.direction == crossing_direction::either ||
             request.rule.initial != initial_event_policy::leave_and_return ||
             request.rule.minimum_flight_time != T(0) || !(request.rule.minimum_crossing_speed > T(0)) ||
             space_->get_size( out ) != space_->get_vector_size() )
            throw std::invalid_argument( "Invalid oriented recurrence search" );

        report result;
        X *previous = &*a_, *current = &*b_;
        while ( result.crossings < p.maximum_crossings && request.start_time < request.end_time )
        {
            result.last_flight = flight.evaluate( result.crossings ? *previous : initial, request, *current );
            if ( result.last_flight.status != section_status::located )
                break;
            const T period = result.last_flight.time - request.start_time;
            if ( !(period > T(0)) || result.last_flight.time > request.end_time )
                throw std::logic_error( "Section flight did not advance within the search interval" );
            ++result.crossings;
            // Compare only consecutive returns, never skip a short flight to manufacture 2T.
            if ( result.crossings >= 2 && period >= p.minimum_period && period <= p.maximum_period )
            {
                space_->add_lin_comb( -1, *current, *previous );
                result.last_distance = space_->norm( *previous );
                if ( traits::is_finite( *result.last_distance ) && *result.last_distance <= p.distance_tolerance )
                {
                    space_->assign( *current, out );
                    result.candidate.emplace( orbit_seed<Space>{ out, parameter, period, seed_origin::section_recurrence } );
                    break;
                }
            }
            request.start_time = result.last_flight.time;
            std::swap( previous, current );
        }
        return result;
    }

private:
    std::shared_ptr<Space>                         space_;
    nmfd::detail::vector_wrap<Space, false, false> a_, b_;
};
}
#endif
