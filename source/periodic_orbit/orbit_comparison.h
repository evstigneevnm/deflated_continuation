#ifndef PERIODIC_ORBIT_ORBIT_COMPARISON_H
#define PERIODIC_ORBIT_ORBIT_COMPARISON_H

#include <optional>
#include <nmfd/detail/vector_wrap.h>
#include <periodic_orbit/events/event_types.h>

namespace periodic_orbit
{
enum class orbit_relation { inconclusive, same, distinct };
enum class crossing_preparation { empty, invalid_orbit, flight_failed, wrong_return_time, not_closed, ready };

// Ordinary autonomous orbits only. Bind one physical problem, parameter, space,
// and oriented section for the lifetime of the comparison (not one per shooting layout).
template <class Space, class Flight>
class orbit_comparison
{
public:
    using T = typename Space::norm_type;
    using X = typename Space::vector_type;
    using request_type = typename Flight::request_type;
    struct params
    {
        T period_tolerance, closure_tolerance, same_distance, distinct_distance;
    };
    struct preparation_report
    {
        crossing_preparation status = crossing_preparation::empty;
        map_report<T> first, next;
        std::optional<T> closure_error;
    };
    struct comparison_report
    {
        orbit_relation relation = orbit_relation::inconclusive;
        std::optional<T> distance;
    };

private:
    using traits = scfd::utils::scalar_traits<T>;
    using storage = nmfd::detail::vector_wrap<Space, true, true>;
    struct domain
    {
        std::shared_ptr<Space> space;
        T parameter;
        request_type section;
    };

public:
    class crossing
    {
        friend class orbit_comparison;
        explicit crossing( std::shared_ptr<const domain> d ) : domain_( std::move( d ) ), state_( *domain_->space ) {}
        std::shared_ptr<const domain> domain_;
        storage state_;
        T period_{};
        bool ready_ = false;
    public:
        const X &state() const
        {
            if ( !domain_ || !ready_ ) throw std::logic_error( "Unprepared orbit crossing" );
            return *state_;
        }
        T period() const
        {
            if ( !domain_ || !ready_ ) throw std::logic_error( "Unprepared orbit crossing" );
            return period_;
        }
    };

    // Only the target and rule of section are used. Each flight gets its own deadline.
    orbit_comparison( std::shared_ptr<Space> space, Flight &flight, T parameter, request_type section, params p )
        : domain_( std::make_shared<domain>( domain{ std::move( space ), parameter, std::move( section ) } ) ),
          flight_( flight ), p_( p ), scratch_( checked_space() )
    {}

    crossing make_crossing() const { return crossing( domain_ ); }
    T parameter() const { return domain_->parameter; }

    // The anchor is a physical state from an independently verified orbit, not a seed
    // or glued matching vector. Reusing a slot invalidates its previous result.
    preparation_report prepare( const X &anchor, T parameter, T period, crossing &out )
    {
        if ( out.domain_ != domain_ )
            throw std::invalid_argument( "Foreign orbit comparison slot" );
        out.ready_ = false;
        preparation_report r;
        r.status = crossing_preparation::invalid_orbit;
        const T limit = period + p_.period_tolerance;
        if ( parameter != domain_->parameter || !traits::is_finite( period ) ||
             period / T(2) <= p_.period_tolerance || !traits::is_finite( limit ) )
            return r;

        auto request = domain_->section;
        request.start_time = T(0);
        request.end_time = limit;
        r.status = crossing_preparation::flight_failed;
        r.first = flight_.evaluate( anchor, request, *out.state_ );
        if ( r.first.status != section_status::located ) return r;
        // Start from the hit and leave the section. Thus a start/end hit is
        // counted once, regardless of the phase or arming state at the anchor.
        request.start_time = r.first.time;
        request.end_time = request.start_time + limit;
        if ( !traits::is_finite( request.end_time ) ) return r;
        r.next = flight_.evaluate( *out.state_, request, *scratch_ );
        if ( r.next.status != section_status::located ) return r;
        if ( traits::abs( r.next.flight_time - period ) > p_.period_tolerance )
        {
            r.status = crossing_preparation::wrong_return_time;
            return r;
        }
        domain_->space->add_lin_comb( T(-1), *out.state_, *scratch_ );
        r.closure_error = domain_->space->norm( *scratch_ );
        if ( !traits::is_finite( *r.closure_error ) || *r.closure_error > p_.closure_tolerance )
        {
            r.status = crossing_preparation::not_closed;
            return r;
        }
        out.period_ = period;
        out.ready_ = true;
        r.status = crossing_preparation::ready;
        return r;
    }

    // A custom distance must be symmetric, use the configured units, and only
    // quotient actions preserving this oriented section. Nonfinite means unavailable.
    template <class Distance>
    comparison_report compare( const crossing &a, const crossing &b, Distance &&distance ) const
    {
        if ( a.domain_ != domain_ || b.domain_ != domain_ || !a.ready_ || !b.ready_ ||
             traits::abs( a.period_ - b.period_ ) > p_.period_tolerance )
            return {};
        const T d = distance( *a.state_, *b.state_ );
        if ( !traits::is_finite( d ) || d < T(0) ) return {};
        return { d <= p_.same_distance ? orbit_relation::same :
                 d >= p_.distinct_distance ? orbit_relation::distinct : orbit_relation::inconclusive, d };
    }

    comparison_report compare( const crossing &a, const crossing &b )
    {
        return compare( a, b, [this]( const X &x, const X &y ) { return physical_distance( x, y ); } );
    }

private:
    Space &checked_space() const
    {
        const auto &r = domain_->section.rule;
        if ( !domain_->space || !domain_->section.target || !traits::is_finite( domain_->parameter ) ||
             r.direction == crossing_direction::either || r.initial != initial_event_policy::leave_and_return ||
             r.minimum_flight_time != T(0) || !traits::is_finite( r.minimum_crossing_speed ) ||
             r.minimum_crossing_speed <= T(0) || !traits::is_finite( p_.period_tolerance ) ||
             p_.period_tolerance <= T(0) || !traits::is_finite( p_.closure_tolerance ) ||
             p_.closure_tolerance <= T(0) || !traits::is_finite( p_.same_distance ) || p_.same_distance < T(0) ||
             !traits::is_finite( p_.distinct_distance ) || p_.distinct_distance <= p_.same_distance )
            throw std::invalid_argument( "Invalid orbit comparison domain or tolerances" );
        return *domain_->space;
    }
    T physical_distance( const X &x, const X &y )
    {
        domain_->space->assign_lin_comb( T(1), y, T(-1), x, *scratch_ );
        return domain_->space->norm( *scratch_ );
    }
    std::shared_ptr<const domain> domain_;
    Flight &flight_;
    params p_;
    storage scratch_;
};
}
#endif
