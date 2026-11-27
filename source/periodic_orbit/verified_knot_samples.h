#ifndef PERIODIC_ORBIT_VERIFIED_KNOT_SAMPLES_H
#define PERIODIC_ORBIT_VERIFIED_KNOT_SAMPLES_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <nmfd/detail/vector_wrap.h>

namespace periodic_orbit
{
struct knot_sample_source
{
    std::string   branch;
    std::uint64_t segment, point;
};

enum class knot_capture_status
{
    pending,
    complete,
    incomplete_trace,
    missing_hit,
    capacity_reached
};

// One branch, one exact observation parameter. Only verified hits own a vector.
template <class Space>
class verified_knot_samples
{
    using X       = typename Space::vector_type;
    using storage = nmfd::detail::vector_wrap<Space, true, true>;

public:
    struct sample
    {
        const knot_sample_source source;
        const double             parameter;
        double                   period() const
        {
            return period_;
        }
        const X &state() const
        {
            return *anchor_;
        }
        sample( Space &space, knot_sample_source id, double p, double t, const X &x )
            : source( std::move( id ) ), parameter( p ), period_( t ), anchor_( space )
        {
            space.assign( x, *anchor_ );
        }

    private:
        double  period_;
        storage anchor_;
    };

    verified_knot_samples( std::shared_ptr<Space> space, double target, std::size_t capacity )
        : space_( std::move( space ) ), target_( target ), capacity_( capacity )
    {
        if ( !space_ || !std::isfinite( target ) || !capacity )
            throw std::invalid_argument( "Invalid knot capture settings" );
    }
    verified_knot_samples( verified_knot_samples && ) = default;
    // Each backend buffer borrows its owning space until destruction.
    verified_knot_samples &operator=( verified_knot_samples && ) = delete;
    double                 parameter() const
    {
        return target_;
    }
    std::size_t size() const
    {
        return hits_.size();
    }
    const sample &at( std::size_t i ) const
    {
        return *hits_.at( i );
    }
    knot_capture_status status() const
    {
        return space_ ? status_ : knot_capture_status::pending;
    }
    bool complete() const
    {
        return status() == knot_capture_status::complete;
    }

    void begin( const std::string &branch )
    {
        if ( !space_ || branch.empty() || !branch_.empty() )
            throw std::logic_error( "Knot capture requires a new trace" );
        branch_ = branch;
    }
    void on_verified( std::uint64_t segment, std::uint64_t point, double p, double period, const X &anchor )
    {
        if ( p != target_ )
            return;
        if ( branch_.empty() || finished_ )
            throw std::logic_error( "Knot capture outside trace" );
        if ( hits_.size() == capacity_ )
        {
            status_ = knot_capture_status::capacity_reached;
            return;
        }
        hits_.push_back(
            std::make_unique<sample>( *space_, knot_sample_source{ branch_, segment, point }, p, period, anchor )
        );
    }

    // Reconcile callbacks with the scalar recorder, never bridge two arms or sort a fold.
    void finish( const nlohmann::json &archive )
    {
        if ( finished_ || branch_.empty() || archive.at( "id" ) != branch_ )
            throw std::logic_error( "Mismatched knot capture trace" );
        finished_ = true;
        if ( status_ != knot_capture_status::pending )
            return;
        if ( !archive.at( "trace_complete" ).get<bool>() )
        {
            status_ = knot_capture_status::incomplete_trace;
            return;
        }
        std::size_t next = 0;
        status_          = knot_capture_status::missing_hit;
        for ( const auto &arm : archive.at( "segments" ) )
        {
            std::optional<double> previous;
            for ( const auto &point : arm.at( "points" ) )
            {
                const double p = point.at( "parameter" );
                if ( previous && ( ( *previous < target_ && target_ < p ) || ( p < target_ && target_ < *previous ) ) )
                    return;
                if ( p == target_ )
                {
                    if ( !point.at( "verified" ).get<bool>() || next == size() )
                        return;
                    const auto &hit = at( next++ );
                    if ( arm.at( "id" ) != hit.source.segment || point.at( "source_index" ) != hit.source.point ||
                         point.at( "period" ) != hit.period() )
                        return;
                }
                previous = p;
            }
        }
        if ( next == size() )
            status_ = knot_capture_status::complete;
    }

private:
    std::shared_ptr<Space>               space_;
    double                               target_;
    std::size_t                          capacity_;
    std::string                          branch_;
    std::vector<std::unique_ptr<sample>> hits_;
    knot_capture_status                  status_   = knot_capture_status::pending;
    bool                                 finished_ = false;
};
}
#endif
