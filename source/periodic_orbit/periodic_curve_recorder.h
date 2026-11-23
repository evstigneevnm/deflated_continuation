#ifndef PERIODIC_ORBIT_PERIODIC_CURVE_RECORDER_H
#define PERIODIC_ORBIT_PERIODIC_CURVE_RECORDER_H

#include <containers/bifurcation_diagram/periodic_branch_archive.h>
#include <continuation/continuation_result.h>
#include <cstdint>
#include <ostream>
#include <utility>

namespace periodic_orbit
{
struct no_verified_point_action
{
    template <class... Args> void operator()( const Args &... ) const {}
};

// Verify(p,z) borrows the backend coordinates and returns an optional scalar
// measurement. An empty result records a failed verification, not a plot value.
template <class Verify, class OnVerified = no_verified_point_action>
class periodic_curve_recorder
{
    using reason = container::curve_endpoint_reason;
    Verify verify_;
    OnVerified on_verified_;
    nlohmann::json archive_;
    std::uint64_t count_ = 0;
    bool verified_ = true;

public:
    periodic_curve_recorder( const container::branch_identity &id, const std::string &method,
                             std::size_t sections, Verify verify, OnVerified action = {} )
        : verify_( std::move( verify ) ), on_verified_( std::move( action ) ),
          archive_( container::periodic_branch_manifest( id, method, sections ) )
    {}
    void set_restart_source( const std::string &branch, std::uint64_t segment, std::uint64_t index )
    {
        if ( get_current_segment_id() || branch.empty() || !segment )
            throw std::logic_error( "Invalid orbit restart source" );
        archive_["restarted_from"] = { { "branch", branch }, { "segment", segment }, { "source_index", index } };
    }
    void start_new_segment()
    {
        auto &segments = archive_.at( "segments" );
        segments.push_back( { { "id", segments.size() + 1 }, { "points", nlohmann::json::array() } } );
    }
    std::uint64_t get_current_segment_id() const { return archive_.at( "segments" ).size(); }

    template <class State>
    void add( double parameter, const State &z, bool /*force_store*/, reason endpoint )
    {
        if ( !std::isfinite( parameter ) || !get_current_segment_id() )
            throw std::logic_error( "Invalid periodic curve point" );
        auto measurement = verify_( parameter, z );
        const bool ok = measurement && measurement->valid();
        nlohmann::json point = { { "source_index", count_ }, { "parameter", parameter },
                                 { "endpoint", container::to_string( endpoint ) }, { "verified", ok } };
        if ( ok )
        {
            point.update( { { "period", measurement->period }, { "observable_rms", measurement->rms },
                            { "observable_min", measurement->minimum }, { "observable_max", measurement->maximum },
                            { "closure_error", measurement->closure_error } } );
            on_verified_( get_current_segment_id(), count_, parameter, endpoint, z, *measurement );
        }
        archive_.at( "segments" ).back().at( "points" ).push_back( std::move( point ) );
        ++count_;
        verified_ &= ok;
    }
    void set_last_endpoint_reason( reason endpoint )
    {
        auto &points = archive_.at( "segments" ).back().at( "points" );
        if ( !points.empty() )
            points.back()["endpoint"] = container::to_string( endpoint );
    }
    void print_curve() const {} // The driver may request progress output. Publish only after it returns.

    template <class Scalar>
    void write( std::ostream &out, const continuation::continuation_curve_result<Scalar> &result )
    {
        auto &segments = archive_.at( "segments" );
        if ( result.semicurves_started != segments.size() || segments.size() > result.semicurves.size() )
            throw std::logic_error( "Periodic recorder/driver segment mismatch" );
        for ( std::size_t i = 0; i < segments.size(); ++i )
        {
            auto &segment = segments[i];
            const auto &arm = result.semicurves[i];
            if ( arm.segment_id != segment.at( "id" ) || arm.accepted_points != segment.at( "points" ).size() )
                throw std::logic_error( "Periodic recorder/driver point mismatch" );
            segment.update( { { "direction", arm.direction }, { "endpoint", container::to_string( arm.endpoint_reason ) },
                              { "failure", continuation::to_string( arm.failure ) } } );
        }
        archive_["trace_complete"] = result.complete();
        archive_["all_points_verified"] = count_ != 0 && verified_;
        out << archive_ << '\n';
        if ( !out )
            throw std::runtime_error( "Could not write periodic branch archive" );
    }
};
}
#endif
