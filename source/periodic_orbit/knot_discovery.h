#ifndef PERIODIC_ORBIT_KNOT_DISCOVERY_H
#define PERIODIC_ORBIT_KNOT_DISCOVERY_H

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <periodic_orbit/matching/chart_deflation.h>
#include <periodic_orbit/orbit_collection.h>
#include <periodic_orbit/verified_knot_samples.h>

namespace periodic_orbit
{
enum class seed_outcome { correction_failed, verification_failed, duplicate, inconclusive, activated, activation_failed };

struct discovery_attempt
{
    std::size_t seed_index, active_before;
    seed_outcome outcome = seed_outcome::correction_failed;
    std::optional<std::size_t> root_index;
    crossing_preparation crossing = crossing_preparation::empty;
    std::optional<chart_root_status> activation;
};

struct orbit_launch
{
    std::size_t root_index;
    chart_root_status preparation;
    bool trace_complete;
};

struct discovery_trace_report
{
    std::vector<discovery_attempt> attempts;
    bool discovery_complete;
    std::vector<orbit_launch> launches;
};

struct knot_admission
{
    orbit_admission                  decision;
    std::optional<std::size_t>       root_index;
    crossing_preparation             crossing;
    std::optional<chart_root_status> activation;
    bool                             active() const
    {
        return decision == orbit_admission::duplicate || activation == chart_root_status::ready;
    }
};

enum class preload_status
{
    ready,
    unresolved_capture,
    wrong_parameter,
    admission_failed
};
struct knot_preload_report
{
    preload_status status;
    struct entry
    {
        knot_sample_source source;
        knot_admission     admission;
    };
    std::vector<entry> entries;
    bool               complete() const
    {
        return status == preload_status::ready;
    }
};

// Serial discovery at one physical parameter and one fixed layout. The supplied
// solvers, independent verifier and rechart flight retain their backend workspace.
template <class Comparison, class Space, class Orbit>
class knot_discovery
{
    using T = typename Space::scalar_type;
public:
    knot_discovery( Comparison &cmp, std::shared_ptr<Space> coords, std::shared_ptr<Orbit> orbit,
                    std::size_t capacity, T matching_tol, T period_tol, T exclusion )
        : roots_( cmp ), chart_( std::move( coords ), std::move( orbit ), cmp.parameter(), capacity,
                                 matching_tol, period_tol, exclusion )
    {}

    const auto &catalog() const { return roots_; }
    std::size_t active_count() const { return chart_.size(); }
    const auto &deflated_operator() const { return chart_.op(); }

    // Import one source trace at a time, before running the target seed batch.
    // A failed import leaves this knot inspectable but unusable for discovery.
    template <class PhysicalSpace, class Rechart>
    knot_preload_report preload_verified( const verified_knot_samples<PhysicalSpace> &samples, Rechart &&rechart )
    {
        require_synchronized();
        preload_blocked_ = true;
        if ( !samples.complete() )
            return { preload_status::unresolved_capture, {} };
        if ( samples.parameter() != roots_.parameter() )
            return { preload_status::wrong_parameter, {} };
        knot_preload_report report{ preload_status::ready, {} };
        for ( std::size_t i = 0; i < samples.size(); ++i )
        {
            const auto &s = samples.at( i );
            // Compare in canonical phase, but reconstruct from the verified sample, not a later return.
            const auto admission = admit( s.state(), s.parameter, s.period(), [&]( const auto & ) {
                return chart_.append( s, rechart );
            } );
            report.entries.push_back( { s.source, admission } );
            if ( !admission.active() )
            {
                report.status = preload_status::admission_failed;
                return report;
            }
        }
        preload_blocked_ = false;
        return report;
    }

    // Corrections write into a common caller-owned candidate. Only on success
    // does verify(anchor) publish a physical anchor and optional verified period.
    template <class Seeds, class Ordinary, class Deflated, class Verify, class Rechart>
    std::vector<discovery_attempt> run( const Seeds &seeds, typename Comparison::X &anchor, Ordinary &&ordinary,
                                        Deflated &&deflated, Verify &&verify, Rechart &&rechart )
    {
        require_synchronized();
        std::vector<discovery_attempt> attempts;
        for ( const auto &seed : seeds )
        {
            require_synchronized();
            attempts.push_back( { attempts.size(), active_count() } );
            auto &a = attempts.back();
            if ( !(a.active_before ? deflated( seed ) : ordinary( seed )) ) continue;
            const auto period = verify( anchor );
            if ( !period )
            {
                a.outcome = seed_outcome::verification_failed;
                continue;
            }
            const auto admission = admit( anchor, roots_.parameter(), *period, [&]( const auto &root ) {
                return chart_.append( root, rechart );
            } );
            a.root_index         = admission.root_index;
            a.crossing           = admission.crossing;
            a.activation         = admission.activation;
            if ( admission.decision != orbit_admission::inserted )
            {
                a.outcome = admission.decision == orbit_admission::duplicate ? seed_outcome::duplicate
                                                                            : seed_outcome::inconclusive;
                continue;
            }
            if ( a.activation != chart_root_status::ready )
            {
                a.outcome = seed_outcome::activation_failed;
                break; // Retain the verified root for diagnosis, never solve with stale exclusions.
            }
            a.outcome = seed_outcome::activated;
        }
        return attempts;
    }

    // Dispatch only this batch's new roots, after discovery has finished.
    // trace(parameter, root_index, coordinates) uses ordinary continuation;
    // false reports an incomplete branch without suppressing other launches.
    template <class Seeds, class Ordinary, class Deflated, class Verify, class Rechart, class Trace>
    discovery_trace_report run_and_continue( const Seeds &seeds, typename Comparison::X &anchor, Ordinary &&ordinary,
                                             Deflated &&deflated, Verify &&verify, Rechart &&rechart, Trace &&trace )
    {
        auto attempts = run( seeds, anchor, ordinary, deflated, verify, rechart );
        discovery_trace_report report{ std::move( attempts ), roots_.size() == active_count(), {} };
        if ( !report.discovery_complete ) return report;
        for ( const auto &a : report.attempts )
        {
            if ( a.outcome != seed_outcome::activated ) continue;
            const auto index = a.root_index.value();
            // Ordinary continuation may replace the state Jacobian underneath the deflated operator.
            chart_.op()->invalidate_linearization();
            bool complete = false;
            const auto status = chart_.with_coordinates( roots_.at( index ), rechart, [&]( const auto &z ) {
                complete = trace( roots_.parameter(), index, z );
            } );
            report.launches.push_back( { index, status, complete } );
        }
        return report;
    }
private:
    template <class Activate>
    knot_admission admit( const typename Comparison::X &anchor, T parameter, T period, Activate activate )
    {
        const auto     r = roots_.admit_verified( anchor, parameter, period );
        knot_admission result{ r.decision, r.index, r.preparation.status, {} };
        if ( r.decision == orbit_admission::inserted )
            result.activation = activate( roots_.at( *r.index ) );
        return result;
    }
    void require_synchronized() const
    {
        if ( preload_blocked_ )
            throw std::logic_error( "Cannot run discovery after incomplete knot preload" );
        if ( roots_.size() != active_count() )
            throw std::logic_error( "Cannot run discovery with an inactive verified orbit" );
    }
    orbit_collection<Comparison> roots_;
    chart_root_set<Space, Orbit> chart_;
    bool                         preload_blocked_ = false;
};
}
#endif
