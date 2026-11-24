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
            const auto admission = roots_.admit_verified( anchor, roots_.parameter(), *period );
            a.root_index = admission.index;
            a.crossing = admission.preparation.status;
            if ( admission.decision != orbit_admission::inserted )
            {
                a.outcome = admission.decision == orbit_admission::duplicate ? seed_outcome::duplicate
                                                                            : seed_outcome::inconclusive;
                continue;
            }
            a.activation = chart_.append( roots_.at( *admission.index ), rechart );
            if ( a.activation != chart_root_status::ready )
            {
                a.outcome = seed_outcome::activation_failed;
                break; // Retain the verified root for diagnosis, never solve with stale exclusions.
            }
            a.outcome = seed_outcome::activated;
        }
        return attempts;
    }
private:
    void require_synchronized() const
    {
        if ( roots_.size() != active_count() )
            throw std::logic_error( "Cannot run discovery with an inactive verified orbit" );
    }
    orbit_collection<Comparison> roots_;
    chart_root_set<Space, Orbit> chart_;
};
}
#endif
