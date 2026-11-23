#ifndef PERIODIC_ORBIT_ORBIT_COLLECTION_H
#define PERIODIC_ORBIT_ORBIT_COLLECTION_H

#include <vector>
#include <periodic_orbit/orbit_comparison.h>

namespace periodic_orbit
{
enum class orbit_admission { inserted, duplicate, inconclusive };

// Append-only, at one parameter. Indices remain valid across admissions.
template <class Comparison>
class orbit_collection
{
public:
    using T = typename Comparison::T;
    using X = typename Comparison::X;
    using crossing = typename Comparison::crossing;
    struct report
    {
        orbit_admission decision = orbit_admission::inconclusive;
        typename Comparison::preparation_report preparation;
        std::optional<std::size_t> index;
    };

    explicit orbit_collection( Comparison &cmp ) : cmp_( cmp ), pending_( new_slot() ) {}
    T parameter() const { return cmp_.parameter(); }
    std::size_t size() const { return entries_.size(); }
    const crossing &at( std::size_t i ) const { return *entries_.at( i ); }

    // The caller has independently verified the anchor (matching and physical closure).
    report admit_verified( const X &anchor, T parameter, T period )
    {
        report r;
        r.preparation = cmp_.prepare( anchor, parameter, period, *pending_ );
        if ( r.preparation.status != crossing_preparation::ready ) return r;
        for ( std::size_t i = 0; i < size(); ++i )
        {
            const auto relation = cmp_.compare( *pending_, at( i ) ).relation;
            if ( relation == orbit_relation::same )
            {
                r.decision = orbit_admission::duplicate;
                r.index = i;
                return r;
            }
            if ( relation == orbit_relation::inconclusive && !r.index ) r.index = i;
        }
        if ( r.index ) return r;
        auto next = new_slot();
        r.index = size();
        entries_.push_back( std::move( pending_ ) );
        pending_ = std::move( next );
        r.decision = orbit_admission::inserted;
        return r;
    }
private:
    std::unique_ptr<crossing> new_slot() const { return std::make_unique<crossing>( cmp_.make_crossing() ); }
    Comparison &cmp_;
    std::unique_ptr<crossing> pending_;
    std::vector<std::unique_ptr<crossing>> entries_;
};
}
#endif
