#ifndef PERIODIC_ORBIT_SECTION_LAYOUT_H
#define PERIODIC_ORBIT_SECTION_LAYOUT_H

#include <vector>
#include <periodic_orbit/events/event_types.h>

namespace periodic_orbit
{
template <class Section>
class section_layout
{
public:
    using T = typename Section::scalar_type;
    struct node
    {
        std::shared_ptr<const Section> section;
        T                              pilot_time{}, expected_flight{};
        std::size_t                    pilot_step = 0, expected_steps = 0;
    };
    section_layout( std::vector<node> nodes, std::shared_ptr<const Section> closing_target )
        : nodes_( std::move( nodes ) ), closing_target_( std::move( closing_target ) )
    {
        if ( nodes_.empty() || !closing_target_ )
            throw std::invalid_argument( "Empty section itinerary" );
        for ( const auto &entry : nodes_ )
            if ( !entry.section || !entry.expected_steps ||
                 !scfd::utils::scalar_traits<T>::is_finite( entry.pilot_time ) ||
                 !scfd::utils::scalar_traits<T>::is_finite( entry.expected_flight ) || entry.expected_flight == T( 0 ) )
                throw std::invalid_argument( "Invalid section itinerary node" );
    }
    std::size_t size() const
    {
        return nodes_.size();
    }
    const node &at( std::size_t i ) const
    {
        return nodes_.at( i );
    }
    const std::shared_ptr<const Section> &closing_target() const
    {
        return closing_target_;
    }

private:
    std::vector<node>              nodes_;
    std::shared_ptr<const Section> closing_target_;
};

template <class T, class Section>
struct pilot_request
{
    T start_time{}, end_time{};
    // A caller-supplied transformed closing plane permits relative-orbit pilot layouts.
    std::shared_ptr<const Section> closing_target;
};

template <class T>
struct placement_params
{
    std::size_t sections                  = 1;
    std::size_t minimum_steps_per_segment = 2;
    std::size_t neighbor_steps            = 2;
    std::size_t maximum_pilot_steps       = 100000;
    T           minimum_speed             = T( 1e-10 );
    T           maximum_endpoint_distance = T( 1e-4 );
    T           flight_limit_factor       = T( 2 );
};

template <class Section>
struct layout_result
{
    section_status                                 status = section_status::idle;
    std::shared_ptr<const section_layout<Section>> layout;
    std::size_t                                    failed_section = 0, pilot_steps = 0;
    map_report<typename Section::scalar_type>      flight;
    bool                                           succeeded() const
    {
        return bool( layout );
    }
};
}
#endif
