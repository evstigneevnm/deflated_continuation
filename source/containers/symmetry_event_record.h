#ifndef __CONTAINERS_SYMMETRY_EVENT_RECORD_H__
#define __CONTAINERS_SYMMETRY_EVENT_RECORD_H__

#include <cstddef>
#include <cstdint>

#include <symmetry/translation/orbit_type.h>

namespace container
{

template<class T>
struct symmetry_event_record
{
    int curve_number = -1;
    std::uint64_t point_index = 0;
    T lambda = T(0);
    bool vector_available = false;
    std::uint64_t vector_file_id = 0;
    std::uint64_t segment_id = 0;
    std::uint64_t semicurve_id = 0;
    std::size_t previous_order = 1;
    std::size_t candidate_order = 1;
    symmetry::translation::orbit_type previous_orbit_type;
    symmetry::translation::orbit_type candidate_orbit_type;
    T previous_transverse_ratio = T(0);
    T candidate_transverse_ratio = T(0);
    unsigned int refinements = 0;
};

} // namespace container

#endif
