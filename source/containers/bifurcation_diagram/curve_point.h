#ifndef __BIFURCATION_DIAGRAM_CURVE_POINT_H__
#define __BIFURCATION_DIAGRAM_CURVE_POINT_H__

#include <cstdint>
#include <vector>

#include <containers/curve_endpoint_reason.h>

namespace boost
{
namespace serialization
{
class access;
}
}

namespace container
{

template<class T>
struct complex_values
{
    T lambda;
    bool is_data_avaliable = false;
    std::vector<T> vector_norms;
    uint64_t id_file_name;
    uint64_t point_index = 0;
    uint64_t segment_id = 0;
    uint64_t semicurve_id = 0;
    bool forced_store = false;
    curve_endpoint_reason endpoint_reason = curve_endpoint_reason::none;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& archive, const unsigned int)
    {
        // Keep this legacy archive order stable. Newer fields live in metadata_curve.dat.
        archive & lambda;
        archive & is_data_avaliable;
        archive & vector_norms;
        archive & id_file_name;
    }
};

template<class T>
using bifurcation_curve_point = complex_values<T>;

template<class T>
bool starts_new_curve_traversal_segment(
    const complex_values<T>& previous,
    const complex_values<T>& current)
{
    return
        previous.segment_id != current.segment_id ||
        previous.semicurve_id != current.semicurve_id ||
        is_terminal_endpoint(previous.endpoint_reason);
}

template<class T>
const complex_values<T>* following_symmetry_endpoint_within_source_points(
    const std::vector<complex_values<T>>& points,
    const complex_values<T>& current,
    uint64_t maximum_source_point_gap)
{
    if(maximum_source_point_gap == 0)
        return nullptr;

    for(const auto& candidate: points)
    {
        if(candidate.point_index <= current.point_index)
            continue;

        const uint64_t source_point_gap =
            candidate.point_index - current.point_index;
        if(source_point_gap > maximum_source_point_gap)
            break;
        if(
            candidate.segment_id != current.segment_id ||
            candidate.semicurve_id != current.semicurve_id)
        {
            break;
        }
        if(
            candidate.endpoint_reason ==
            curve_endpoint_reason::symmetry_intersection)
        {
            return &candidate;
        }
        if(is_terminal_endpoint(candidate.endpoint_reason))
            break;
    }
    return nullptr;
}

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_POINT_H__
