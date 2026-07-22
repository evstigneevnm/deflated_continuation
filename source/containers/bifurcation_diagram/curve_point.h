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

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_POINT_H__
