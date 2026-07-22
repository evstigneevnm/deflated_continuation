#ifndef __BIFURCATION_DIAGRAM_CURVE_METADATA_IO_H__
#define __BIFURCATION_DIAGRAM_CURVE_METADATA_IO_H__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <containers/bifurcation_diagram/curve_point.h>

namespace container
{

struct curve_metadata_load_result
{
    bool loaded_any = false;
    std::vector<uint64_t> incomplete_segment_ids;
};

template<class T>
bool write_curve_metadata(
    const std::string& file_name,
    const std::vector<complex_values<T>>& points)
{
    std::ofstream file(file_name, std::ofstream::out);
    if(!file)
    {
        return false;
    }

    file << "# index lambda saved id_file_name segment_id semicurve_id forced_store endpoint_reason\n";
    for(std::size_t index = 0; index < points.size(); ++index)
    {
        const auto& point = points[index];
        file << index << " "
             << std::setprecision(16) << point.lambda << " "
             << (point.is_data_avaliable ? 1 : 0) << " "
             << point.id_file_name << " "
             << point.segment_id << " "
             << point.semicurve_id << " "
             << (point.forced_store ? 1 : 0) << " "
             << to_string(point.endpoint_reason) << "\n";
    }
    return static_cast<bool>(file);
}

template<class T>
curve_metadata_load_result load_curve_metadata(
    const std::string& file_name,
    std::vector<complex_values<T>>& points)
{
    curve_metadata_load_result result;
    std::ifstream file(file_name);
    if(!file)
    {
        return result;
    }

    std::string line;
    while(std::getline(file, line))
    {
        if(line.empty() || line.front() == '#')
        {
            continue;
        }

        std::istringstream stream(line);
        std::size_t index = 0;
        T lambda = T(0);
        unsigned int saved = 0;
        uint64_t id_file_name = 0;
        uint64_t segment_id = 0;
        uint64_t semicurve_id = 0;
        unsigned int forced_store = 0;
        std::string endpoint_reason_value;
        stream >> index >> lambda >> saved >> id_file_name >> segment_id >> semicurve_id >> forced_store;
        if(!stream || index >= points.size())
        {
            continue;
        }
        stream >> endpoint_reason_value;

        auto& point = points[index];
        point.point_index = static_cast<uint64_t>(index);
        point.segment_id = segment_id;
        point.semicurve_id = semicurve_id;
        point.forced_store = forced_store != 0;
        point.endpoint_reason = curve_endpoint_reason_from_string(endpoint_reason_value);
        if(is_incomplete_endpoint(point.endpoint_reason) &&
           std::find(
               result.incomplete_segment_ids.begin(),
               result.incomplete_segment_ids.end(),
               segment_id) == result.incomplete_segment_ids.end())
        {
            result.incomplete_segment_ids.push_back(segment_id);
        }
        result.loaded_any = true;
    }
    return result;
}

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_METADATA_IO_H__
