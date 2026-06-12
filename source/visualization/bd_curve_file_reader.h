#ifndef __VISUALIZATION_BD_CURVE_FILE_READER_H__
#define __VISUALIZATION_BD_CURVE_FILE_READER_H__

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace visualization
{

struct bd_curve_row
{
    std::size_t index = 0;
    double lambda = 0.0;
    std::vector<double> norms;
    std::uint64_t file_id = 0;
};

struct bd_curve_metadata_row
{
    std::size_t index = 0;
    double lambda = 0.0;
    bool saved = false;
    std::uint64_t file_id = 0;
    std::uint64_t segment_id = 0;
    std::uint64_t semicurve_id = 0;
    bool forced_store = false;
    std::string endpoint_reason = "none";
};

inline std::vector<bd_curve_row> read_debug_curve_file(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if(!file)
    {
        throw std::runtime_error("visualization: failed to open curve file " + path.string());
    }

    std::vector<bd_curve_row> rows;
    std::string line;
    while(std::getline(file, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");
        if(first == std::string::npos || line[first] == '#')
        {
            continue;
        }

        std::istringstream stream(line);
        std::vector<double> values;
        double value = 0.0;
        while(stream >> value)
        {
            values.push_back(value);
        }
        if(values.size() < 2)
        {
            continue;
        }

        bd_curve_row row;
        row.index = rows.size();
        row.lambda = values.front();
        row.file_id = static_cast<std::uint64_t>(values.back());
        if(values.size() > 2)
        {
            row.norms.assign(values.begin() + 1, values.end() - 1);
        }
        rows.push_back(row);
    }
    return rows;
}

inline std::unordered_map<std::size_t, bd_curve_metadata_row> read_curve_metadata_file(
    const std::filesystem::path& path)
{
    std::unordered_map<std::size_t, bd_curve_metadata_row> rows;
    std::ifstream file(path);
    if(!file)
    {
        return rows;
    }

    std::string line;
    while(std::getline(file, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");
        if(first == std::string::npos || line[first] == '#')
        {
            continue;
        }

        bd_curve_metadata_row row;
        int saved = 0;
        int forced = 0;
        std::istringstream stream(line);
        stream >> row.index
               >> row.lambda
               >> saved
               >> row.file_id
               >> row.segment_id
               >> row.semicurve_id
               >> forced;
        if(!stream)
        {
            continue;
        }
        stream >> row.endpoint_reason;
        row.saved = saved != 0;
        row.forced_store = forced != 0;
        rows[row.index] = row;
    }
    return rows;
}

inline std::vector<std::filesystem::path> numeric_branch_directories(const std::filesystem::path& project_dir)
{
    std::vector<std::filesystem::path> result;
    if(!std::filesystem::is_directory(project_dir))
    {
        throw std::runtime_error("visualization: project directory does not exist: " + project_dir.string());
    }

    for(const auto& entry: std::filesystem::directory_iterator(project_dir))
    {
        if(!entry.is_directory())
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if(name.empty() ||
           !std::all_of(name.begin(), name.end(), [](const unsigned char c){ return c >= '0' && c <= '9'; }))
        {
            continue;
        }
        result.push_back(entry.path());
    }

    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right)
    {
        return std::stoull(left.filename().string()) < std::stoull(right.filename().string());
    });
    return result;
}

} // namespace visualization

#endif
