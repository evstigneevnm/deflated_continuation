#ifndef __BIFURCATION_DIAGRAM_CURVE_PROVENANCE_H__
#define __BIFURCATION_DIAGRAM_CURVE_PROVENANCE_H__

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>

namespace container
{

enum class curve_origin
{
    computed,
    analytical
};

inline const char* to_string(const curve_origin origin)
{
    return origin == curve_origin::analytical ? "analytical" : "computed";
}

inline curve_origin curve_origin_from_string(const std::string& value)
{
    return value == "analytical"
        ? curve_origin::analytical
        : curve_origin::computed;
}

struct curve_provenance
{
    curve_origin origin = curve_origin::computed;
    std::uint64_t analytical_branch_id = 0;
    std::string analytical_branch_name;

    bool is_analytical() const
    {
        return origin == curve_origin::analytical;
    }
};

inline bool write_curve_provenance(
    const std::string& file_name,
    const curve_provenance& provenance)
{
    std::ofstream file(file_name, std::ofstream::out);
    if(!file)
    {
        return false;
    }

    file << "# bifurcation curve provenance v1\n"
         << "origin " << to_string(provenance.origin) << "\n"
         << "analytical_branch_id " << provenance.analytical_branch_id << "\n"
         << "analytical_branch_name "
         << std::quoted(provenance.analytical_branch_name) << "\n";
    return static_cast<bool>(file);
}

inline bool load_curve_provenance(
    const std::string& file_name,
    curve_provenance& provenance)
{
    std::ifstream file(file_name);
    if(!file)
    {
        provenance = {};
        return false;
    }

    curve_provenance loaded;
    std::string key;
    while(file >> key)
    {
        if(!key.empty() && key.front() == '#')
        {
            std::string ignored;
            std::getline(file, ignored);
            continue;
        }
        if(key == "origin")
        {
            std::string value;
            file >> value;
            loaded.origin = curve_origin_from_string(value);
        }
        else if(key == "analytical_branch_id")
        {
            file >> loaded.analytical_branch_id;
        }
        else if(key == "analytical_branch_name")
        {
            file >> std::quoted(loaded.analytical_branch_name);
        }
        else
        {
            std::string ignored;
            std::getline(file, ignored);
        }
        if(!file)
        {
            provenance = {};
            return false;
        }
    }

    provenance = std::move(loaded);
    return true;
}

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_PROVENANCE_H__
