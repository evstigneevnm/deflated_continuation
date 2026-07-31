#ifndef __DEFLATION_SEED_REGISTRY_H__
#define __DEFLATION_SEED_REGISTRY_H__

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>

namespace container
{

template<class T>
class deflation_seed_registry
{
public:
    struct entry
    {
        T parameter = T(0);
        std::uint64_t next_seed = 0;
    };

    explicit deflation_seed_registry(std::string file_name_):
        file_name(std::move(file_name_))
    {
        load();
    }

    std::uint64_t next(const T parameter) const
    {
        for(const auto& item: entries)
        {
            if(same_parameter(item.parameter, parameter))
            {
                return item.next_seed;
            }
        }
        return 0;
    }

    void advance(
        const T parameter,
        const std::uint64_t count)
    {
        for(auto& item: entries)
        {
            if(same_parameter(item.parameter, parameter))
            {
                item.next_seed += count;
                return;
            }
        }
        entries.push_back(entry{parameter, count});
    }

    void save() const
    {
        if(file_name.empty())
        {
            return;
        }
        nlohmann::json root;
        root["version"] = 1;
        root["entries"] = nlohmann::json::array();
        for(const auto& item: entries)
        {
            root["entries"].push_back(
                {
                    {"parameter", static_cast<double>(item.parameter)},
                    {"next_seed", item.next_seed}
                });
        }
        std::ofstream stream(file_name.c_str());
        if(stream)
        {
            stream << root.dump(4) << '\n';
        }
    }

    const std::vector<entry>& all() const
    {
        return entries;
    }

private:
    static T abs_value(const T value)
    {
        return value < T(0) ? -value : value;
    }

    static bool same_parameter(
        const T left,
        const T right)
    {
        const T scale = std::max<T>(
            T(1),
            std::max<T>(
                abs_value(left),
                abs_value(right)));
        return abs_value(left - right) <=
            T(64)*std::numeric_limits<T>::epsilon()*scale;
    }

    void load()
    {
        entries.clear();
        if(file_name.empty())
        {
            return;
        }
        std::ifstream stream(file_name.c_str());
        if(!stream)
        {
            return;
        }

        nlohmann::json root;
        stream >> root;
        const auto stored =
            root.value("entries", nlohmann::json::array());
        for(const auto& item: stored)
        {
            entry parsed;
            parsed.parameter = static_cast<T>(
                item.value("parameter", 0.0));
            parsed.next_seed =
                item.value("next_seed", std::uint64_t(0));
            entries.push_back(parsed);
        }
    }

    std::string file_name;
    std::vector<entry> entries;
};

} // namespace container

#endif // __DEFLATION_SEED_REGISTRY_H__
