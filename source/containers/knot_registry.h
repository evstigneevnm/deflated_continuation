#ifndef __KNOT_REGISTRY_H__
#define __KNOT_REGISTRY_H__

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>

#include <containers/intersection_status.h>

namespace container
{

template<class T>
class knot_registry
{
public:
    struct entry
    {
        T requested = T(0);
        T effective = T(0);
        std::string reason;
        intersection_status status;
    };

    explicit knot_registry(std::string file_name_):
        file_name(std::move(file_name_))
    {
        load();
    }

    bool find(const T requested, entry& found) const
    {
        for(const auto& item: entries)
        {
            if(same_lambda(item.requested, requested))
            {
                found = item;
                return true;
            }
        }
        return false;
    }

    bool resolve(const T requested, T& effective) const
    {
        entry found;
        if(find(requested, found))
        {
            effective = found.effective;
            return true;
        }
        return false;
    }

    void set(
        const T requested,
        const T effective,
        const std::string& reason,
        const intersection_status& status)
    {
        for(auto& item: entries)
        {
            if(same_lambda(item.requested, requested))
            {
                item.effective = effective;
                item.reason = reason;
                item.status = status;
                return;
            }
        }
        entries.push_back(entry{requested, effective, reason, status});
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
                    {"requested", static_cast<double>(item.requested)},
                    {"effective", static_cast<double>(item.effective)},
                    {"reason", item.reason},
                    {"status",
                        {
                            {"added", item.status.added},
                            {"failed", item.status.failed},
                            {"missing_data", item.status.missing_data},
                            {"skipped_discontinuous", item.status.skipped_discontinuous},
                            {"skipped_incomplete", item.status.skipped_incomplete}
                        }
                    }
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
    std::string file_name;
    std::vector<entry> entries;

    static bool same_lambda(const T a, const T b)
    {
        const T scale = std::max<T>(T(1), std::max<T>(scalar_abs(a), scalar_abs(b)));
        return scalar_abs(a - b) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
    }

    static T scalar_abs(const T value)
    {
        return value < T(0) ? -value : value;
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
        const auto items = root.value("entries", nlohmann::json::array());
        for(const auto& item: items)
        {
            entry parsed;
            parsed.requested = static_cast<T>(item.value("requested", 0.0));
            parsed.effective = static_cast<T>(item.value("effective", static_cast<double>(parsed.requested)));
            parsed.reason = item.value("reason", std::string{});
            const auto status = item.value("status", nlohmann::json::object());
            parsed.status.added = status.value("added", 0u);
            parsed.status.failed = status.value("failed", 0u);
            parsed.status.missing_data = status.value("missing_data", 0u);
            parsed.status.skipped_discontinuous = status.value("skipped_discontinuous", 0u);
            parsed.status.skipped_incomplete = status.value("skipped_incomplete", 0u);
            entries.push_back(parsed);
        }
    }
};

}

#endif
