#ifndef __CONTAINERS_SYMMETRY_EVENT_REGISTRY_H__
#define __CONTAINERS_SYMMETRY_EVENT_REGISTRY_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>

#include <containers/symmetry_event_record.h>

namespace container
{

template<class T>
class symmetry_event_registry
{
public:
    using record_type = symmetry_event_record<T>;

    struct event_node
    {
        std::string id;
        T lambda = T(0);
        T lambda_min = T(0);
        T lambda_max = T(0);
        symmetry::translation::orbit_type previous_orbit_type;
        symmetry::translation::orbit_type candidate_orbit_type;
        record_type state_reference;
        std::vector<record_type> incidents;
    };

    explicit symmetry_event_registry(std::filesystem::path file_name_ = {}):
        file_name(std::move(file_name_))
    {
    }

    template<class StateDistance>
    void rebuild(
        std::vector<record_type> records,
        const T lambda_tolerance,
        const T state_tolerance,
        StateDistance&& state_distance)
    {
        std::sort(
            records.begin(),
            records.end(),
            [](const record_type& left, const record_type& right)
            {
                if(left.curve_number != right.curve_number)
                {
                    return left.curve_number < right.curve_number;
                }
                return left.point_index < right.point_index;
            });

        events.clear();
        for(const auto& record: records)
        {
            event_node* match = nullptr;
            for(auto& event: events)
            {
                if(event.previous_orbit_type != record.previous_orbit_type ||
                   event.candidate_orbit_type != record.candidate_orbit_type ||
                   scalar_abs(event.lambda - record.lambda) > lambda_tolerance)
                {
                    continue;
                }
                if(!event.state_reference.vector_available || !record.vector_available)
                {
                    continue;
                }
                const T distance = state_distance(event.state_reference, record);
                if(distance <= state_tolerance)
                {
                    match = &event;
                    break;
                }
            }

            if(match == nullptr)
            {
                event_node event;
                event.id = make_id(record);
                event.lambda = record.lambda;
                event.lambda_min = record.lambda;
                event.lambda_max = record.lambda;
                event.previous_orbit_type = record.previous_orbit_type;
                event.candidate_orbit_type = record.candidate_orbit_type;
                event.state_reference = record;
                event.incidents.push_back(record);
                events.push_back(std::move(event));
            }
            else
            {
                const T incident_count = static_cast<T>(match->incidents.size());
                match->lambda =
                    (incident_count*match->lambda + record.lambda)/
                    (incident_count + T(1));
                match->lambda_min = std::min(match->lambda_min, record.lambda);
                match->lambda_max = std::max(match->lambda_max, record.lambda);
                match->incidents.push_back(record);
            }
        }
    }

    bool save() const
    {
        if(file_name.empty())
        {
            return false;
        }
        nlohmann::json root;
        root["version"] = 2;
        root["events"] = nlohmann::json::array();
        for(const auto& event: events)
        {
            nlohmann::json encoded;
            encoded["id"] = event.id;
            encoded["lambda"] = static_cast<double>(event.lambda);
            encoded["lambda_min"] = static_cast<double>(event.lambda_min);
            encoded["lambda_max"] = static_cast<double>(event.lambda_max);
            encoded["previous_orbit_type"] = encode_orbit_type(event.previous_orbit_type);
            encoded["candidate_orbit_type"] = encode_orbit_type(event.candidate_orbit_type);
            encoded["state_reference"] = encode_record(event.state_reference);
            encoded["incidents"] = nlohmann::json::array();
            for(const auto& incident: event.incidents)
            {
                encoded["incidents"].push_back(encode_record(incident));
            }
            root["events"].push_back(std::move(encoded));
        }

        const std::filesystem::path temporary = file_name.string() + ".tmp";
        std::ofstream stream(temporary, std::ofstream::out | std::ofstream::trunc);
        if(!stream)
        {
            return false;
        }
        stream << root.dump(4) << '\n';
        stream.close();
        if(!stream)
        {
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            return false;
        }

        std::error_code rename_error;
        std::filesystem::rename(temporary, file_name, rename_error);
        if(rename_error)
        {
            std::error_code remove_error;
            std::filesystem::remove(temporary, remove_error);
            return false;
        }
        return true;
    }

    bool load()
    {
        events.clear();
        std::ifstream stream(file_name);
        if(!stream)
        {
            return false;
        }
        nlohmann::json root;
        try
        {
            stream >> root;
            for(const auto& encoded: root.value("events", nlohmann::json::array()))
            {
                event_node event;
                event.id = encoded.value("id", std::string{});
                event.lambda = static_cast<T>(encoded.value("lambda", 0.0));
                event.lambda_min = static_cast<T>(
                    encoded.value("lambda_min", static_cast<double>(event.lambda)));
                event.lambda_max = static_cast<T>(
                    encoded.value("lambda_max", static_cast<double>(event.lambda)));
                event.previous_orbit_type = decode_orbit_type(
                    encoded.value("previous_orbit_type", nlohmann::json::object()));
                event.candidate_orbit_type = decode_orbit_type(
                    encoded.value("candidate_orbit_type", nlohmann::json::object()));
                event.state_reference = decode_record(
                    encoded.value("state_reference", nlohmann::json::object()));
                for(const auto& incident: encoded.value("incidents", nlohmann::json::array()))
                {
                    event.incidents.push_back(decode_record(incident));
                }
                events.push_back(std::move(event));
            }
        }
        catch(const std::exception&)
        {
            events.clear();
            return false;
        }
        return true;
    }

    const std::vector<event_node>& all() const
    {
        return events;
    }

private:
    std::filesystem::path file_name;
    std::vector<event_node> events;

    static T scalar_abs(const T value)
    {
        return value < T(0) ? -value : value;
    }

    static std::string make_id(const record_type& record)
    {
        std::ostringstream stream;
        stream << "symmetry-event-c" << record.curve_number
               << "-p" << record.point_index;
        return stream.str();
    }

    static nlohmann::json encode_orbit_type(
        const symmetry::translation::orbit_type& orbit)
    {
        return {
            {"group_dimension", orbit.group_dimension},
            {"active_rank", orbit.active_rank},
            {"continuous_isotropy_dimension", orbit.continuous_isotropy_dimension},
            {"finite_invariants", orbit.finite_invariants}
        };
    }

    static symmetry::translation::orbit_type decode_orbit_type(
        const nlohmann::json& encoded)
    {
        symmetry::translation::orbit_type orbit;
        orbit.group_dimension = encoded.value("group_dimension", std::size_t(0));
        orbit.active_rank = encoded.value("active_rank", std::size_t(0));
        orbit.continuous_isotropy_dimension =
            encoded.value("continuous_isotropy_dimension", std::size_t(0));
        orbit.finite_invariants = encoded.value(
            "finite_invariants",
            std::vector<std::uint64_t>{});
        return orbit;
    }

    static nlohmann::json encode_record(const record_type& record)
    {
        return {
            {"curve_number", record.curve_number},
            {"point_index", record.point_index},
            {"lambda", static_cast<double>(record.lambda)},
            {"vector_available", record.vector_available},
            {"vector_file_id", record.vector_file_id},
            {"segment_id", record.segment_id},
            {"semicurve_id", record.semicurve_id},
            {"previous_order", record.previous_order},
            {"candidate_order", record.candidate_order},
            {"previous_orbit_type", encode_orbit_type(record.previous_orbit_type)},
            {"candidate_orbit_type", encode_orbit_type(record.candidate_orbit_type)},
            {"previous_transverse_ratio", static_cast<double>(record.previous_transverse_ratio)},
            {"candidate_transverse_ratio", static_cast<double>(record.candidate_transverse_ratio)},
            {"refinements", record.refinements}
        };
    }

    static record_type decode_record(const nlohmann::json& encoded)
    {
        record_type record;
        record.curve_number = encoded.value("curve_number", -1);
        record.point_index = encoded.value("point_index", std::uint64_t(0));
        record.lambda = static_cast<T>(encoded.value("lambda", 0.0));
        record.vector_available = encoded.value("vector_available", false);
        record.vector_file_id = encoded.value("vector_file_id", std::uint64_t(0));
        record.segment_id = encoded.value("segment_id", std::uint64_t(0));
        record.semicurve_id = encoded.value("semicurve_id", std::uint64_t(0));
        record.previous_order = encoded.value("previous_order", std::size_t(1));
        record.candidate_order = encoded.value("candidate_order", std::size_t(1));
        if(encoded.contains("previous_orbit_type"))
        {
            record.previous_orbit_type = decode_orbit_type(
                encoded["previous_orbit_type"]);
        }
        else
        {
            record.previous_orbit_type =
                symmetry::translation::orbit_type::cyclic_1d(record.previous_order);
        }
        if(encoded.contains("candidate_orbit_type"))
        {
            record.candidate_orbit_type = decode_orbit_type(
                encoded["candidate_orbit_type"]);
        }
        else
        {
            record.candidate_orbit_type =
                symmetry::translation::orbit_type::cyclic_1d(record.candidate_order);
        }
        record.previous_transverse_ratio = static_cast<T>(
            encoded.value("previous_transverse_ratio", 0.0));
        record.candidate_transverse_ratio = static_cast<T>(
            encoded.value("candidate_transverse_ratio", 0.0));
        record.refinements = encoded.value("refinements", 0u);
        return record;
    }
};

} // namespace container

#endif
