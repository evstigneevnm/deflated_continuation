#ifndef STABILITY_PERSISTENCE_CLASSIFICATION_UNCERTAINTY_REGISTRY_H
#define STABILITY_PERSISTENCE_CLASSIFICATION_UNCERTAINTY_REGISTRY_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>

#include <stability/analysis/stability_point_result.h>

namespace stability
{
namespace persistence
{

enum class classification_uncertainty_stage
{
    point_classification,
    endpoint_confirmation,
    transition_refinement,
    source_path_topology
};

inline const char* classification_uncertainty_stage_name(
    classification_uncertainty_stage stage)
{
    switch(stage)
    {
    case classification_uncertainty_stage::point_classification:
        return "point_classification";
    case classification_uncertainty_stage::endpoint_confirmation:
        return "endpoint_confirmation";
    case classification_uncertainty_stage::transition_refinement:
        return "transition_refinement";
    case classification_uncertainty_stage::source_path_topology:
        return "source_path_topology";
    }
    return "point_classification";
}

inline classification_uncertainty_stage
classification_uncertainty_stage_from_name(const std::string& name)
{
    if(name == "endpoint_confirmation")
    {
        return classification_uncertainty_stage::endpoint_confirmation;
    }
    if(name == "transition_refinement")
    {
        return classification_uncertainty_stage::transition_refinement;
    }
    if(name == "source_path_topology")
    {
        return classification_uncertainty_stage::source_path_topology;
    }
    return classification_uncertainty_stage::point_classification;
}

template<class Real>
class classification_uncertainty_registry
{
public:
    using observation_type =
        analysis::unstable_dimension_observation;
    using stage_type = classification_uncertainty_stage;

    struct options
    {
        bool enabled = true;
        std::filesystem::path file_name =
            "stability_uncertainty_registry.json";
        std::size_t maximum_diagnostic_length = 16384;
        bool retain_resolved = true;
    };

    struct entry
    {
        std::string id;
        std::size_t curve_number = 0;
        std::uint64_t lower_source_point = 0;
        std::uint64_t upper_source_point = 0;
        Real lower_parameter = Real{};
        Real upper_parameter = Real{};
        Real failed_parameter = Real{};
        stage_type stage = stage_type::point_classification;
        std::size_t failure_count = 0;
        std::size_t resolution_count = 0;
        bool resolved = false;
        std::vector<observation_type> observations;
        std::string diagnostic;
    };

    explicit classification_uncertainty_registry(
        options configured_options = {})
        : options_(std::move(configured_options))
    {
        if(
            options_.enabled &&
            (
                options_.file_name.empty() ||
                options_.maximum_diagnostic_length == 0))
        {
            throw std::invalid_argument(
                "classification uncertainty registry requires a file "
                "name and a positive diagnostic length limit");
        }
        if(
            options_.enabled &&
            std::filesystem::exists(options_.file_name) &&
            !load())
        {
            throw std::runtime_error(
                "failed to load classification uncertainty registry: " +
                options_.file_name.string());
        }
    }

    bool enabled() const
    {
        return options_.enabled;
    }

    const options& settings() const
    {
        return options_;
    }

    const std::vector<entry>& all() const
    {
        return entries_;
    }

    std::size_t unresolved_count() const
    {
        return static_cast<std::size_t>(std::count_if(
            entries_.begin(),
            entries_.end(),
            [](const entry& value)
            {
                return !value.resolved;
            }));
    }

    bool record_failure(
        std::size_t curve_number,
        std::uint64_t lower_source_point,
        Real lower_parameter,
        std::uint64_t upper_source_point,
        Real upper_parameter,
        Real failed_parameter,
        stage_type stage,
        std::vector<observation_type> observations,
        std::string diagnostic)
    {
        if(!options_.enabled)
            return true;

        const std::vector<entry> previous_entries = entries_;

        entry* value = find(
            curve_number,
            lower_source_point,
            upper_source_point,
            stage);
        if(value == nullptr)
        {
            entry inserted;
            inserted.id = make_id(
                curve_number,
                lower_source_point,
                upper_source_point,
                stage);
            inserted.curve_number = curve_number;
            inserted.lower_source_point = lower_source_point;
            inserted.upper_source_point = upper_source_point;
            inserted.stage = stage;
            entries_.push_back(std::move(inserted));
            value = &entries_.back();
        }

        value->lower_parameter = lower_parameter;
        value->upper_parameter = upper_parameter;
        value->failed_parameter = failed_parameter;
        ++value->failure_count;
        value->resolved = false;
        value->observations = std::move(observations);
        value->diagnostic = truncate(std::move(diagnostic));
        if(save())
            return true;
        entries_ = previous_entries;
        return false;
    }

    bool resolve(
        std::size_t curve_number,
        std::uint64_t lower_source_point,
        std::uint64_t upper_source_point,
        stage_type stage)
    {
        if(!options_.enabled)
            return true;
        entry* value = find(
            curve_number,
            lower_source_point,
            upper_source_point,
            stage);
        if(value == nullptr || value->resolved)
            return true;

        const std::vector<entry> previous_entries = entries_;

        if(options_.retain_resolved)
        {
            value->resolved = true;
            ++value->resolution_count;
        }
        else
        {
            const std::size_t target_curve = value->curve_number;
            const std::uint64_t target_lower =
                value->lower_source_point;
            const std::uint64_t target_upper =
                value->upper_source_point;
            const stage_type target_stage = value->stage;
            entries_.erase(
                std::remove_if(
                    entries_.begin(),
                    entries_.end(),
                    [
                        target_curve,
                        target_lower,
                        target_upper,
                        target_stage
                    ](const entry& candidate)
                    {
                        return
                            candidate.curve_number == target_curve &&
                            candidate.lower_source_point == target_lower &&
                            candidate.upper_source_point == target_upper &&
                            candidate.stage == target_stage;
                    }),
                entries_.end());
        }
        if(save())
            return true;
        entries_ = previous_entries;
        return false;
    }

    bool load()
    {
        entries_.clear();
        if(!options_.enabled || options_.file_name.empty())
            return false;

        std::ifstream stream(options_.file_name);
        if(!stream)
            return false;

        try
        {
            nlohmann::json root;
            stream >> root;
            for(const auto& encoded :
                root.value("entries", nlohmann::json::array()))
            {
                entry value;
                value.id = encoded.value("id", std::string{});
                value.curve_number = encoded.value(
                    "curve_number",
                    std::size_t(0));
                value.lower_source_point = encoded.value(
                    "lower_source_point",
                    std::uint64_t(0));
                value.upper_source_point = encoded.value(
                    "upper_source_point",
                    std::uint64_t(0));
                value.lower_parameter = static_cast<Real>(
                    encoded.value("lower_parameter", 0.0));
                value.upper_parameter = static_cast<Real>(
                    encoded.value("upper_parameter", 0.0));
                value.failed_parameter = static_cast<Real>(
                    encoded.value("failed_parameter", 0.0));
                value.stage = classification_uncertainty_stage_from_name(
                    encoded.value("stage", std::string{}));
                value.failure_count = encoded.value(
                    "failure_count",
                    std::size_t(0));
                value.resolution_count = encoded.value(
                    "resolution_count",
                    std::size_t(0));
                value.resolved = encoded.value("resolved", false);
                value.diagnostic = truncate(encoded.value(
                    "diagnostic",
                    std::string{}));
                for(const auto& observed : encoded.value(
                    "observations",
                    nlohmann::json::array()))
                {
                    observation_type observation;
                    observation.signature.real = observed.value(
                        "real",
                        0);
                    observation.signature.complex_pairs = observed.value(
                        "complex_pairs",
                        0);
                    observation.occurrences = observed.value(
                        "occurrences",
                        std::size_t(0));
                    value.observations.push_back(observation);
                }
                if(value.id.empty())
                {
                    value.id = make_id(
                        value.curve_number,
                        value.lower_source_point,
                        value.upper_source_point,
                        value.stage);
                }
                entries_.push_back(std::move(value));
            }
        }
        catch(const std::exception&)
        {
            entries_.clear();
            return false;
        }
        return true;
    }

    bool save() const
    {
        if(!options_.enabled || options_.file_name.empty())
            return false;

        const std::filesystem::path parent =
            options_.file_name.parent_path();
        std::error_code error;
        if(!parent.empty())
        {
            std::filesystem::create_directories(parent, error);
            if(error)
                return false;
        }

        nlohmann::json root;
        root["version"] = 1;
        root["entries"] = nlohmann::json::array();
        for(const auto& value : entries_)
        {
            nlohmann::json encoded;
            encoded["id"] = value.id;
            encoded["curve_number"] = value.curve_number;
            encoded["lower_source_point"] = value.lower_source_point;
            encoded["upper_source_point"] = value.upper_source_point;
            encoded["lower_parameter"] =
                static_cast<double>(value.lower_parameter);
            encoded["upper_parameter"] =
                static_cast<double>(value.upper_parameter);
            encoded["failed_parameter"] =
                static_cast<double>(value.failed_parameter);
            encoded["stage"] =
                classification_uncertainty_stage_name(value.stage);
            encoded["failure_count"] = value.failure_count;
            encoded["resolution_count"] = value.resolution_count;
            encoded["resolved"] = value.resolved;
            encoded["diagnostic"] = value.diagnostic;
            encoded["observations"] = nlohmann::json::array();
            for(const auto& observation : value.observations)
            {
                encoded["observations"].push_back({
                    {"real", observation.signature.real},
                    {"complex_pairs",
                        observation.signature.complex_pairs},
                    {"occurrences", observation.occurrences}
                });
            }
            root["entries"].push_back(std::move(encoded));
        }

        const std::filesystem::path temporary =
            options_.file_name.string() + ".tmp";
        std::ofstream stream(
            temporary,
            std::ofstream::out | std::ofstream::trunc);
        if(!stream)
            return false;
        stream << root.dump(4) << '\n';
        stream.close();
        if(!stream)
        {
            std::filesystem::remove(temporary, error);
            return false;
        }

        std::filesystem::rename(
            temporary,
            options_.file_name,
            error);
        if(error)
        {
            std::filesystem::remove(temporary, error);
            return false;
        }
        return true;
    }

private:
    options options_;
    std::vector<entry> entries_;

    entry* find(
        std::size_t curve_number,
        std::uint64_t lower_source_point,
        std::uint64_t upper_source_point,
        stage_type stage)
    {
        for(auto& value : entries_)
        {
            if(
                value.curve_number == curve_number &&
                value.lower_source_point == lower_source_point &&
                value.upper_source_point == upper_source_point &&
                value.stage == stage)
            {
                return &value;
            }
        }
        return nullptr;
    }

    std::string truncate(std::string diagnostic) const
    {
        if(diagnostic.size() > options_.maximum_diagnostic_length)
            diagnostic.resize(options_.maximum_diagnostic_length);
        return diagnostic;
    }

    static std::string make_id(
        std::size_t curve_number,
        std::uint64_t lower_source_point,
        std::uint64_t upper_source_point,
        stage_type stage)
    {
        return
            "stability-uncertainty-c" +
            std::to_string(curve_number) + "-p" +
            std::to_string(lower_source_point) + "-" +
            std::to_string(upper_source_point) + "-" +
            classification_uncertainty_stage_name(stage);
    }
};

} // namespace persistence
} // namespace stability

#endif
