#ifndef __SYMMETRY_FINITE_GROUP_MANIFEST_H__
#define __SYMMETRY_FINITE_GROUP_MANIFEST_H__

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <contrib/json/nlohmann/json.hpp>

namespace symmetry
{

enum class finite_group_manifest_status
{
    success,
    missing,
    read_failed,
    malformed,
    commit_failed
};

struct finite_group_manifest
{
    // Version 2 records that archive compatibility is validated with the
    // direct group-orbit distance, not independent canonical representatives.
    static constexpr unsigned int current_version = 2;

    unsigned int version = current_version;
    std::string fingerprint;
    std::vector<std::string> actions;
};

struct finite_group_manifest_result
{
    finite_group_manifest_status status =
        finite_group_manifest_status::read_failed;
    finite_group_manifest manifest;
    std::string message;

    bool succeeded() const
    {
        return status == finite_group_manifest_status::success;
    }
};

inline finite_group_manifest_result load_finite_group_manifest(
    const std::filesystem::path& file_name)
{
    std::ifstream input(file_name);
    if(!input)
    {
        std::error_code error;
        return {
            std::filesystem::exists(file_name, error)
                ? finite_group_manifest_status::read_failed
                : finite_group_manifest_status::missing,
            {},
            "unable to open symmetry-group manifest"};
    }

    try
    {
        nlohmann::json json;
        input >> json;
        finite_group_manifest manifest;
        manifest.version = json.at("version").get<unsigned int>();
        manifest.fingerprint = json.at("fingerprint").get<std::string>();
        manifest.actions =
            json.at("actions").get<std::vector<std::string>>();
        const std::size_t declared_order =
            json.at("order").get<std::size_t>();
        if(manifest.version == 0 ||
           manifest.version > finite_group_manifest::current_version)
        {
            return {
                finite_group_manifest_status::malformed,
                std::move(manifest),
                "unsupported symmetry-group manifest version"};
        }
        if(manifest.fingerprint.empty() || manifest.actions.empty())
        {
            return {
                finite_group_manifest_status::malformed,
                std::move(manifest),
                "symmetry-group manifest has an empty definition"};
        }
        if(declared_order != manifest.actions.size())
        {
            return {
                finite_group_manifest_status::malformed,
                std::move(manifest),
                "symmetry-group manifest order does not match its actions"};
        }
        if(manifest.actions.front() != "identity")
        {
            return {
                finite_group_manifest_status::malformed,
                std::move(manifest),
                "symmetry-group manifest does not start with identity"};
        }
        auto sorted_actions = manifest.actions;
        std::sort(sorted_actions.begin(), sorted_actions.end());
        if(std::adjacent_find(
               sorted_actions.begin(),
               sorted_actions.end()) != sorted_actions.end())
        {
            return {
                finite_group_manifest_status::malformed,
                std::move(manifest),
                "symmetry-group manifest contains duplicate action names"};
        }
        return {
            finite_group_manifest_status::success,
            std::move(manifest),
            {}};
    }
    catch(const std::exception& error)
    {
        return {
            finite_group_manifest_status::malformed,
            {},
            error.what()};
    }
}

inline finite_group_manifest_result save_finite_group_manifest(
    const std::filesystem::path& file_name,
    const finite_group_manifest& manifest)
{
    if(manifest.fingerprint.empty() || manifest.actions.empty())
    {
        return {
            finite_group_manifest_status::malformed,
            {},
            "cannot save an empty symmetry-group definition"};
    }

    const std::filesystem::path temporary =
        file_name.string() + ".tmp";
    const auto remove_temporary = [&temporary]()
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
    };
    remove_temporary();

    std::ofstream output(temporary);
    if(!output)
    {
        return {
            finite_group_manifest_status::read_failed,
            {},
            "unable to open temporary symmetry-group manifest"};
    }
    try
    {
        nlohmann::json json{
            {"version", finite_group_manifest::current_version},
            {"fingerprint", manifest.fingerprint},
            {"order", manifest.actions.size()},
            {"actions", manifest.actions}};
        output << json.dump(2) << '\n';
        output.flush();
        output.close();
        if(!output)
        {
            remove_temporary();
            return {
                finite_group_manifest_status::read_failed,
                {},
                "symmetry-group manifest write did not complete"};
        }
    }
    catch(const std::exception& error)
    {
        output.close();
        remove_temporary();
        return {
            finite_group_manifest_status::malformed,
            {},
            error.what()};
    }

    std::error_code error;
    std::filesystem::rename(temporary, file_name, error);
    if(error)
    {
        remove_temporary();
        return {
            finite_group_manifest_status::commit_failed,
            {},
            "unable to replace symmetry-group manifest: " +
                error.message()};
    }
    return {
        finite_group_manifest_status::success,
        manifest,
        {}};
}

inline bool finite_group_manifest_matches(
    const finite_group_manifest& manifest,
    const std::string& fingerprint,
    const std::vector<std::string>& actions)
{
    return manifest.version == finite_group_manifest::current_version &&
        manifest.fingerprint == fingerprint &&
        manifest.actions == actions;
}

} // namespace symmetry

#endif
