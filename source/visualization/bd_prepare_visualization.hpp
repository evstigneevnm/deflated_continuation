#ifndef __VISUALIZATION_BD_PREPARE_VISUALIZATION_HPP__
#define __VISUALIZATION_BD_PREPARE_VISUALIZATION_HPP__

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <visualization/bd_curve_file_reader.h>
#include <visualization/physical_solution_writer.h>

namespace visualization
{

struct bd_prepare_options
{
    std::filesystem::path project_dir;
    std::filesystem::path output_dir;
    std::size_t stride = 1;
    std::size_t max_points_per_branch = 0;
    std::set<std::size_t> branches;
    bool quiet = false;
};

inline void print_prepare_usage(const char* executable)
{
    std::cerr
        << "Usage: " << executable
        << " [config.json] [--project-dir path] [--output-dir path] [--stride n] [--max-points n] [--branch id] [--quiet]\n";
}

inline std::size_t parse_size_argument(const std::string& value, const char* label)
{
    std::istringstream stream(value);
    std::size_t result = 0;
    stream >> result;
    if(!stream)
    {
        throw std::runtime_error(std::string("failed to parse ") + label + " from '" + value + "'");
    }
    return result;
}

inline bool parse_common_prepare_argument(
    const std::string& arg,
    int& argi,
    const int argc,
    char const* argv[],
    bd_prepare_options& options)
{
    if(arg == "--quiet")
    {
        options.quiet = true;
        return true;
    }
    if(arg == "--output-dir")
    {
        if(argi + 1 >= argc)
        {
            throw std::runtime_error("--output-dir requires a path");
        }
        options.output_dir = argv[++argi];
        return true;
    }
    if(arg == "--project-dir")
    {
        if(argi + 1 >= argc)
        {
            throw std::runtime_error("--project-dir requires a path");
        }
        options.project_dir = argv[++argi];
        return true;
    }
    if(arg == "--stride")
    {
        if(argi + 1 >= argc)
        {
            throw std::runtime_error("--stride requires a positive integer");
        }
        options.stride = std::max<std::size_t>(parse_size_argument(argv[++argi], "stride"), 1);
        return true;
    }
    if(arg == "--max-points")
    {
        if(argi + 1 >= argc)
        {
            throw std::runtime_error("--max-points requires an integer");
        }
        options.max_points_per_branch = parse_size_argument(argv[++argi], "max points");
        return true;
    }
    if(arg == "--branch")
    {
        if(argi + 1 >= argc)
        {
            throw std::runtime_error("--branch requires an integer branch id");
        }
        options.branches.insert(parse_size_argument(argv[++argi], "branch id"));
        return true;
    }
    return false;
}

inline std::filesystem::path default_visualization_output_dir(const std::filesystem::path& project_dir)
{
    return project_dir / "visualization";
}

template<class VectorOperations, class VectorFileOperations, class Writer>
class bd_visualization_preparer
{
public:
    using vector_type = typename VectorOperations::vector_type;

    bd_visualization_preparer(VectorOperations* vec_ops_, VectorFileOperations* file_ops_):
        vec_ops(vec_ops_),
        file_ops(file_ops_)
    {
        vec_ops->init_vector(state);
        vec_ops->start_use_vector(state);
    }

    bd_visualization_preparer(const bd_visualization_preparer&) = delete;
    bd_visualization_preparer& operator=(const bd_visualization_preparer&) = delete;

    ~bd_visualization_preparer()
    {
        vec_ops->stop_use_vector(state);
        vec_ops->free_vector(state);
    }

    std::size_t prepare(const bd_prepare_options& raw_options, Writer& writer)
    {
        bd_prepare_options options = raw_options;
        if(options.project_dir.empty())
        {
            throw std::runtime_error("visualization: project_dir is empty");
        }
        if(options.output_dir.empty())
        {
            options.output_dir = default_visualization_output_dir(options.project_dir);
        }
        if(options.stride == 0)
        {
            options.stride = 1;
        }

        std::filesystem::create_directories(options.output_dir);
        const auto manifest_path = options.output_dir / "manifest.jsonl";
        std::ofstream manifest(manifest_path);
        if(!manifest)
        {
            throw std::runtime_error("visualization: failed to open manifest " + manifest_path.string());
        }

        std::size_t total_written = 0;
        for(const auto& branch_dir: numeric_branch_directories(options.project_dir))
        {
            const std::size_t branch_id = parse_size_argument(branch_dir.filename().string(), "branch id");
            if(!options.branches.empty() && options.branches.count(branch_id) == 0)
            {
                continue;
            }
            total_written += prepare_branch(branch_id, branch_dir, options, writer, manifest);
        }

        if(!options.quiet)
        {
            std::cout << "visualization: wrote " << total_written
                      << " solution snapshots to " << options.output_dir << std::endl;
            std::cout << "visualization: manifest " << manifest_path << std::endl;
        }
        return total_written;
    }

private:
    std::size_t prepare_branch(
        const std::size_t branch_id,
        const std::filesystem::path& branch_dir,
        const bd_prepare_options& options,
        Writer& writer,
        std::ofstream& manifest)
    {
        const auto curve_file = std::filesystem::exists(branch_dir / "debug_curve_all.dat")
            ? branch_dir / "debug_curve_all.dat"
            : branch_dir / "debug_curve.dat";
        if(!std::filesystem::exists(curve_file))
        {
            return 0;
        }

        const auto rows = read_debug_curve_file(curve_file);
        const auto metadata = read_curve_metadata_file(branch_dir / "metadata_curve.dat");
        const auto branch_output_dir = options.output_dir / ("branch_" + padded_number(branch_id, 4));
        std::filesystem::create_directories(branch_output_dir);

        std::size_t saved_candidates = 0;
        std::size_t written = 0;
        for(const auto& row: rows)
        {
            bd_curve_metadata_row meta;
            bool has_meta = false;
            const auto meta_it = metadata.find(row.index);
            if(meta_it != metadata.end())
            {
                meta = meta_it->second;
                has_meta = true;
            }

            const bool saved = has_meta ? meta.saved : row.file_id != 0;
            const std::uint64_t file_id = has_meta ? meta.file_id : row.file_id;
            if(!saved || file_id == 0)
            {
                continue;
            }

            if(saved_candidates%options.stride != 0)
            {
                ++saved_candidates;
                continue;
            }
            ++saved_candidates;

            if(options.max_points_per_branch != 0 && written >= options.max_points_per_branch)
            {
                break;
            }

            const auto state_file = branch_dir / std::to_string(file_id);
            if(!std::filesystem::exists(state_file))
            {
                if(!options.quiet)
                {
                    std::cerr << "visualization: missing saved vector " << state_file << ", skipping\n";
                }
                continue;
            }

            file_ops->read_vector(state_file.string(), state);
            const auto output_prefix = branch_output_dir / ("point_" + padded_number(row.index, 6));
            const auto write_result = writer.write(state, output_prefix, branch_id, row.index);
            write_manifest_row(
                manifest,
                branch_id,
                row,
                has_meta ? meta : default_metadata(row),
                state_file,
                write_result,
                options.output_dir);
            ++written;
        }

        if(!options.quiet)
        {
            std::cout << "visualization: branch " << branch_id << " wrote " << written << " snapshots\n";
        }
        return written;
    }

    static bd_curve_metadata_row default_metadata(const bd_curve_row& row)
    {
        bd_curve_metadata_row meta;
        meta.index = row.index;
        meta.lambda = row.lambda;
        meta.saved = row.file_id != 0;
        meta.file_id = row.file_id;
        return meta;
    }

    static std::string padded_number(const std::size_t value, const int width)
    {
        std::ostringstream stream;
        stream << std::setw(width) << std::setfill('0') << value;
        return stream.str();
    }

    static std::string path_for_manifest(
        const std::filesystem::path& path,
        const std::filesystem::path& base)
    {
        std::error_code error;
        const auto relative = std::filesystem::relative(path, base, error);
        return error ? path.string() : relative.string();
    }

    static void write_manifest_row(
        std::ofstream& manifest,
        const std::size_t branch_id,
        const bd_curve_row& row,
        const bd_curve_metadata_row& meta,
        const std::filesystem::path& state_file,
        const visualization_write_result& result,
        const std::filesystem::path& output_dir)
    {
        manifest << "{";
        manifest << "\"branch\":" << branch_id << ",";
        manifest << "\"point_index\":" << row.index << ",";
        manifest << "\"lambda\":" << std::setprecision(17) << row.lambda << ",";
        manifest << "\"file_id\":" << row.file_id << ",";
        manifest << "\"state_file\":\"" << detail::json_escape(state_file.string()) << "\",";
        manifest << "\"data_file\":\"" << detail::json_escape(path_for_manifest(result.data_file, output_dir)) << "\",";
        manifest << "\"kind\":\"" << detail::json_escape(result.kind) << "\",";
        manifest << "\"points\":" << result.points << ",";
        manifest << "\"components\":" << result.components << ",";
        manifest << "\"coordinate_extent\":" << std::setprecision(17) << result.coordinate_extent << ",";
        manifest << "\"segment_id\":" << meta.segment_id << ",";
        manifest << "\"semicurve_id\":" << meta.semicurve_id << ",";
        manifest << "\"forced_store\":" << (meta.forced_store ? "true" : "false") << ",";
        manifest << "\"endpoint_reason\":\"" << detail::json_escape(meta.endpoint_reason) << "\",";
        manifest << "\"norms\":[";
        for(std::size_t i = 0; i < row.norms.size(); ++i)
        {
            if(i != 0)
            {
                manifest << ",";
            }
            manifest << std::setprecision(17) << row.norms[i];
        }
        manifest << "]}";
        manifest << "\n";
    }

    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    vector_type state;
};

} // namespace visualization

#endif
