#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/bifurcation_diagram/diagram_archive.h>
#include <containers/stability_diagram.h>

namespace
{

struct vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;
};

struct vector_file_operations
{
    void write_vector(
        const std::string& file_name,
        const vector_operations::vector_type& vector)
    {
        std::ofstream output(file_name);
        for(const double value : vector)
            output << value << '\n';
    }

    void read_vector(
        const std::string&,
        vector_operations::vector_type&)
    {
    }
};

struct null_log
{
    template<class... Args>
    void info_f(const char*, Args&&...)
    {
    }

    template<class... Args>
    void warning_f(const char*, Args&&...)
    {
    }
};

using diagram_type = container::stability_diagram<
    vector_operations,
    vector_file_operations,
    null_log>;

struct throwing_diagram
{
    template<class Archive>
    void serialize(Archive&, const unsigned int)
    {
        throw std::runtime_error("intentional serialization failure");
    }
};

void require(bool condition, const std::string& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

std::vector<std::string> data_lines(
    const std::filesystem::path& file_name)
{
    std::ifstream input(file_name);
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(input, line))
    {
        if(!line.empty() && line.front() != '#')
            lines.push_back(line);
    }
    return lines;
}

}

int main()
{
    const auto unique_id =
        std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
    const auto project_path =
        std::filesystem::temp_directory_path()/
        ("deflated_continuation_stability_archive_" +
         std::to_string(unique_id));
    const auto archive_path =
        project_path/"stability_diagram.dat";

    vector_operations vector_ops;
    vector_file_operations file_ops;
    null_log log;

    try
    {
        std::filesystem::create_directory(project_path);

        diagram_type source(
            &vector_ops,
            &file_ops,
            &log,
            project_path.string());
        source.open_curve(0);
        source.add_with_plot_data(
            1.0,
            0,
            0,
            3,
            {0.25, 1.5},
            {0, 0},
            {0, 0});
        source.add_with_plot_data(
            2.0,
            1,
            0,
            7,
            {0.5, 2.5},
            {1, 0},
            {1, 0});
        require(
            source.update_regular_point_dimension(3, {0, 1}),
            "pending regular stability point can be revised");
        source.close_curve();

        require(source.curve_count() == 1, "source curve count");
        require(
            !std::filesystem::exists(
                project_path/"0"/"debug_curve_stability.dat.tmp"),
            "curve output leaves no temporary file");
        require(
            std::filesystem::is_regular_file(
                project_path/"0"/
                    "debug_curve_stability_plot.dat") &&
                !std::filesystem::exists(
                    project_path/"0"/
                        "debug_curve_stability_plot.dat.tmp"),
            "plot sidecar is committed transactionally");
        const auto plot_lines = data_lines(
            project_path/"0"/"debug_curve_stability_plot.dat");
        require(
            plot_lines.size() == 2 &&
                plot_lines[0].find(
                    "3 1 unstable 0 1 0 1 0 1 none 0 2 "
                    "0.25 1.5") == 0 &&
                plot_lines[1].find(
                    "7 2 unstable 1 0 1 0 1 0 none 0 2 "
                    "0.5 2.5") == 0,
            "plot sidecar records source indices and norms");
        const auto source_points =
            source.get_curve_points_vector(0);
        require(source_points.size() == 2, "source point count");
        require(
            source_points[0].point_type == "unstable" &&
                source_points[0].unstable_dim_C == 1 &&
                source_points[1].point_type == "unstable",
            "source point classification revision");

        const auto saved = container::save_diagram_archive(
            archive_path.string(),
            source);
        require(saved.succeeded(), "initial archive save");
        require(
            !std::filesystem::exists(
                archive_path.string() + ".tmp"),
            "archive save leaves no temporary file");

        throwing_diagram invalid;
        const auto failed_save = container::save_diagram_archive(
            archive_path.string(),
            invalid);
        require(
            failed_save.status ==
                container::diagram_archive_status::archive_failed,
            "failed serialization reports archive failure");
        require(
            std::filesystem::is_regular_file(archive_path) &&
                !std::filesystem::exists(
                    archive_path.string() + ".tmp"),
            "failed serialization preserves committed archive");

        diagram_type restored(
            &vector_ops,
            &file_ops,
            &log,
            project_path.string());
        const auto loaded = container::load_diagram_archive(
            archive_path.string(),
            restored);
        require(loaded.succeeded(), "archive load");
        require(restored.curve_count() == 1, "restored curve count");
        require(
            restored.get_curve_points_vector(0).size() == 2,
            "restored point count");

        restored.open_curve(1);
        restored.add(3.0, 0, 1);
        restored.close_curve();
        require(restored.curve_count() == 2, "restart append count");
        const auto appended =
            restored.get_curve_points_vector(1);
        require(
            appended.size() == 1 &&
                appended[0].point_type == "unstable" &&
                appended[0].unstable_dim_C == 1,
            "restart append record");

        restored.open_curve(2);
        const vector_operations::vector_type bifurcation_state{
            1.0,
            2.0};
        restored.add_with_plot_data(
            4.0,
            1,
            0,
            11,
            {3.0, 4.0},
            {0, 0},
            {1, 0},
            bifurcation_state);
        const auto pending_file = project_path/"2"/"s1";
        require(
            std::filesystem::is_regular_file(pending_file),
            "transactional vector was written");
        restored.abandon_curve();
        require(
            !std::filesystem::exists(pending_file) &&
                !std::filesystem::exists(
                    project_path/"2"/
                        "debug_curve_stability_plot.dat") &&
                restored.curve_count() == 2,
            "aborted curve removes pending vectors");

        restored.open_curve(2);
        restored.add_with_plot_data(
            4.0,
            1,
            0,
            11,
            {3.0, 4.0},
            {0, 0},
            {1, 0},
            bifurcation_state);
        restored.close_curve();
        const auto event_lines = data_lines(
            project_path/"2"/"debug_curve_stability_plot.dat");
        require(
            event_lines.size() == 1 &&
                event_lines[0].find(
                    "11 4 bifurcation 1 0 0 0 1 0 steady 1 2 "
                    "3 4") == 0,
            "plot sidecar stores refined steady event metadata");

        const auto resaved = container::save_diagram_archive(
            archive_path.string(),
            restored);
        require(resaved.succeeded(), "restart archive save");
    }
    catch(const std::exception& error)
    {
        std::filesystem::remove_all(project_path);
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::filesystem::remove_all(project_path);
    std::cout << "PASSED\n";
    return 0;
}
