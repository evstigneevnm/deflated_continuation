#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/bifurcation_diagram/curve_metadata_io.h>

namespace
{

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

}

int main()
{
    using point_type = container::complex_values<double>;
    const auto file_name =
        std::filesystem::temp_directory_path()/"deflated_continuation_curve_metadata_test.dat";
    std::error_code error;
    std::filesystem::remove(file_name, error);

    try
    {
        std::vector<point_type> points(4);
        points[0].lambda = 1.25;
        points[0].is_data_avaliable = true;
        points[0].id_file_name = 7;
        points[0].segment_id = 2;
        points[0].semicurve_id = 3;
        points[0].forced_store = true;
        points[0].endpoint_reason = container::curve_endpoint_reason::known_branch;

        points[1].lambda = 2.5;
        points[1].id_file_name = 0;
        points[1].segment_id = 4;
        points[1].semicurve_id = 5;
        points[1].endpoint_reason = container::curve_endpoint_reason::hard_failure;

        points[2].lambda = 3.75;
        points[2].id_file_name = 11;
        points[2].segment_id = 4;
        points[2].semicurve_id = 5;
        points[2].endpoint_reason = container::curve_endpoint_reason::hard_failure;

        points[3].lambda = 4.0;
        points[3].is_data_avaliable = true;
        points[3].id_file_name = 12;
        points[3].segment_id = 4;
        points[3].semicurve_id = 5;
        points[3].endpoint_reason =
            container::curve_endpoint_reason::symmetry_intersection;

        require(container::write_curve_metadata(file_name.string(), points), "metadata write");

        std::ifstream raw(file_name);
        std::string header;
        std::getline(raw, header);
        require(
            header == "# index lambda saved id_file_name segment_id semicurve_id forced_store endpoint_reason",
            "metadata header");

        std::vector<point_type> loaded(4);
        const auto result = container::load_curve_metadata(file_name.string(), loaded);
        require(result.loaded_any, "metadata load");
        require(result.incomplete_segment_ids.size() == 1, "incomplete segment deduplication");
        require(result.incomplete_segment_ids.front() == 4, "incomplete segment id");
        require(loaded[0].point_index == 0, "point index");
        require(loaded[0].segment_id == 2 && loaded[0].semicurve_id == 3, "segment metadata");
        require(loaded[0].forced_store, "forced store");
        require(loaded[0].endpoint_reason == container::curve_endpoint_reason::known_branch,
                "known branch endpoint");
        require(loaded[1].endpoint_reason == container::curve_endpoint_reason::hard_failure,
                "hard failure endpoint");
        require(
            container::curve_endpoint_reason_from_string(
                "boundary_max_approximate") ==
                container::curve_endpoint_reason::
                    boundary_max_approximate,
            "approximate boundary endpoint parsing");
        require(
            !container::is_incomplete_endpoint(
                container::curve_endpoint_reason::
                    boundary_max_approximate),
            "approximate boundary endpoint is complete");
        require(
            container::is_stability_refinement_barrier(
                container::curve_endpoint_reason::known_branch) &&
                container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::
                        analytical_branch) &&
                container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::
                        symmetry_intersection) &&
                container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::
                        closed_return) &&
                container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::
                        self_intersection),
            "branch-topology endpoints block smooth stability refinement");
        require(
            !container::is_stability_refinement_barrier(
                container::curve_endpoint_reason::boundary_max) &&
                !container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::
                        boundary_max_approximate) &&
                !container::is_stability_refinement_barrier(
                    container::curve_endpoint_reason::max_steps),
            "ordinary and incomplete stopping endpoints remain on the smooth segment");
        require(
            container::starts_new_curve_traversal_segment(
                loaded[0],
                loaded[1]),
            "different segment starts a traversal segment");
        loaded[1].segment_id = loaded[0].segment_id;
        loaded[1].semicurve_id = loaded[0].semicurve_id;
        loaded[0].endpoint_reason =
            container::curve_endpoint_reason::none;
        require(
            !container::starts_new_curve_traversal_segment(
                loaded[0],
                loaded[1]),
            "same open semicurve remains contiguous");
        loaded[0].endpoint_reason =
            container::curve_endpoint_reason::known_branch;
        require(
            container::starts_new_curve_traversal_segment(
                loaded[0],
                loaded[1]),
            "terminal endpoint closes a traversal segment");
        loaded[1].is_data_avaliable = true;
        loaded[1].endpoint_reason =
            container::curve_endpoint_reason::none;
        loaded[2].is_data_avaliable = false;
        loaded[2].segment_id = loaded[1].segment_id;
        loaded[2].semicurve_id = loaded[1].semicurve_id;
        loaded[2].endpoint_reason =
            container::curve_endpoint_reason::none;
        loaded[3].is_data_avaliable = true;
        loaded[3].segment_id = loaded[1].segment_id;
        loaded[3].semicurve_id = loaded[1].semicurve_id;
        const auto* nearby_symmetry_endpoint =
            container::following_symmetry_endpoint_within_source_points(
                loaded,
                loaded[1],
                2);
        require(
            nearby_symmetry_endpoint == &loaded[3],
            "guard detects a nearby symmetry endpoint");
        require(
            container::following_symmetry_endpoint_within_source_points(
                loaded,
                loaded[1],
                1) == nullptr,
            "guard excludes a symmetry endpoint outside its source-point window");
        require(
            container::following_symmetry_endpoint_within_source_points(
                loaded,
                loaded[1],
                0) == nullptr,
            "zero source-point guard disables symmetry endpoint omission");
        require(
            container::following_symmetry_endpoint_within_source_points(
                loaded,
                loaded[3],
                2) == nullptr,
            "symmetry endpoint does not guard itself");

        std::vector<point_type> missing(1);
        const auto missing_result = container::load_curve_metadata(
            (file_name.string() + ".missing"), missing);
        require(!missing_result.loaded_any, "missing metadata file");
    }
    catch(const std::exception& exception)
    {
        std::filesystem::remove(file_name, error);
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::filesystem::remove(file_name, error);
    std::cout << "PASSED\n";
    return 0;
}
