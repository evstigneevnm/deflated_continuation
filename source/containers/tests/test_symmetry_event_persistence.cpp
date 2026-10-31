#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <containers/symmetry_event_journal.h>
#include <containers/symmetry_event_registry.h>

namespace
{

using real     = double;
using record_t = container::symmetry_event_record<real>;

int checks   = 0;
int failures = 0;

void require_true( const bool value, const std::string &label )
{
    ++checks;
    if ( !value )
    {
        ++failures;
        std::cerr << "FAIL " << label << std::endl;
    }
}

record_t make_record( const int curve, const std::uint64_t point, const real lambda, const std::size_t candidate_order )
{
    record_t record;
    record.curve_number               = curve;
    record.point_index                = point;
    record.lambda                     = lambda;
    record.vector_available           = true;
    record.vector_file_id             = point + 1;
    record.segment_id                 = 2;
    record.semicurve_id               = 2;
    record.previous_order             = 1;
    record.candidate_order            = candidate_order;
    record.previous_orbit_type        = symmetry::translation::orbit_type::cyclic_1d( 1 );
    record.candidate_orbit_type       = symmetry::translation::orbit_type::cyclic_1d( candidate_order );
    record.previous_transverse_ratio  = 1.0e-4;
    record.candidate_transverse_ratio = 1.0e-12;
    record.refinements                = 6;
    return record;
}

symmetry::translation::orbit_type make_two_translation_orbit(
    const std::size_t active_rank, const std::size_t continuous_isotropy_dimension,
    std::vector<std::uint64_t> finite_invariants
)
{
    symmetry::translation::orbit_type orbit;
    orbit.group_dimension               = 2;
    orbit.active_rank                   = active_rank;
    orbit.continuous_isotropy_dimension = continuous_isotropy_dimension;
    orbit.finite_invariants             = std::move( finite_invariants );
    return orbit;
}

void test_journal_is_transactional( const std::filesystem::path &directory )
{
    const auto event_file     = directory / "symmetry_events.dat";
    const auto discarded_file = directory / "discarded_events.dat";

    {
        container::symmetry_event_journal<real> discarded( discarded_file );
        discarded.stage( make_record( 8, 10, 16.0, 2 ) );
        require_true( !std::filesystem::exists( discarded_file ), "staged discarded event is not visible on disk" );
    }
    require_true( !std::filesystem::exists( discarded_file ), "uncommitted journal leaves no file" );

    container::symmetry_event_journal<real> journal( event_file );
    journal.stage( make_record( 1, 511, 16.139856, 2 ) );
    require_true( !std::filesystem::exists( event_file ), "staged accepted event remains transactional" );
    require_true( journal.commit(), "accepted event journal commits" );
    require_true( std::filesystem::exists( event_file ), "committed event file exists" );

    container::symmetry_event_journal<real> loaded( event_file );
    require_true( loaded.load(), "committed event journal reloads" );
    require_true( loaded.all().size() == 1, "reloaded journal has one event" );
    require_true( loaded.all().front().vector_file_id == 512, "reloaded journal preserves vector reference" );
}

void test_legacy_event_file_is_read( const std::filesystem::path &directory )
{
    const auto    event_file = directory / "legacy_symmetry_events.dat";
    std::ofstream stream( event_file );
    stream << "# point_index lambda segment_id semicurve_id previous_order candidate_order "
           << "previous_transverse_ratio candidate_transverse_ratio refinements\n"
           << "95 50.90951545934403 1 1 1 3 2.0e-5 5.0e-11 6\n";
    stream.close();

    container::symmetry_event_journal<real> journal( event_file );
    require_true( journal.load(), "legacy event file reloads" );
    require_true( journal.all().size() == 1, "legacy event count" );
    require_true( journal.all().front().candidate_order == 3, "legacy event order is preserved" );
}

void test_registry_links_incident_endpoints( const std::filesystem::path &directory )
{
    const auto                               registry_file = directory / "symmetry_event_registry.json";
    container::symmetry_event_registry<real> registry( registry_file );
    const record_t                           first    = make_record( 4, 616, 50.90951545934403, 3 );
    const record_t                           second   = make_record( 6, 606, 50.90985766627742, 3 );
    const record_t                           separate = make_record( 9, 416, 50.9097, 3 );

    registry.rebuild( { first, second, separate }, 1.0e-3, 2.0e-3, []( const record_t &left, const record_t &right ) {
        const bool same_event = ( left.curve_number == 4 && right.curve_number == 6 ) ||
                                ( left.curve_number == 6 && right.curve_number == 4 );
        return same_event ? 6.3e-4 : 1.0;
    } );

    require_true( registry.all().size() == 2, "registry forms two event nodes" );
    require_true( registry.all().front().incidents.size() == 2, "registry links two incident curve endpoints" );
    require_true(
        registry.all().front().lambda_min == first.lambda && registry.all().front().lambda_max == second.lambda,
        "registry preserves the incident lambda bracket"
    );
    require_true( registry.all().front().id == "symmetry-event-c4-p616", "registry event ID is deterministic" );
    require_true( registry.save(), "registry saves atomically" );

    container::symmetry_event_registry<real> loaded( registry_file );
    require_true( loaded.load(), "registry reloads" );
    require_true( loaded.all().size() == 2, "reloaded registry node count" );
    require_true( loaded.all().front().incidents.size() == 2, "reloaded registry incident count" );
    require_true(
        loaded.all().front().lambda_min == first.lambda && loaded.all().front().lambda_max == second.lambda,
        "reloaded registry preserves the lambda bracket"
    );
    require_true(
        loaded.all().front().candidate_orbit_type.finite_isotropy_order() == 3, "reloaded registry orbit type"
    );
    require_true( loaded.save(), "reloaded registry atomically replaces existing file" );
}

void test_registry_preserves_general_orbit_types( const std::filesystem::path &directory )
{
    const auto                               registry_file = directory / "general_symmetry_event_registry.json";
    container::symmetry_event_registry<real> registry( registry_file );

    record_t first             = make_record( 1, 10, 12.5, 6 );
    first.previous_orbit_type  = make_two_translation_orbit( 2, 0, { 2, 3 } );
    first.candidate_orbit_type = make_two_translation_orbit( 1, 1, { 6 } );

    record_t same             = make_record( 2, 20, 12.50001, 6 );
    same.previous_orbit_type  = first.previous_orbit_type;
    same.candidate_orbit_type = first.candidate_orbit_type;

    record_t distinct             = make_record( 3, 30, 12.50002, 6 );
    distinct.previous_orbit_type  = make_two_translation_orbit( 2, 0, { 1, 6 } );
    distinct.candidate_orbit_type = first.candidate_orbit_type;

    registry.rebuild( { first, same, distinct }, 1.0e-3, 1.0e-3, []( const record_t &, const record_t & ) {
        return 0.0;
    } );
    require_true( registry.all().size() == 2, "equal finite orders with distinct invariants remain separate events" );
    require_true( registry.save(), "general orbit registry saves" );

    container::symmetry_event_registry<real> loaded( registry_file );
    require_true( loaded.load(), "general orbit registry reloads" );
    require_true(
        loaded.all().front().state_reference.previous_orbit_type == first.previous_orbit_type,
        "incident record preserves multidimensional previous orbit type"
    );
    require_true(
        loaded.all().front().state_reference.candidate_orbit_type == first.candidate_orbit_type,
        "incident record preserves rank-deficient candidate orbit type"
    );
}

void test_stabilizer_embeddings_and_journal_versions( const std::filesystem::path &directory )
{
    auto first                 = make_record( 1, 10, 28.1, 3 );
    first.previous_orbit_type  = make_two_translation_orbit( 2, 0, { 1, 1 } );
    first.candidate_orbit_type = make_two_translation_orbit( 2, 0, { 1, 3 } );
    first.previous_stabilizer  = { "translation_lattice", "torus_2d", 2, { 1, 0, 0, 1 }, {} };
    first.candidate_stabilizer = { "translation_lattice", "torus_2d", 2, { 1, 2, 0, 3 }, {} };
    container::symmetry_event_journal<real> journal( directory / "multidimensional_events.dat" );
    journal.stage( first );
    require_true( journal.commit() && journal.load(), "multidimensional event journal round trip" );
    require_true(
        journal.all().front().candidate_orbit_type == first.candidate_orbit_type,
        "journal preserves dimension and finite invariants"
    );
    require_true(
        journal.all().front().candidate_stabilizer == first.candidate_stabilizer, "journal preserves subgroup embedding"
    );
    auto different                               = first;
    different.curve_number                       = 2;
    different.candidate_stabilizer.lattice_basis = { 1, 1, 0, 3 };
    container::symmetry_event_registry<real> registry( directory / "embedded_events.json" );
    registry.rebuild( { first, different }, 1e-3, 1e-3, []( const auto &, const auto & ) { return 0.; } );
    require_true( registry.all().size() == 2, "different embeddings with identical orbit invariants remain separate" );
    require_true( registry.save() && registry.load(), "embedded registry round trip" );
    require_true(
        registry.all()[0].state_reference.candidate_stabilizer == first.candidate_stabilizer,
        "registry incident preserves embedding"
    );
    const auto old_path = directory / "version2_events.dat";
    {
        std::ofstream stream( old_path );
        stream << "# symmetry_events_version 2\n10 28.1 1 11 1 1 1 3 1e-3 1e-12 2\n";
    }
    container::symmetry_event_journal<real> legacy( old_path );
    require_true( legacy.load() && legacy.all().front().candidate_order == 3, "version 2 journal remains readable" );
    require_true( legacy.all().front().candidate_stabilizer.empty(), "legacy unknown embedding is not invented" );
    {
        std::ofstream stream( old_path );
        stream << "# symmetry_events_version 999\n";
    }
    require_true( !legacy.load() && legacy.all().empty(), "unknown journal versions fail even without records" );
    const auto finite                = make_record( 3, 20, 29., 2 );
    auto       finite_copy           = finite;
    finite_copy.previous_stabilizer  = { "finite_actions", "reflection_group", 0, {}, { 0 } };
    finite_copy.candidate_stabilizer = { "finite_actions", "reflection_group", 0, {}, { 0, 2 } };
    container::symmetry_event_journal<real> finite_journal( directory / "finite_events.dat" );
    finite_journal.stage( finite_copy );
    require_true(
        finite_journal.commit() && finite_journal.load() &&
            finite_journal.all().front().candidate_stabilizer == finite_copy.candidate_stabilizer,
        "finite action metadata round trip"
    );
}

} // namespace

int main()
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "deflated_continuation_symmetry_event_test";
    std::error_code cleanup_error;
    std::filesystem::remove_all( directory, cleanup_error );
    std::filesystem::create_directories( directory );

    test_journal_is_transactional( directory );
    test_legacy_event_file_is_read( directory );
    test_registry_links_incident_endpoints( directory );
    test_registry_preserves_general_orbit_types( directory );
    test_stabilizer_embeddings_and_journal_versions( directory );

    std::filesystem::remove_all( directory, cleanup_error );
    std::cout << "Checks: " << checks << ", failures: " << failures << std::endl;
    if ( failures != 0 )
    {
        std::cout << "FAILED" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
