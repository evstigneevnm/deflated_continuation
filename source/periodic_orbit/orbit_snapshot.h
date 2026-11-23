#ifndef PERIODIC_ORBIT_ORBIT_SNAPSHOT_H
#define PERIODIC_ORBIT_ORBIT_SNAPSHOT_H

#include <containers/bifurcation_diagram/periodic_branch_archive.h>
#include <periodic_orbit/sections/affine_section.h>
#include <periodic_orbit/sections/section_layout.h>
#include <ostream>
#include <type_traits>

namespace periodic_orbit
{
// V1: binary64 Euclidean affine sections, forward flights, identity closure.
// Vector transport is injected and only invoked during explicit checkpoint I/O.
inline const nlohmann::json &verified_snapshot_point( const nlohmann::json &branch, std::size_t segment, std::size_t index )
{
    for ( const auto &arm : branch.at( "segments" ) )
        if ( arm.at( "id" ) == segment )
            for ( const auto &row : arm.at( "points" ) )
                if ( row.at( "source_index" ) == index && row.at( "verified" ) == true )
                    return row;
    throw std::invalid_argument( "Snapshot does not identify a verified branch point" );
}

namespace snapshot_detail
{
using json = nlohmann::json;
inline void check_vector( const json &v, std::size_t n )
{
    if ( !v.is_array() || v.size() != n )
        throw std::invalid_argument( "Orbit snapshot vector dimension mismatch" );
    for ( const auto &x : v )
        if ( !x.is_number() || !std::isfinite( x.get<double>() ) )
            throw std::invalid_argument( "Nonfinite orbit snapshot vector" );
}

inline void validate( const json &s, const json &branch, const json &config, std::size_t dimension )
{
    if ( s.at( "format" ) != "affine_orbit_snapshot" || s.at( "version" ) != 1 ||
         s.at( "configuration" ) != config || s.at( "dimension" ) != dimension || dimension < 2 ||
         s.at( "branch" ) != branch.at( "id" ) || branch.at( "closure" ) != "identity" ||
         branch.at( "kind" ) != "periodic" || s.at( "model" ) != branch.at( "model" ) ||
         s.at( "discretization" ) != branch.at( "discretization" ) || s.at( "method" ) != branch.at( "method" ) ||
         s.at( "parameter_name" ) != branch.at( "parameter" ) ||
         !s.at( "segment" ).is_number_unsigned() || !s.at( "source_index" ).is_number_unsigned() )
        throw std::invalid_argument( "Incompatible periodic orbit snapshot" );
    const auto &row = verified_snapshot_point( branch, s.at( "segment" ), s.at( "source_index" ) );
    const auto &nodes = s.at( "nodes" );
    if ( row.at( "parameter" ) != s.at( "parameter" ) || row.at( "period" ) != s.at( "period" ) ||
         !nodes.is_array() || nodes.empty() || nodes.size() != branch.at( "sections" ) )
        throw std::invalid_argument( "Snapshot point or layout differs from archive" );
    for ( const auto &node : nodes )
    {
        check_vector( node.at( "center" ), dimension );
        check_vector( node.at( "normal" ), dimension );
        check_vector( node.at( "coordinates" ), dimension - 1 );
        double length = 0;
        for ( const auto &n : node.at( "normal" ) )
            length = std::hypot( length, n.get<double>() );
        const double flight = node.at( "expected_flight" );
        if ( std::abs( length - 1 ) > 1e-9 || !std::isfinite( node.at( "pilot_time" ).get<double>() ) ||
             !std::isfinite( flight ) || flight <= 0 || !node.at( "expected_steps" ).is_number_unsigned() ||
             node.at( "expected_steps" ) == 0 || !node.at( "pilot_step" ).is_number_unsigned() )
            throw std::invalid_argument( "Invalid orbit snapshot section geometry" );
    }
}
}

template <class Space, class State, class Download>
void write_orbit_snapshot( std::ostream &out, const nlohmann::json &branch, std::size_t segment, std::size_t index,
                           const nlohmann::json &config, std::size_t dimension,
                           const section_layout<affine_section<Space>> &layout, const State &z, Download download )
{
    static_assert( std::is_same_v<typename Space::scalar_type, double> );
    if ( layout.closing_target() != layout.at( 0 ).section || z.size() != layout.size() )
        throw std::invalid_argument( "Snapshot requires identity closure and matching coordinates" );
    const auto &point = verified_snapshot_point( branch, segment, index );
    nlohmann::json s = { { "format", "affine_orbit_snapshot" }, { "version", 1 }, { "branch", branch.at( "id" ) },
        { "model", branch.at( "model" ) }, { "discretization", branch.at( "discretization" ) },
        { "method", branch.at( "method" ) }, { "parameter_name", branch.at( "parameter" ) }, { "segment", segment },
        { "source_index", index }, { "parameter", point.at( "parameter" ) }, { "period", point.at( "period" ) },
        { "dimension", dimension }, { "configuration", config }, { "nodes", nlohmann::json::array() } };
    for ( std::size_t i = 0; i < layout.size(); ++i )
    {
        const auto &node = layout.at( i );
        s["nodes"].push_back( { { "center", download( node.section->center() ) },
            { "normal", download( node.section->normal() ) }, { "coordinates", download( z.comp( i ) ) },
            { "pilot_time", node.pilot_time }, { "expected_flight", node.expected_flight },
            { "pilot_step", node.pilot_step }, { "expected_steps", node.expected_steps } } );
    }
    snapshot_detail::validate( s, branch, config, dimension );
    if ( !(out << s << '\n') )
        throw std::runtime_error( "Cannot write orbit snapshot" );
}

// Restore into a fresh physical space and fresh coordinate workspace; do not
// overwrite any active continuation state. The caller verifies the orbit before use.
template <class Space, class State, class Upload>
auto restore_orbit_snapshot( std::istream &in, const nlohmann::json &branch, const nlohmann::json &config,
                             std::size_t segment, std::size_t index, std::shared_ptr<Space> space,
                             State &z, Upload upload )
{
    using plane = affine_section<Space>;
    using layout = section_layout<plane>;
    static_assert( std::is_same_v<typename Space::scalar_type, double> );
    if ( !space )
        throw std::invalid_argument( "Missing physical space for orbit snapshot" );
    const auto s = nlohmann::json::parse( in );
    snapshot_detail::validate( s, branch, config, space->get_vector_size() );
    if ( s.at( "segment" ) != segment || s.at( "source_index" ) != index || z.size() != s.at( "nodes" ).size() )
        throw std::invalid_argument( "Wrong orbit snapshot requested" );
    for ( std::size_t i = 0; i < z.size(); ++i )
        if ( space->get_size( z.comp( i ) ) != space->get_vector_size() - 1 )
            throw std::invalid_argument( "Wrong coordinate space for orbit snapshot" );
    nmfd::detail::vector_wrap<Space, true, true> center( *space ), normal( *space );
    std::vector<typename layout::node> nodes;
    for ( const auto &node : s.at( "nodes" ) )
    {
        upload( node.at( "center" ), *center );
        upload( node.at( "normal" ), *normal );
        upload( node.at( "coordinates" ), z.comp( nodes.size() ) );
        nodes.push_back( { std::make_shared<const plane>( space, *center, *normal ), node.at( "pilot_time" ),
                           node.at( "expected_flight" ), node.at( "pilot_step" ), node.at( "expected_steps" ) } );
    }
    auto closing = nodes.front().section;
    return std::make_shared<const layout>( std::move( nodes ), std::move( closing ) );
}
}
#endif
