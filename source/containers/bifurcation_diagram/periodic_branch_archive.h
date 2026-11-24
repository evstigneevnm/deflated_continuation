#ifndef BIFURCATION_DIAGRAM_PERIODIC_BRANCH_ARCHIVE_H
#define BIFURCATION_DIAGRAM_PERIODIC_BRANCH_ARCHIVE_H

#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace container
{
// Only scalar diagram data. Solution snapshots and the stationary Boost archive
// have separate lifetimes and are not needed to read this document.
struct branch_identity
{
    std::string id, model, discretization, parameter, observable, normalization;
};

struct periodic_measurement
{
    double period, rms, minimum, maximum, closure_error;

    bool valid() const
    {
        return std::isfinite( period ) && period > 0 && std::isfinite( minimum ) && minimum >= 0 &&
               std::isfinite( rms ) && rms >= 0 && std::isfinite( maximum ) && maximum >= minimum &&
               std::isfinite( closure_error ) && closure_error >= 0;
    }
};

inline nlohmann::json periodic_branch_manifest( const branch_identity &id, const std::string &method,
                                                std::size_t sections )
{
    if ( id.id.empty() || id.model.empty() || id.discretization.empty() || id.parameter.empty() ||
         id.observable.empty() || id.normalization.empty() || method.empty() || !sections )
        throw std::invalid_argument( "Incomplete periodic branch identity" );
    return { { "format", "periodic_branch" }, { "version", 1 }, { "kind", "periodic" }, { "id", id.id },
             { "model", id.model }, { "discretization", id.discretization }, { "parameter", id.parameter },
             { "observable", id.observable }, { "normalization", id.normalization }, { "method", method },
             { "sections", sections }, { "closure", "identity" }, { "segments", nlohmann::json::array() } };
}

inline nlohmann::json read_periodic_branch( std::istream &in )
{
    auto data = nlohmann::json::parse( in );
    if ( data.at( "format" ) != "periodic_branch" || data.at( "version" ) != 1 ||
         data.at( "kind" ) != "periodic" || data.at( "closure" ) != "identity" ||
         !data.at( "segments" ).is_array() || !data.at( "trace_complete" ).is_boolean() ||
         !data.at( "all_points_verified" ).is_boolean() )
        throw std::invalid_argument( "Unsupported or unfinished periodic branch archive" );
    for ( const char *key : { "id", "model", "discretization", "parameter", "observable", "normalization", "method" } )
        if ( !data.at( key ).is_string() || data.at( key ) == "" )
            throw std::invalid_argument( "Missing periodic branch identity" );
    if ( !data.at( "sections" ).is_number_unsigned() || data.at( "sections" ) == 0 )
        throw std::invalid_argument( "Invalid section count" );
    if ( data.contains( "discovered_at" ) )
    {
        const auto &source = data.at( "discovered_at" );
        if ( !source.at( "parameter" ).is_number() || !std::isfinite( source.at( "parameter" ).get<double>() ) ||
             !source.at( "root_index" ).is_number_unsigned() )
            throw std::invalid_argument( "Invalid periodic discovery source" );
    }
    std::size_t index = 0, segment_id = 0;
    bool verified = true;
    for ( const auto &segment : data.at( "segments" ) )
    {
        if ( segment.at( "id" ) != ++segment_id || !segment.at( "points" ).is_array() )
            throw std::invalid_argument( "Invalid periodic segment" );
        for ( const auto &point : segment.at( "points" ) )
        {
            if ( point.at( "source_index" ) != index++ || !std::isfinite( point.at( "parameter" ).get<double>() ) )
                throw std::invalid_argument( "Invalid periodic point" );
            if ( point.at( "verified" ).get<bool>() )
            {
                const periodic_measurement values{ point.at( "period" ), point.at( "observable_rms" ),
                    point.at( "observable_min" ), point.at( "observable_max" ), point.at( "closure_error" ) };
                if ( !values.valid() )
                    throw std::invalid_argument( "Invalid verified orbit measurement" );
            }
            else
                verified = false;
        }
    }
    if ( data.at( "all_points_verified" ) != (index != 0 && verified) )
        throw std::invalid_argument( "Inconsistent periodic branch verification status" );
    return data;
}
}
#endif
