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
#include <containers/symmetry_event_codec.h>

namespace container
{

template <class T>
class symmetry_event_registry
{
public:
    using record_type = symmetry_event_record<T>;

    struct event_node
    {
        std::string                       id;
        T                                 lambda     = T( 0 );
        T                                 lambda_min = T( 0 );
        T                                 lambda_max = T( 0 );
        symmetry::translation::orbit_type previous_orbit_type;
        symmetry::translation::orbit_type candidate_orbit_type;
        symmetry::stabilizer_description  previous_stabilizer;
        symmetry::stabilizer_description  candidate_stabilizer;
        record_type                       state_reference;
        std::vector<record_type>          incidents;
    };

    explicit symmetry_event_registry( std::filesystem::path file_name_ = {} ) : file_name( std::move( file_name_ ) )
    {
    }

    template <class StateDistance>
    void rebuild(
        std::vector<record_type> records, const T lambda_tolerance, const T state_tolerance,
        StateDistance &&state_distance
    )
    {
        std::sort( records.begin(), records.end(), []( const record_type &left, const record_type &right ) {
            if ( left.curve_number != right.curve_number )
            {
                return left.curve_number < right.curve_number;
            }
            return left.point_index < right.point_index;
        } );

        events.clear();
        for ( const auto &record : records )
        {
            event_node *match = nullptr;
            for ( auto &event : events )
            {
                if ( event.previous_orbit_type != record.previous_orbit_type ||
                     event.candidate_orbit_type != record.candidate_orbit_type ||
                     event.previous_stabilizer != record.previous_stabilizer ||
                     event.candidate_stabilizer != record.candidate_stabilizer ||
                     scalar_abs( event.lambda - record.lambda ) > lambda_tolerance )
                {
                    continue;
                }
                if ( !event.state_reference.vector_available || !record.vector_available )
                {
                    continue;
                }
                const T distance = state_distance( event.state_reference, record );
                if ( distance <= state_tolerance )
                {
                    match = &event;
                    break;
                }
            }

            if ( match == nullptr )
            {
                event_node event;
                event.id                   = make_id( record );
                event.lambda               = record.lambda;
                event.lambda_min           = record.lambda;
                event.lambda_max           = record.lambda;
                event.previous_orbit_type  = record.previous_orbit_type;
                event.candidate_orbit_type = record.candidate_orbit_type;
                event.previous_stabilizer  = record.previous_stabilizer;
                event.candidate_stabilizer = record.candidate_stabilizer;
                event.state_reference      = record;
                event.incidents.push_back( record );
                events.push_back( std::move( event ) );
            }
            else
            {
                const T incident_count = static_cast<T>( match->incidents.size() );
                match->lambda     = ( incident_count * match->lambda + record.lambda ) / ( incident_count + T( 1 ) );
                match->lambda_min = std::min( match->lambda_min, record.lambda );
                match->lambda_max = std::max( match->lambda_max, record.lambda );
                match->incidents.push_back( record );
            }
        }
    }

    bool save() const
    {
        if ( file_name.empty() )
        {
            return false;
        }
        nlohmann::json root;
        root["version"] = 3;
        root["events"]  = nlohmann::json::array();
        for ( const auto &event : events )
        {
            nlohmann::json encoded;
            encoded["id"]                   = event.id;
            encoded["lambda"]               = static_cast<double>( event.lambda );
            encoded["lambda_min"]           = static_cast<double>( event.lambda_min );
            encoded["lambda_max"]           = static_cast<double>( event.lambda_max );
            encoded["previous_orbit_type"]  = encode_orbit_type( event.previous_orbit_type );
            encoded["candidate_orbit_type"] = encode_orbit_type( event.candidate_orbit_type );
            encoded["previous_stabilizer"]  = detail::encode_stabilizer( event.previous_stabilizer );
            encoded["candidate_stabilizer"] = detail::encode_stabilizer( event.candidate_stabilizer );
            encoded["state_reference"]      = encode_record( event.state_reference );
            encoded["incidents"]            = nlohmann::json::array();
            for ( const auto &incident : event.incidents )
            {
                encoded["incidents"].push_back( encode_record( incident ) );
            }
            root["events"].push_back( std::move( encoded ) );
        }

        const std::filesystem::path temporary = file_name.string() + ".tmp";
        std::ofstream               stream( temporary, std::ofstream::out | std::ofstream::trunc );
        if ( !stream )
        {
            return false;
        }
        stream << root.dump( 4 ) << '\n';
        stream.close();
        if ( !stream )
        {
            std::error_code remove_error;
            std::filesystem::remove( temporary, remove_error );
            return false;
        }

        std::error_code rename_error;
        std::filesystem::rename( temporary, file_name, rename_error );
        if ( rename_error )
        {
            std::error_code remove_error;
            std::filesystem::remove( temporary, remove_error );
            return false;
        }
        return true;
    }

    bool load()
    {
        events.clear();
        std::ifstream stream( file_name );
        if ( !stream )
        {
            return false;
        }
        nlohmann::json root;
        try
        {
            stream >> root;
            for ( const auto &encoded : root.value( "events", nlohmann::json::array() ) )
            {
                event_node event;
                event.id         = encoded.value( "id", std::string{} );
                event.lambda     = static_cast<T>( encoded.value( "lambda", 0.0 ) );
                event.lambda_min = static_cast<T>( encoded.value( "lambda_min", static_cast<double>( event.lambda ) ) );
                event.lambda_max = static_cast<T>( encoded.value( "lambda_max", static_cast<double>( event.lambda ) ) );
                event.previous_orbit_type =
                    decode_orbit_type( encoded.value( "previous_orbit_type", nlohmann::json::object() ) );
                event.candidate_orbit_type =
                    decode_orbit_type( encoded.value( "candidate_orbit_type", nlohmann::json::object() ) );
                event.previous_stabilizer =
                    detail::decode_stabilizer( encoded.value( "previous_stabilizer", nlohmann::json::object() ) );
                event.candidate_stabilizer =
                    detail::decode_stabilizer( encoded.value( "candidate_stabilizer", nlohmann::json::object() ) );
                event.state_reference = decode_record( encoded.value( "state_reference", nlohmann::json::object() ) );
                for ( const auto &incident : encoded.value( "incidents", nlohmann::json::array() ) )
                {
                    event.incidents.push_back( decode_record( incident ) );
                }
                events.push_back( std::move( event ) );
            }
        }
        catch ( const std::exception & )
        {
            events.clear();
            return false;
        }
        return true;
    }

    const std::vector<event_node> &all() const
    {
        return events;
    }

private:
    std::filesystem::path   file_name;
    std::vector<event_node> events;

    static T scalar_abs( const T value )
    {
        return value < T( 0 ) ? -value : value;
    }

    static std::string make_id( const record_type &record )
    {
        std::ostringstream stream;
        stream << "symmetry-event-c" << record.curve_number << "-p" << record.point_index;
        return stream.str();
    }

    static nlohmann::json encode_orbit_type( const symmetry::translation::orbit_type &orbit )
    {
        return detail::encode_symmetry_orbit( orbit );
    }

    static symmetry::translation::orbit_type decode_orbit_type( const nlohmann::json &encoded )
    {
        return detail::decode_symmetry_orbit( encoded );
    }

    static nlohmann::json encode_record( const record_type &record )
    {
        return detail::encode_symmetry_record( record );
    }

    static record_type decode_record( const nlohmann::json &encoded )
    {
        return detail::decode_symmetry_record<T>( encoded );
    }
};

} // namespace container

#endif
