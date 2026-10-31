#ifndef __CONTAINERS_SYMMETRY_EVENT_CODEC_H__
#define __CONTAINERS_SYMMETRY_EVENT_CODEC_H__

#include <stdexcept>

#include <contrib/json/nlohmann/json.hpp>
#include <containers/symmetry_event_record.h>

namespace container
{
namespace detail
{

inline nlohmann::json encode_symmetry_orbit( const symmetry::translation::orbit_type &orbit )
{
    return {
        { "group_dimension", orbit.group_dimension },
        { "active_rank", orbit.active_rank },
        { "continuous_isotropy_dimension", orbit.continuous_isotropy_dimension },
        { "finite_invariants", orbit.finite_invariants }
    };
}

inline symmetry::translation::orbit_type decode_symmetry_orbit( const nlohmann::json &value )
{
    symmetry::translation::orbit_type result;
    result.group_dimension               = value.value( "group_dimension", std::size_t( 0 ) );
    result.active_rank                   = value.value( "active_rank", std::size_t( 0 ) );
    result.continuous_isotropy_dimension = value.value( "continuous_isotropy_dimension", std::size_t( 0 ) );
    result.finite_invariants             = value.value( "finite_invariants", std::vector<std::uint64_t>{} );
    return result;
}

inline nlohmann::json encode_stabilizer( const symmetry::stabilizer_description &value )
{
    return {
        { "representation", value.representation },
        { "group_fingerprint", value.group_fingerprint },
        { "dimension", value.dimension },
        { "lattice_basis", value.lattice_basis },
        { "finite_elements", value.finite_elements }
    };
}

inline symmetry::stabilizer_description decode_stabilizer( const nlohmann::json &value )
{
    symmetry::stabilizer_description result;
    result.representation    = value.value( "representation", std::string{} );
    result.group_fingerprint = value.value( "group_fingerprint", std::string{} );
    result.dimension         = value.value( "dimension", std::size_t( 0 ) );
    result.lattice_basis     = value.value( "lattice_basis", std::vector<std::int64_t>{} );
    result.finite_elements   = value.value( "finite_elements", std::vector<std::size_t>{} );
    if ( result.representation == "translation_lattice" &&
         ( result.dimension == 0 || result.dimension > 3 ||
           result.lattice_basis.size() != result.dimension * result.dimension ) )
        throw std::invalid_argument( "Invalid translation stabilizer description" );
    return result;
}

template <class T>
nlohmann::json encode_symmetry_record( const symmetry_event_record<T> &record )
{
    return {
        { "curve_number", record.curve_number },
        { "point_index", record.point_index },
        { "lambda", static_cast<double>( record.lambda ) },
        { "vector_available", record.vector_available },
        { "vector_file_id", record.vector_file_id },
        { "segment_id", record.segment_id },
        { "semicurve_id", record.semicurve_id },
        { "previous_order", record.previous_order },
        { "candidate_order", record.candidate_order },
        { "previous_orbit_type", encode_symmetry_orbit( record.previous_orbit_type ) },
        { "candidate_orbit_type", encode_symmetry_orbit( record.candidate_orbit_type ) },
        { "previous_stabilizer", encode_stabilizer( record.previous_stabilizer ) },
        { "candidate_stabilizer", encode_stabilizer( record.candidate_stabilizer ) },
        { "previous_transverse_ratio", static_cast<double>( record.previous_transverse_ratio ) },
        { "candidate_transverse_ratio", static_cast<double>( record.candidate_transverse_ratio ) },
        { "refinements", record.refinements }
    };
}

template <class T>
symmetry_event_record<T> decode_symmetry_record( const nlohmann::json &value )
{
    symmetry_event_record<T> record;
    record.curve_number         = value.value( "curve_number", -1 );
    record.point_index          = value.value( "point_index", std::uint64_t( 0 ) );
    record.lambda               = static_cast<T>( value.value( "lambda", 0.0 ) );
    record.vector_available     = value.value( "vector_available", false );
    record.vector_file_id       = value.value( "vector_file_id", std::uint64_t( 0 ) );
    record.segment_id           = value.value( "segment_id", std::uint64_t( 0 ) );
    record.semicurve_id         = value.value( "semicurve_id", std::uint64_t( 0 ) );
    record.previous_order       = value.value( "previous_order", std::size_t( 1 ) );
    record.candidate_order      = value.value( "candidate_order", std::size_t( 1 ) );
    record.previous_orbit_type  = value.contains( "previous_orbit_type" )
                                      ? decode_symmetry_orbit( value["previous_orbit_type"] )
                                      : symmetry::translation::orbit_type::cyclic_1d( record.previous_order );
    record.candidate_orbit_type = value.contains( "candidate_orbit_type" )
                                      ? decode_symmetry_orbit( value["candidate_orbit_type"] )
                                      : symmetry::translation::orbit_type::cyclic_1d( record.candidate_order );
    record.previous_stabilizer  = decode_stabilizer( value.value( "previous_stabilizer", nlohmann::json::object() ) );
    record.candidate_stabilizer = decode_stabilizer( value.value( "candidate_stabilizer", nlohmann::json::object() ) );
    record.previous_transverse_ratio  = static_cast<T>( value.value( "previous_transverse_ratio", 0.0 ) );
    record.candidate_transverse_ratio = static_cast<T>( value.value( "candidate_transverse_ratio", 0.0 ) );
    record.refinements                = value.value( "refinements", 0u );
    return record;
}

}
}

#endif
