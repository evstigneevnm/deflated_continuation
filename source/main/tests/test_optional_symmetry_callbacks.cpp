#include <cmath>
#include <iostream>
#include <stdexcept>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <main/deflation_continuation.hpp>

namespace
{

// Missing optional hooks must borrow vectors, never pass them through C varargs.
struct vector
{
    int value                = 0;
    vector()                 = default;
    vector( const vector & ) = delete;
};

struct plain_storage
{
};

struct symmetry_storage
{
    void stabilize_in_place( vector &x )
    {
        ++x.value;
    }
    double nearest_stabilized_distance( vector & )
    {
        return .25;
    }
    double canonical_distance( const vector &a, const vector &b )
    {
        return std::abs( a.value - b.value );
    }
    void align_endpoint_geometry(
        const vector &, const vector &source, const vector &tangent, vector &aligned, vector &aligned_tangent
    )
    {
        aligned.value         = source.value;
        aligned_tangent.value = tangent.value;
    }
};

void require( bool condition )
{
    if ( !condition )
        throw std::runtime_error( "Optional symmetry hook selected the wrong overload" );
}

}

int main()
{
    namespace hooks = main_classes::detail;
    plain_storage    plain;
    symmetry_storage symmetry;
    vector           reference, source, tangent, aligned, aligned_tangent;
    source.value    = 2;
    tangent.value   = 3;
    double distance = 7;
    hooks::stabilize_solution_if_available( &plain, source );
    require( source.value == 2 );
    require( !hooks::nearest_stabilized_distance_if_available( &plain, source, distance ) && distance == 7 );
    require( !hooks::symmetry_orbit_distance_if_available( &plain, reference, source, distance ) && distance == 7 );
    require(
        !hooks::align_endpoint_geometry_if_available(
            &plain, reference, source, tangent, aligned, aligned_tangent, 0
        ) &&
        aligned.value == 0
    );
    hooks::stabilize_solution_if_available( &symmetry, source );
    require( source.value == 3 );
    require( hooks::nearest_stabilized_distance_if_available( &symmetry, source, distance ) && distance == .25 );
    require( hooks::symmetry_orbit_distance_if_available( &symmetry, reference, source, distance ) && distance == 3 );
    require(
        hooks::align_endpoint_geometry_if_available(
            &symmetry, reference, source, tangent, aligned, aligned_tangent, 0
        ) &&
        aligned.value == source.value && aligned_tangent.value == tangent.value
    );
    std::cout << "Optional symmetry hooks: 8 checks passed\n";
}
