#ifndef __SYMMETRY_STABILIZER_DESCRIPTION_H__
#define __SYMMETRY_STABILIZER_DESCRIPTION_H__

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <symmetry/translation/integer_lattice.h>
#include <symmetry/translation/orbit_type.h>

namespace symmetry
{

struct stabilizer_description
{
    std::string               representation;
    std::string               group_fingerprint;
    std::size_t               dimension = 0;
    std::vector<std::int64_t> lattice_basis;
    std::vector<std::size_t>  finite_elements;

    bool empty() const
    {
        return representation.empty();
    }
};

inline bool operator==( const stabilizer_description &left, const stabilizer_description &right )
{
    return left.representation == right.representation && left.group_fingerprint == right.group_fingerprint &&
           left.dimension == right.dimension && left.lattice_basis == right.lattice_basis &&
           left.finite_elements == right.finite_elements;
}

inline bool operator!=( const stabilizer_description &left, const stabilizer_description &right )
{
    return !( left == right );
}

template <std::size_t Dimension>
stabilizer_description
describe_lattice( const translation::integer_lattice<Dimension> &lattice, const std::string &fingerprint )
{
    if ( lattice.overflow )
        throw std::overflow_error( "Translation lattice exceeds the supported integer range" );
    stabilizer_description result;
    result.representation    = "translation_lattice";
    result.group_fingerprint = fingerprint;
    result.dimension         = Dimension;
    for ( const auto &row : lattice.basis )
        result.lattice_basis.insert( result.lattice_basis.end(), row, row + Dimension );
    return result;
}

// Determinantal divisors give the finite invariants, including for rank-deficient lattices.
template <std::size_t Dimension>
translation::orbit_type describe_orbit( const translation::integer_lattice<Dimension> &lattice )
{
    using lattice_type           = translation::integer_lattice<Dimension>;
    auto                    work = lattice;
    translation::orbit_type result;
    result.group_dimension               = Dimension;
    result.active_rank                   = lattice.rank();
    result.continuous_isotropy_dimension = Dimension - result.active_rank;
    std::int64_t divisors[3]{};
    for ( const auto &row : lattice.basis )
        for ( const auto value : row )
            divisors[0] = lattice_type::gcd( divisors[0], value );
    if constexpr ( Dimension >= 2 )
        for ( std::size_t a = 0; a < Dimension; ++a )
            for ( std::size_t b = a + 1; b < Dimension; ++b )
                for ( std::size_t c = 0; c < Dimension; ++c )
                    for ( std::size_t d = c + 1; d < Dimension; ++d )
                    {
                        std::int64_t minor = 0;
                        work.subtract_product( minor, -lattice.basis[a][c], lattice.basis[b][d] );
                        work.subtract_product( minor, lattice.basis[a][d], lattice.basis[b][c] );
                        divisors[1] = lattice_type::gcd( divisors[1], minor );
                    }
    if constexpr ( Dimension == 3 )
    {
        divisors[2] = 1;
        for ( std::size_t i = 0; i < Dimension; ++i )
        {
            std::int64_t product = 0;
            work.subtract_product( product, -divisors[2], lattice.basis[i][i] );
            divisors[2] = product;
        }
    }
    if ( work.overflow )
        throw std::overflow_error( "Translation orbit invariants exceed the supported integer range" );
    std::int64_t previous = 1;
    for ( std::size_t i = 0; i < result.active_rank; ++i )
    {
        result.finite_invariants.push_back( static_cast<std::uint64_t>( divisors[i] / previous ) );
        previous = divisors[i];
    }
    return result;
}

}

#endif
