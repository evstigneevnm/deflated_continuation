#ifndef __SYMMETRY_TRANSLATION_INTEGER_LATTICE_H__
#define __SYMMETRY_TRANSLATION_INTEGER_LATTICE_H__

#include <cstddef>
#include <cstdint>
#include <limits>

#include <scfd/utils/device_tag.h>

namespace symmetry
{
namespace translation
{

// Fixed-dimensional row Hermite form. Merging computes the sum of two lattices.
template <std::size_t Dimension>
struct integer_lattice
{
    static_assert( Dimension > 0 && Dimension <= 3, "Translation lattices support dimensions one through three" );
    using integer_type                          = std::int64_t;
    static constexpr integer_type minimum_value = std::numeric_limits<integer_type>::min();
    static constexpr integer_type maximum_value = std::numeric_limits<integer_type>::max();
    integer_type                  basis[Dimension][Dimension]{};
    bool                          overflow = false;

    __DEVICE_TAG__ static integer_type magnitude( integer_type value )
    {
        return value < 0 ? -value : value;
    }

    __DEVICE_TAG__ static integer_type gcd( integer_type a, integer_type b )
    {
        a = magnitude( a );
        b = magnitude( b );
        while ( b != 0 )
        {
            const auto remainder = a % b;
            a                    = b;
            b                    = remainder;
        }
        return a;
    }

    __DEVICE_TAG__ bool subtract_product( integer_type &value, integer_type factor, integer_type term )
    {
        constexpr auto maximum = maximum_value;
        if ( factor != 0 && magnitude( term ) > maximum / magnitude( factor ) )
        {
            overflow = true;
            return false;
        }
        const auto product = factor * term;
        if ( ( product > 0 && value < -maximum + product ) || ( product < 0 && value > maximum + product ) )
        {
            overflow = true;
            return false;
        }
        value -= product;
        return true;
    }

    __DEVICE_TAG__ void insert( const integer_type *input )
    {
        if ( overflow )
            return;
        integer_type row[Dimension];
        for ( std::size_t j = 0; j < Dimension; ++j )
        {
            if ( input[j] == minimum_value )
            {
                overflow = true;
                return;
            }
            row[j] = input[j];
        }
        for ( std::size_t column = 0; column < Dimension; ++column )
        {
            while ( row[column] != 0 )
            {
                if ( basis[column][column] == 0 )
                {
                    const integer_type sign = row[column] < 0 ? -1 : 1;
                    for ( std::size_t j = column; j < Dimension; ++j )
                    {
                        basis[column][j] = sign * row[j];
                        row[j]           = 0;
                    }
                    break;
                }
                const auto quotient = row[column] / basis[column][column];
                for ( std::size_t j = column; j < Dimension; ++j )
                    if ( !subtract_product( row[j], quotient, basis[column][j] ) )
                        return;
                if ( row[column] != 0 )
                {
                    const integer_type sign = row[column] < 0 ? -1 : 1;
                    for ( std::size_t j = column; j < Dimension; ++j )
                    {
                        const auto old   = basis[column][j];
                        basis[column][j] = sign * row[j];
                        row[j]           = old;
                    }
                }
            }
        }
        for ( std::size_t column = 0; column < Dimension; ++column )
        {
            const auto pivot = basis[column][column];
            if ( pivot == 0 )
                continue;
            for ( std::size_t i = 0; i < column; ++i )
            {
                auto quotient = basis[i][column] / pivot;
                if ( basis[i][column] % pivot < 0 )
                    --quotient;
                for ( std::size_t j = column; j < Dimension; ++j )
                    if ( !subtract_product( basis[i][j], quotient, basis[column][j] ) )
                        return;
            }
        }
    }

    __DEVICE_TAG__ bool contains( const integer_type *input ) const
    {
        auto         work = *this;
        integer_type row[Dimension];
        for ( std::size_t j = 0; j < Dimension; ++j )
        {
            if ( input[j] == minimum_value )
                return false;
            row[j] = input[j];
        }
        for ( std::size_t column = 0; column < Dimension; ++column )
        {
            if ( row[column] == 0 )
                continue;
            const auto pivot = basis[column][column];
            if ( pivot == 0 || row[column] % pivot != 0 )
                return false;
            const auto quotient = row[column] / pivot;
            for ( std::size_t j = column; j < Dimension; ++j )
                if ( !work.subtract_product( row[j], quotient, basis[column][j] ) )
                    return false;
        }
        return !overflow;
    }

    __DEVICE_TAG__ bool contains( const integer_lattice &other ) const
    {
        if ( overflow || other.overflow )
            return false;
        for ( std::size_t i = 0; i < Dimension; ++i )
            if ( !contains( other.basis[i] ) )
                return false;
        return true;
    }

    __DEVICE_TAG__ std::size_t rank() const
    {
        std::size_t result = 0;
        for ( std::size_t i = 0; i < Dimension; ++i )
            result += basis[i][i] != 0;
        return result;
    }
};

template <std::size_t Dimension>
struct lattice_sum
{
    __DEVICE_TAG__ integer_lattice<Dimension>
                   operator()( integer_lattice<Dimension> left, const integer_lattice<Dimension> &right ) const
    {
        left.overflow = left.overflow || right.overflow;
        for ( std::size_t i = 0; i < Dimension && !left.overflow; ++i )
            left.insert( right.basis[i] );
        return left;
    }
};

}
}

#endif
