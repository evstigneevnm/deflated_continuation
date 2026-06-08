#ifndef __SYMMETRY_SLICE_DATA_H__
#define __SYMMETRY_SLICE_DATA_H__

#include <array>
#include <cstddef>
#include <stdexcept>

namespace symmetry
{

template <class Real, std::size_t MaxRank>
struct slice_data
{
    static_assert( MaxRank > 0, "slice_data requires MaxRank > 0" );

    using real_type = Real;

    std::size_t group_dimension = 0;
    std::size_t active_rank = 0;
    std::size_t residual_group_order_value = 1;
    Real tolerance = Real{};
    std::array<Real, MaxRank> active_shift{};

    bool active() const
    {
        return active_rank != 0;
    }

    std::size_t residual_group_order() const
    {
        return active() ? residual_group_order_value : std::size_t( 1 );
    }

    void set_shift( const std::size_t component, const Real value )
    {
        if ( component >= MaxRank )
            throw std::out_of_range( "symmetry::slice_data::set_shift" );
        active_shift[component] = value;
    }

    Real shift( const std::size_t component ) const
    {
        if ( component >= MaxRank )
            throw std::out_of_range( "symmetry::slice_data::shift" );
        return active_shift[component];
    }
};

} // namespace symmetry

#endif
