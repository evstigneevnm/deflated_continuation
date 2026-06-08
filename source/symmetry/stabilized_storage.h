#ifndef __SYMMETRY_STABILIZED_STORAGE_H__
#define __SYMMETRY_STABILIZED_STORAGE_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace symmetry
{

template <class T>
double storage_abs_sq( const T &value )
{
    const double v = static_cast<double>( value );
    return v * v;
}

template <class T>
double storage_abs_sq( const std::complex<T> &value )
{
    return static_cast<double>( std::norm( value ) );
}

template <class Vector>
struct l2_storage_distance;

template <class Scalar>
struct l2_storage_distance<std::vector<Scalar>>
{
    using real_type = double;

    real_type operator()( const std::vector<Scalar> &left, const std::vector<Scalar> &right ) const
    {
        if ( left.size() != right.size() )
            throw std::invalid_argument( "l2_storage_distance vector sizes do not match" );
        double sum = 0.0;
        for ( std::size_t i = 0; i < left.size(); ++i )
            sum += storage_abs_sq( left[i] - right[i] );
        return std::sqrt( sum );
    }
};

template <class State, class Distance = l2_storage_distance<State>>
class stabilized_storage
{
public:
    using state_type = State;
    using distance_type = Distance;
    using real_type = decltype( std::declval<Distance>()( std::declval<const State &>(), std::declval<const State &>() ) );

    explicit stabilized_storage( real_type duplicate_tolerance, Distance distance = Distance{} )
        : duplicate_tolerance_( duplicate_tolerance ),
          distance_( std::move( distance ) )
    {
    }

    std::size_t size() const
    {
        return states_.size();
    }

    bool empty() const
    {
        return states_.empty();
    }

    void clear()
    {
        states_.clear();
    }

    const State &operator[]( const std::size_t index ) const
    {
        return states_.at( index );
    }

    const std::vector<State> &states() const
    {
        return states_;
    }

    real_type duplicate_tolerance() const
    {
        return duplicate_tolerance_;
    }

    void add( const State &state )
    {
        states_.push_back( state );
    }

    bool add_if_new( const State &state )
    {
        if ( contains_near( state ) )
            return false;
        add( state );
        return true;
    }

    bool contains_near( const State &state ) const
    {
        return nearest_distance( state ).first <= duplicate_tolerance_;
    }

    std::pair<real_type, std::size_t> nearest_distance( const State &state ) const
    {
        if ( states_.empty() )
            return { std::numeric_limits<real_type>::infinity(), static_cast<std::size_t>( -1 ) };

        real_type best = std::numeric_limits<real_type>::infinity();
        std::size_t best_index = 0;
        for ( std::size_t i = 0; i < states_.size(); ++i )
        {
            const real_type dist = distance_( state, states_[i] );
            if ( dist < best )
            {
                best = dist;
                best_index = i;
            }
        }
        return { best, best_index };
    }

private:
    real_type duplicate_tolerance_;
    Distance distance_;
    std::vector<State> states_;
};

} // namespace symmetry

#endif
