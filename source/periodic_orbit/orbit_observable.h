#ifndef PERIODIC_ORBIT_ORBIT_OBSERVABLE_H
#define PERIODIC_ORBIT_ORBIT_OBSERVABLE_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <nmfd/time_steppers/detail/status.h>

namespace periodic_orbit
{
template <class T>
struct orbit_observable_summary
{
    T rms, minimum, maximum;
};

// Read-only manager for one verification pass. Discard it if the pass fails.
// Trapezoidal quadrature of q^2 includes the shortened last step.
template <class T, class Evaluate>
class orbit_observable
{
public:
    explicit orbit_observable( Evaluate evaluate ) : evaluate_( std::move( evaluate ) ) {}
    void set_time_interval( T first, T last ) { dt_ = std::abs( last - first ); }
    template <class State>
    bool apply( const State &current, const State &next )
    {
        if ( !(dt_ > 0) || !std::isfinite( dt_ ) )
            throw std::domain_error( "Invalid diagram sampling interval" );
        if ( duration_ == 0 )
            minimum_ = maximum_ = previous_ = sample( current );
        const T value = sample( next );
        minimum_ = std::min( minimum_, value );
        // Scale by the largest magnitude seen so finite magnitudes need not have representable squares.
        if ( value > maximum_ )
        {
            const T ratio = maximum_ / value;
            integral_ *= ratio * ratio;
            maximum_ = value;
        }
        if ( maximum_ > 0 )
        {
            const T a = previous_ / maximum_, b = value / maximum_;
            integral_ += dt_ * ((a * a + b * b) / 2);
        }
        duration_ += dt_;
        previous_ = value;
        return false;
    }
    auto get_status() const { return nmfd::time_steppers::integration_status::running; }
    orbit_observable_summary<T> summary() const
    {
        if ( !(duration_ > 0) || !std::isfinite( integral_ ) || !std::isfinite( duration_ ) )
            throw std::domain_error( "Empty or nonfinite orbit measurement" );
        return { maximum_ * std::sqrt( integral_ / duration_ ), minimum_, maximum_ };
    }
private:
    template <class State>
    T sample( const State &x )
    {
        const T q = evaluate_( x );
        if ( !std::isfinite( q ) || q < 0 )
            throw std::domain_error( "Time-RMS diagram requires a finite nonnegative magnitude" );
        return q;
    }
    Evaluate evaluate_;
    T dt_{}, duration_{}, integral_{}, previous_{}, minimum_{}, maximum_{};
};
}
#endif
