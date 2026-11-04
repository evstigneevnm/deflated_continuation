#ifndef PERIODIC_ORBIT_EVENT_LOCATOR_H
#define PERIODIC_ORBIT_EVENT_LOCATOR_H

#include <algorithm>
#include <memory>
#include <nmfd/detail/vector_wrap.h>
#include <periodic_orbit/events/event_types.h>

namespace periodic_orbit
{
template <class Space>
class event_locator
{
public:
    using T            = typename Space::norm_type;
    using X            = typename Space::vector_type;
    using traits       = scfd::utils::scalar_traits<T>;
    using storage_type = nmfd::detail::vector_wrap<Space, false, false>;
    event_locator( std::shared_ptr<Space> space, const event_params<T> &params = {} )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing event space" ) ),
          params_( params ), sample_( *space_ )
    {
        params_.validate();
        sample_.init();
        sample_.start_use();
    }

    template <class DenseView, class Event, class StatePolicy>
    event_report<T>
    locate( const DenseView &dense, const event_bracket<T> &bracket, Event &event, StatePolicy &policy, X &out )
    {
        event_report<T> report;
        auto            lo = bracket.lower, hi = bracket.upper, fl = bracket.lower_value, fh = bracket.upper_value;
        if ( !traits::is_finite( bracket.step_start ) || !traits::is_finite( bracket.step_size ) ||
             bracket.step_size == T( 0 ) || !traits::is_finite( lo ) || !traits::is_finite( hi ) || lo < T( 0 ) ||
             hi > T( 1 ) || lo >= hi || !traits::is_finite( fl ) || !traits::is_finite( fh ) ||
             ( ( fl > T( 0 ) ) == ( fh > T( 0 ) ) && fl != T( 0 ) && fh != T( 0 ) ) )
        {
            report.status = section_status::invalid_input;
            return report;
        }
        for ( std::size_t iteration = 0; iteration < params_.maximum_iterations; ++iteration )
        {
            T theta;
            if ( fl == T( 0 ) )
                theta = lo;
            else if ( fh == T( 0 ) )
                theta = hi;
            else
            {
                theta               = lo + ( hi - lo ) / T( 2 );
                const T denominator = fh - fl;
                if ( traits::is_finite( denominator ) && denominator != T( 0 ) )
                {
                    const T secant = lo - fl * ( hi - lo ) / denominator;
                    if ( secant > lo + T( .1 ) * ( hi - lo ) && secant < hi - T( .1 ) * ( hi - lo ) )
                        theta = secant;
                }
            }
            report.time       = dense.evaluate( theta, *sample_ );
            report.iterations = iteration + 1;
            if ( !traits::is_finite( report.time ) || !space_->is_valid_number( *sample_ ) )
            {
                report.status = section_status::invalid_state;
                return report;
            }
            if ( !policy.reconstruct( report.time, *sample_ ) )
            {
                report.status = section_status::reconstruction_failure;
                return report;
            }
            if ( !space_->is_valid_number( *sample_ ) )
            {
                report.status = section_status::invalid_state;
                return report;
            }
            report.value = event.value( report.time, *sample_ );
            if ( !traits::is_finite( report.value ) )
            {
                report.status = section_status::nonfinite_event;
                return report;
            }
            const T width     = traits::abs( ( hi - lo ) * bracket.step_size );
            const T tolerance = params_.time_tolerance + params_.relative_time_tolerance * traits::abs( report.time );
            if ( report.value == T( 0 ) ||
                 ( traits::abs( report.value ) <= params_.value_tolerance && width <= tolerance ) )
            {
                space_->assign( *sample_, out );
                report.status = section_status::located;
                return report;
            }
            if ( theta == lo || theta == hi )
                break;
            if ( ( report.value > T( 0 ) ) == ( fl > T( 0 ) ) )
            {
                lo = theta;
                fl = report.value;
            }
            else
            {
                hi = theta;
                fh = report.value;
            }
        }
        report.status = section_status::refinement_limit;
        return report;
    }

private:
    std::shared_ptr<Space> space_;
    event_params<T>        params_;
    storage_type           sample_;
};
}
#endif
