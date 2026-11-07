#ifndef PERIODIC_ORBIT_RECORDING_STEPPER_H
#define PERIODIC_ORBIT_RECORDING_STEPPER_H

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>
#include <periodic_orbit/section_map.h>

namespace periodic_orbit
{
template <class T>
struct flight_step_record
{
    T time{}, dt{}, end{}, theta = T( 1 );
    template <class Stepper>
    static flight_step_record capture( const Stepper &, T time, T dt, T end, T theta, T )
    {
        return { time, dt, end, theta };
    }
};

namespace detail
{
template <class Record>
void validate_flight_schedule( const std::vector<Record> &steps, std::size_t maximum_steps )
{
    if ( steps.empty() || steps.size() > maximum_steps )
        throw std::invalid_argument( "Empty or over-budget frozen flight schedule" );
    using T      = std::decay_t<decltype( steps.front().time )>;
    using traits = scfd::utils::scalar_traits<T>;
    for ( std::size_t i = 0; i < steps.size(); ++i )
    {
        const auto &r = steps[i];
        if ( !traits::is_finite( r.time ) || !traits::is_finite( r.dt ) || !traits::is_finite( r.end ) ||
             !traits::is_finite( r.theta ) || r.dt == T( 0 ) || r.theta <= T( 0 ) || r.theta > T( 1 ) ||
             ( i && r.time != steps[i - 1].end ) || ( i + 1 < steps.size() && r.theta != T( 1 ) ) || r.end == r.time ||
             ( r.end > r.time ) != ( r.dt > T( 0 ) ) || ( r.dt > T( 0 ) ) != ( steps.front().dt > T( 0 ) ) ||
             traits::abs( r.end - ( r.time + r.theta * r.dt ) ) >
                 T( 64 ) * std::numeric_limits<T>::epsilon() *
                     std::max( T( 1 ), std::max( traits::abs( r.time ), traits::abs( r.end ) ) ) )
            throw std::invalid_argument( "Invalid frozen flight schedule" );
    }
}
}

// Observe accepted steps through the existing finalize boundary. Recording never
// changes adaptation, the dense interpolant, or the committed endpoint.
template <class Stepper, class Record = flight_step_record<typename Stepper::scalar_type>>
class recording_stepper
{
public:
    using scalar_type = typename Stepper::scalar_type;
    using vector_type = typename Stepper::vector_type;
    using T           = scalar_type;
    recording_stepper( std::shared_ptr<Stepper> step, std::size_t maximum_steps )
        : step_( step ? std::move( step ) : throw std::invalid_argument( "Missing recorded stepper" ) ),
          maximum_steps_( maximum_steps )
    {
        if ( !maximum_steps )
            throw std::invalid_argument( "Empty flight recording budget" );
        records_.reserve( std::min( maximum_steps, std::size_t( 1024 ) ) );
    }
    void reset()
    {
        step_->reset();
        records_.clear();
    }
    void set_time( T time )
    {
        step_->set_time( time );
        time_ = time;
    }
    void set_target_time( T time )
    {
        step_->set_target_time( time );
        target_ = time;
    }
    T get_dt() const
    {
        return step_->get_dt();
    }
    auto get_status() const
    {
        return step_->get_status();
    }
    auto dense_output_order() const
    {
        return step_->dense_output_order();
    }
    auto get_continuous_integration() const
    {
        return step_->get_continuous_integration();
    }
    const auto &table() const
    {
        return step_->table();
    }
    const std::vector<Record> &records() const
    {
        return records_;
    }
    void apply( const vector_type &in, vector_type &out )
    {
        step_->apply( in, out );
    }
    void finalize( nmfd::time_steppers::adaptation_status outcome, T time, const vector_type &state )
    {
        using status = nmfd::time_steppers::adaptation_status;
        if ( outcome == status::accepted || outcome == status::accepted_modified )
        {
            if ( records_.size() == maximum_steps_ )
                throw std::length_error( "Flight recording budget exhausted" );
            const T    dt            = get_dt();
            const T    theta         = outcome == status::accepted ? T( 1 ) : ( time - time_ ) / dt;
            const T    candidate_end = dt == target_ - time_ ? target_ : time_ + dt;
            const auto record = Record::capture( *step_, time_, dt, time, std::min( T( 1 ), theta ), candidate_end );
            // Reserve before committing, so a failed allocation/commit cannot
            // append a step that never became part of the accepted trajectory.
            if ( records_.size() == records_.capacity() )
                records_.reserve(
                    std::min( maximum_steps_, records_.size() + std::max( std::size_t( 1 ), records_.size() ) )
                );
            step_->finalize( outcome, time, state );
            records_.push_back( record );
        }
        else
            step_->finalize( outcome, time, state );
    }

private:
    std::shared_ptr<Stepper> step_;
    std::size_t              maximum_steps_;
    T                        time_{}, target_ = T( 1 );
    std::vector<Record>      records_;
};

// Identity-state flights only. Constraint reconstruction needs its own prepared
// derivative and is deliberately not inferred from an ODE rate provider.
template <
    class Space, class Stepper, class Section, class RateProvider,
    class Record = flight_step_record<typename Stepper::scalar_type>>
class recorded_section_map
{
public:
    using recorder_type = recording_stepper<Stepper, Record>;
    using map_type      = section_map<Space, recorder_type, Section, identity_state_policy<Space>, RateProvider>;
    using T             = typename Space::norm_type;
    using X             = typename Space::vector_type;
    using request_type  = typename map_type::request_type;
    recorded_section_map(
        std::shared_ptr<Space> space, std::shared_ptr<Stepper> step, std::shared_ptr<RateProvider> rate,
        const typename map_type::params &p = {}
    )
        : step_( std::make_shared<recorder_type>( std::move( step ), p.maximum_steps ) ),
          map_( space, step_, std::make_shared<identity_state_policy<Space>>( space ), p, std::move( rate ) )
    {
    }
    map_report<T> evaluate( const X &in, const request_type &request, X &out )
    {
        return map_.evaluate( in, request, out );
    }
    const auto &records() const
    {
        return step_->records();
    }
    const auto &table() const
    {
        return step_->table();
    }

private:
    std::shared_ptr<recorder_type> step_;
    map_type                       map_;
};
}
#endif
