#ifndef PERIODIC_ORBIT_RECORDING_STEPPER_H
#define PERIODIC_ORBIT_RECORDING_STEPPER_H

#include <algorithm>
#include <vector>
#include <periodic_orbit/section_map.h>

namespace periodic_orbit
{
template <class T>
struct flight_step_record
{
    T time{}, dt{}, end{}, theta = T( 1 );
};

// Observe accepted steps through the existing finalize boundary. Recording never
// changes adaptation, the dense interpolant, or the committed endpoint.
template <class Stepper>
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
    const std::vector<flight_step_record<T>> &records() const
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
            const T dt    = get_dt();
            const T theta = outcome == status::accepted ? T( 1 ) : ( time - time_ ) / dt;
            records_.push_back( { time_, dt, time, std::min( T( 1 ), theta ) } );
        }
        step_->finalize( outcome, time, state );
    }

private:
    std::shared_ptr<Stepper>           step_;
    std::size_t                        maximum_steps_;
    T                                  time_{};
    std::vector<flight_step_record<T>> records_;
};

// Identity-state flights only. Constraint reconstruction needs its own prepared
// derivative and is deliberately not inferred from an ODE rate provider.
template <class Space, class Stepper, class Section, class RateProvider>
class recorded_section_map
{
public:
    using recorder_type = recording_stepper<Stepper>;
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
