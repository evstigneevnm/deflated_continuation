#ifndef PERIODIC_ORBIT_SECTION_MAP_H
#define PERIODIC_ORBIT_SECTION_MAP_H

#include <nmfd/time_steppers/integration/time_integrator.h>
#include <periodic_orbit/events/section_manager.h>

namespace periodic_orbit
{
template <
    class Space, class Stepper, class Event, class StatePolicy = identity_state_policy<Space>,
    class RateProvider = no_section_rate>
class section_map
{
public:
    using T               = typename Space::norm_type;
    using X               = typename Space::vector_type;
    using manager_type    = section_manager<Space, Stepper, Event, StatePolicy, RateProvider>;
    using integrator_type = nmfd::time_steppers::integration::time_integrator<Space, Stepper, manager_type>;
    using request_type    = flight_request<T, Event>;
    struct params
    {
        std::size_t     maximum_steps = 1000000;
        event_params<T> event;
    };

    section_map(
        std::shared_ptr<Space> space, std::shared_ptr<Stepper> step, std::shared_ptr<StatePolicy> policy,
        const params &p = {}, std::shared_ptr<RateProvider> rate = nullptr
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Missing flight space" ) ),
          step_( step ? std::move( step ) : throw std::invalid_argument( "Missing flight stepper" ) ),
          policy_( policy ? std::move( policy ) : throw std::invalid_argument( "Missing flight state policy" ) ),
          params_( p ),
          manager_( std::make_shared<manager_type>( space_, step_, policy_, p.event, std::move( rate ) ) ),
          integrator_( space_, step_, typename integrator_type::params{ p.maximum_steps }, manager_ ),
          trajectory_( *space_ ), work_( space_ )
    {
        if ( !step_->dense_output_order() )
            throw std::invalid_argument( "Section flights require dense output" );
        trajectory_.init();
        trajectory_.start_use();
    }

    // Output is published only for a successful flight, including when in and out alias.
    map_report<T> evaluate( const X &in, const request_type &request, X &out )
    {
        manager_->reset( request );
        map_report<T> report;
        report.time = request.start_time;
        if ( space_->get_size( in ) != space_->get_vector_size() ||
             space_->get_size( out ) != space_->get_vector_size() )
        {
            report.status = section_status::invalid_input;
            return report;
        }
        if ( !space_->is_valid_number( in ) || !policy_->validate_initial( request.start_time, in ) )
        {
            report.status = section_status::invalid_state;
            return report;
        }
        auto event   = request.target->bind( work_ );
        report.value = event.value( request.start_time, in );
        if ( !scfd::utils::scalar_traits<T>::is_finite( report.value ) )
        {
            report.status = section_status::nonfinite_event;
            return report;
        }
        if ( request.rule.initial == initial_event_policy::report_initial &&
             scfd::utils::scalar_traits<T>::abs( report.value ) <= params_.event.value_tolerance )
        {
            step_->reset();
            space_->assign( in, out );
            report.status = section_status::initial_event;
            return report;
        }
        integrator_.set_time_interval( request.start_time, request.end_time );
        try
        {
            integrator_.apply( in, *trajectory_ );
        }
        catch ( ... )
        {
            step_->reset();
            throw;
        }
        static_cast<event_report<T> &>( report ) = manager_->report();
        report.steps                             = integrator_.get_steps();
        if ( integrator_.get_status() == nmfd::time_steppers::integration_status::stopped_by_external_operation &&
             report.succeeded() )
        {
            report.flight_time = report.time - request.start_time;
            space_->assign( *trajectory_, out );
        }
        else
        {
            if ( report.status == section_status::idle || report.status == section_status::not_found ||
                 report.succeeded() )
                report.status = integrator_.get_status() == nmfd::time_steppers::integration_status::step_failure
                                    ? section_status::step_failure
                                    : section_status::flight_limit;
            report.time        = integrator_.get_final_time();
            report.flight_time = report.time - request.start_time;
        }
        return report;
    }

private:
    std::shared_ptr<Space>                         space_;
    std::shared_ptr<Stepper>                       step_;
    std::shared_ptr<StatePolicy>                   policy_;
    params                                         params_;
    std::shared_ptr<manager_type>                  manager_;
    integrator_type                                integrator_;
    nmfd::detail::vector_wrap<Space, false, false> trajectory_;
    typename Event::evaluation_workspace           work_;
};
}
#endif
