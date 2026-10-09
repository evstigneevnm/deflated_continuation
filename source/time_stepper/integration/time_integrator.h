#ifndef NMFD_TIME_STEPPERS_TIME_INTEGRATOR_H
#define NMFD_TIME_STEPPERS_TIME_INTEGRATOR_H
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <time_stepper/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
template <class VectorOperations, class SingleStepMethod, class ExternalManagement = detail::no_external_operations>
class time_integrator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    struct params
    {
        std::size_t maximum_steps = 1000000;
    };

    time_integrator(
        VectorOperations &operations, SingleStepMethod &step, const params &p = {},
        ExternalManagement *external = nullptr
    )
        : operations_( operations ), step_( step ), params_( p ), external_( external ), candidate_( operations )
    {
        if ( !p.maximum_steps )
            throw std::invalid_argument( "Empty integration step budget" );
        candidate_.start_use();
    }
    void set_time_interval( scalar_type start, scalar_type end )
    {
        if ( !std::isfinite( start ) || !std::isfinite( end ) || !std::isfinite( end - start ) )
        {
            throw std::invalid_argument( "Invalid integration interval" );
        }
        start_ = start;
        end_   = end;
    }
    integration_status get_status() const
    {
        return status_;
    }
    scalar_type get_final_time() const
    {
        return time_;
    }
    std::size_t get_steps() const
    {
        return steps_;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        step_.reset();
        time_           = start_;
        steps_          = 0;
        status_         = integration_status::running;
        auto &candidate = *candidate_;
        // Stage the initial state once so apply(x, x) needs no backend alias check.
        operations_.assign( in, candidate );
        operations_.assign( candidate, out );
        const auto &current = out;
        if ( !operations_.check_is_valid_number( current ) )
        {
            status_ = integration_status::step_failure;
            return;
        }
        while ( time_ != end_ )
        {
            if ( steps_ == params_.maximum_steps )
            {
                status_ = integration_status::attempt_limit_reached;
                return;
            }
            step_.set_time( time_ );
            step_.set_target_time( end_ );
            step_.apply( current, candidate );
            if ( step_.get_status() != single_step_status::converged )
            {
                status_ = integration_status::step_failure;
                return;
            }
            auto next_time = time_ + step_.get_dt();
            if ( step_.get_dt() == end_ - time_ )
            {
                next_time = end_;
            }
            const auto trial_time = next_time;
            bool       modified   = false;
            if constexpr ( !std::is_same_v<ExternalManagement, detail::no_external_operations> )
            {
                if ( external_ )
                {
                    external_->set_time_interval( time_, next_time );
                    modified = external_->apply( current, candidate );
                    status_  = external_->get_status();
                }
            }
            const auto direction = end_ > time_ ? scalar_type( 1 ) : scalar_type( -1 );
            if ( !std::isfinite( next_time ) || direction * ( next_time - time_ ) <= 0 ||
                 direction * ( next_time - trial_time ) > 0 || !operations_.check_is_valid_number( candidate ) ||
                 ( status_ != integration_status::running && status_ != integration_status::completed &&
                   status_ != integration_status::stopped_by_external_operation ) )
            {
                step_.finalize( adaptation_status::failed, time_, current );
                status_ = integration_status::external_operation_failure;
                return;
            }
            step_.finalize(
                modified || next_time != trial_time ? adaptation_status::accepted_modified
                                                    : adaptation_status::accepted,
                next_time, candidate
            );
            operations_.assign( candidate, out );
            time_ = next_time;
            ++steps_;
            if ( status_ != integration_status::running )
                return;
        }
        status_ = integration_status::completed;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations>;
    VectorOperations   &operations_;
    SingleStepMethod   &step_;
    params              params_;
    ExternalManagement *external_;
    vector_wrap_type    candidate_;
    scalar_type         start_ = 0, end_ = 1, time_ = 0;
    std::size_t         steps_  = 0;
    integration_status  status_ = integration_status::running;
};
}
}
}
#endif
