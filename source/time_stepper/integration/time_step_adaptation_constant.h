#ifndef NMFD_TIME_STEPPERS_ADAPTATION_CONSTANT_H
#define NMFD_TIME_STEPPERS_ADAPTATION_CONSTANT_H
#include <cmath>
#include <stdexcept>
#include <time_stepper/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
template <class VectorOperations>
class time_step_adaptation_constant
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    struct params
    {
        scalar_type step_size = scalar_type( .01 );
    };

    explicit time_step_adaptation_constant( const params &p = {} ) : params_( p )
    {
        if ( !std::isfinite( p.step_size ) || p.step_size <= 0 )
        {
            throw std::invalid_argument( "Constant step size must be positive and finite" );
        }
        reset();
    }
    scalar_type initialize( scalar_type, const vector_type &, scalar_type dt = -1 )
    {
        dt_ = dt > 0 ? dt : params_.step_size;
        if ( !std::isfinite( dt_ ) )
        {
            throw std::invalid_argument( "Invalid initial step size" );
        }
        return dt_;
    }
    void reset()
    {
        dt_ = params_.step_size;
    }
    scalar_type get_dt() const
    {
        return dt_;
    }
    bool requires_error_estimate() const
    {
        return false;
    }
    adaptation_status assess(
        scalar_type, scalar_type, const vector_type &, const vector_type &, scalar_type &dt_next, unsigned int = 0,
        const vector_type * = nullptr
    )
    {
        dt_next = dt_;
        return adaptation_status::accepted;
    }
    void update( adaptation_status, scalar_type, scalar_type, const vector_type & )
    {
    }
    adaptation_status reject_step( scalar_type )
    {
        return adaptation_status::failed;
    }

private:
    params      params_;
    scalar_type dt_;
};
}
}
}
#endif
