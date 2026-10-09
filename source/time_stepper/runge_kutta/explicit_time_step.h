#ifndef NMFD_TIME_STEPPERS_EXPLICIT_TIME_STEP_H
#define NMFD_TIME_STEPPERS_EXPLICIT_TIME_STEP_H
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>
#include <nmfd/detail/vector_wrap.h>
#include <time_stepper/detail/status.h>
#include <time_stepper/runge_kutta/butcher_tables.h>
#include <time_stepper/runge_kutta/continuous_integration.h>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{
namespace detail
{
template <class Problem, class T, class = void>
struct has_set_time : std::false_type
{
};

template <class Problem, class T>
struct has_set_time<Problem, T, std::void_t<decltype( std::declval<Problem &>().set_time( std::declval<T>() ) )>>
    : std::true_type
{
};

template <class Problem, class Vector, class = void>
struct has_apply : std::false_type
{
};

template <class Problem, class Vector>
struct has_apply<
    Problem, Vector,
    std::void_t<
        decltype( std::declval<Problem &>().apply( std::declval<const Vector &>(), std::declval<Vector &>() ) )>>
    : std::true_type
{
};

template <class Adaptation, class = void>
struct has_error_requirement : std::false_type
{
};

template <class Adaptation>
struct has_error_requirement<
    Adaptation, std::void_t<decltype( std::declval<const Adaptation &>().requires_error_estimate() )>> : std::true_type
{
};

template <class Adaptation, class T, class Vector, class = void>
struct has_dual_error_assessment : std::false_type
{
};

template <class Adaptation, class T, class Vector>
struct has_dual_error_assessment<
    Adaptation, T, Vector,
    std::void_t<decltype( std::declval<Adaptation &>().assess(
        std::declval<T>(), std::declval<T>(), std::declval<const Vector &>(), std::declval<const Vector &>(),
        std::declval<T &>(), std::declval<unsigned int>(), std::declval<const Vector *>(),
        std::declval<const Vector *>()
    ) )>> : std::true_type
{
};
}

template <class VectorOperations, class Problem, class TimeStepAdaptation, bool DenseOutput = false>
class explicit_time_step
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string  method           = "RK45";
        unsigned int maximum_attempts = 32;
    };
    static_assert(
        detail::has_apply<Problem, vector_type>::value,
        "Explicit RK requires problem.apply(state, rate); optional set_time(stage_time)"
    );

    explicit_time_step(
        VectorOperations &operations, Problem &problem, TimeStepAdaptation &adaptation, const params &p = {}
    )
        : operations_( operations ), problem_( problem ), adaptation_( adaptation ), params_( p ),
          table_( make_butcher_table( p.method ) ), combined_error_( use_combined_error() ), work_( operations ),
          error_( operations, table_.is_embedded() || ( combined_error_ && supports_combined_error ) ),
          secondary_error_( operations, combined_error_ && supports_combined_error ),
          dense_( operations, !table_.has_dense_output() )
    {
        if ( table_.type() != butcher_table::scheme_type::explicit_rk )
        {
            throw std::invalid_argument( "Explicit RK cannot execute an implicit table" );
        }
        if ( p.maximum_attempts == 0 )
        {
            throw std::invalid_argument( "Empty RK attempt budget" );
        }
        work_.start_use();
        if ( table_.is_embedded() || ( combined_error_ && supports_combined_error ) )
        {
            error_.start_use();
        }
        if ( combined_error_ && supports_combined_error )
        {
            secondary_error_.start_use();
        }
        const auto stage_count = DenseOutput ? table_.dense_outout_stage_count() : table_.size();
        stages_.reserve( stage_count );
        for ( std::size_t i = 0; i < stage_count; ++i )
        {
            stages_.emplace_back( operations );
            stages_.back().start_use();
        }
    }

    void set_time( scalar_type time )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        if ( !std::isfinite( time ) )
        {
            throw std::invalid_argument( "Nonfinite RK time" );
        }
        time_ = time;
    }

    void set_target_time( scalar_type target )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        if ( !std::isfinite( target ) )
        {
            throw std::invalid_argument( "Nonfinite RK target" );
        }
        target_ = target;
    }

    scalar_type get_dt() const
    {
        return dt_;
    }

    single_step_status get_status() const
    {
        return status_;
    }

    unsigned int get_attempts() const
    {
        return attempts_;
    }

    const butcher_table &table() const
    {
        return table_;
    }

    unsigned int dense_output_order() const
    {
        return DenseOutput ? ( table_.has_dense_output() ? table_.dense_order() : std::min( 3u, table_.order() ) ) : 0;
    }

    template <bool Enabled = DenseOutput, std::enable_if_t<Enabled && DenseOutput, int> = 0>
    continuous_integration<explicit_time_step> get_continuous_integration() const
    {
        if ( !pending_ )
        {
            throw std::logic_error( "No pending RK dense output" );
        }
        return {
            *this, generation_
        }; // continuous_integration<explicit_time_step>(const SingleStepMethod& owner, std::size_t generation)
    }

    const vector_type &error_estimate() const
    {
        if ( !pending_ || !table_.is_embedded() )
        {
            throw std::logic_error( "No pending RK error estimate" );
        }
        return *error_;
    }

    // DOP853 components are raw E5/E3 defects, not standalone combined estimates.
    const vector_type &error_estimate( std::size_t component ) const
    {
        if ( !pending_ || ( !table_.is_embedded() && !combined_error_ ) )
        {
            throw std::logic_error( "No pending RK error components" );
        }
        if ( component == 0 )
        {
            return *error_;
        }
        if ( component == 1 && combined_error_ )
        {
            return *secondary_error_;
        }
        throw std::out_of_range( "RK error component index" );
    }

    void reset()
    {
        adaptation_.reset();
        initialized_ = pending_ = false;
        ++generation_;
        dt_       = 0;
        attempts_ = 0;
        status_   = single_step_status::converged;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        ++generation_;
        if constexpr ( detail::has_error_requirement<TimeStepAdaptation>::value )
        {
            if ( adaptation_.requires_error_estimate() && !table_.has_error_estimate() )
            {
                status_ = single_step_status::error_estimate_unavailable;
                return;
            }
        }
        if ( combined_error_ && !supports_combined_error )
        {
            status_ = single_step_status::error_estimate_unavailable;
            return;
        }
        if ( !initialized_ )
        {
            adaptation_.initialize( time_, in );
            initialized_ = true;
        }
        status_ = single_step_status::attempt_limit_reached;
        for ( attempts_ = 0; attempts_ < params_.maximum_attempts; )
        {
            ++attempts_;
            const auto proposed  = adaptation_.get_dt();
            const auto remaining = target_ - time_;
            dt_                  = std::copysign( std::min( proposed, std::abs( remaining ) ), remaining );
            if ( !std::isfinite( proposed ) || proposed <= 0 || !std::isfinite( dt_ ) || time_ + dt_ == time_ )
            {
                status_ = single_step_status::step_size_underflow;
                return;
            }
            if ( !compute( in ) )
            {
                // No assessment/callback may inspect an invalid numerical candidate.
                if ( adaptation_.reject_step( dt_ ) != adaptation_status::rejected )
                {
                    status_ = single_step_status::failed_nonfinite;
                    return;
                }
                adaptation_.update( adaptation_status::rejected, time_, dt_, in );
                continue;
            }
            scalar_type next_dt  = proposed;
            const auto  decision = assess( in, next_dt );
            if ( decision == adaptation_status::accepted )
            {
                if constexpr ( DenseOutput )
                {
                    if ( !prepare_dense( in ) )
                    {
                        if ( adaptation_.reject_step( dt_ ) != adaptation_status::rejected )
                        {
                            status_ = single_step_status::failed_nonfinite;
                            return;
                        }
                        adaptation_.update( adaptation_status::rejected, time_, dt_, in );
                        continue;
                    }
                }
                operations_.assign( *work_, out );
                status_  = single_step_status::converged;
                pending_ = true;
                return;
            }
            adaptation_.update( decision, time_, dt_, in );
            if ( decision != adaptation_status::rejected )
            {
                status_ = single_step_status::failed_minimum_dt;
                return;
            }
        }
    }

    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
        {
            throw std::logic_error( "No RK candidate to finalize" );
        }
        const auto elapsed = committed_time - time_;
        if ( !std::isfinite( committed_time ) || elapsed * dt_ < 0 ||
             std::abs( elapsed ) > std::abs( dt_ ) * scalar_type( 1.00000001 ) )
        {
            throw std::invalid_argument( "Committed time is outside the pending RK step" );
        }
        adaptation_.update( outcome, committed_time, elapsed, state );
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations>;
    static constexpr bool supports_combined_error =
        detail::has_dual_error_assessment<TimeStepAdaptation, scalar_type, vector_type>::value;
    VectorOperations                                    &operations_;
    Problem                                             &problem_;
    TimeStepAdaptation                                  &adaptation_;
    params                                               params_;
    butcher_table                                        table_;
    bool                                                 combined_error_;
    vector_wrap_type                                     work_, error_, secondary_error_;
    std::vector<vector_wrap_type>                        stages_;
    detail::dense_storage<VectorOperations, DenseOutput> dense_;
    std::size_t                                          generation_ = 0;
    scalar_type                                          time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                                         attempts_    = 0;
    bool                                                 initialized_ = false, pending_ = false;
    single_step_status                                   status_ = single_step_status::converged;

    friend class continuous_integration<explicit_time_step>;

    bool use_combined_error() const
    {
        if ( table_.error_estimator() != butcher_table::error_estimator_type::dop853_combined )
        {
            return false;
        }
        if constexpr ( detail::has_error_requirement<TimeStepAdaptation>::value )
        {
            return adaptation_.requires_error_estimate();
        }
        return supports_combined_error;
    }

    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( supports_combined_error )
        {
            if ( combined_error_ )
            {
                return adaptation_.assess(
                    time_, dt_, in, *work_, next_dt, table_.error_order(), &*error_, &*secondary_error_
                );
            }
        }
        return adaptation_.assess(
            time_, dt_, in, *work_, next_dt, table_.error_order(), table_.is_embedded() ? &*error_ : nullptr
        );
    }

    scalar_type evaluate_dense( scalar_type theta, vector_type &out, std::size_t generation ) const
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        if ( !pending_ || generation != generation_ )
        {
            throw std::logic_error( "Expired RK dense output" );
        }
        if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
        {
            throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
        }
        if ( theta == 0 )
        {
            operations_.assign( *dense_.previous, out );
        }
        else if ( theta == 1 )
        {
            operations_.assign( *work_, out );
        }
        else if ( table_.has_dense_output() )
        {
            operations_.assign( *dense_.previous, out );
            for ( std::size_t i = 0; i < table_.dense_outout_stage_count(); ++i )
            {
                const auto weight = static_cast<scalar_type>( table_.dense_b( i, theta ) );
                if ( weight != 0 )
                {
                    operations_.add_mul( dt_ * weight, *stages_[i], out );
                }
            }
        }
        else
        {
            const auto a = theta * theta * ( 3 - 2 * theta ), b = theta * ( 1 - theta ) * ( 1 - theta ),
                       c = -theta * theta * ( 1 - theta );
            operations_.assign_mul( 1 - a, *dense_.previous, a, *work_, out );
            operations_.add_mul( dt_ * b, *stages_[0], out );
            operations_.add_mul( dt_ * c, *dense_.endpoint_rate, out );
        }
        return time_ + theta * dt_;
    }

    bool prepare_dense( const vector_type &in )
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        // Reuse the snapshot buffer as scratch without disturbing the candidate.
        auto &scratch = *dense_.previous;
        for ( std::size_t i = table_.size(); i < table_.dense_outout_stage_count(); ++i )
        {
            operations_.assign( in, scratch );
            for ( std::size_t j = 0; j < i; ++j )
            {
                const auto weight = static_cast<scalar_type>( table_.dense_outout_a( i, j ) );
                if ( weight != 0 )
                {
                    operations_.add_mul( dt_ * weight, *stages_[j], scratch );
                }
            }
            if ( !operations_.check_is_valid_number( scratch ) )
            {
                return false;
            }
            if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
            {
                problem_.set_time( time_ + dt_ * static_cast<scalar_type>( table_.dense_outout_c( i ) ) );
            }
            problem_.apply( scratch, *stages_[i] );
            if ( !operations_.check_is_valid_number( *stages_[i] ) )
            {
                return false;
            }
        }
        if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
        {
            problem_.set_time( time_ + dt_ );
        }
        if ( !table_.has_dense_output() )
        {
            problem_.apply( *work_, *dense_.endpoint_rate );
            if ( !operations_.check_is_valid_number( *dense_.endpoint_rate ) )
            {
                return false;
            }
        }
        // Snapshot before publishing the candidate: step.apply(x, x) is supported.
        operations_.assign( in, scratch );
        return true;
    }

    bool compute( const vector_type &in )
    {
        auto &work = *work_;
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            operations_.assign( in, work );
            for ( std::size_t j = 0; j < i; ++j )
            {
                if ( table_.a( i, j ) != 0 )
                {
                    operations_.add_mul( dt_ * static_cast<scalar_type>( table_.a( i, j ) ), *stages_[j], work );
                }
            }
            if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
            {
                problem_.set_time( time_ + dt_ * static_cast<scalar_type>( table_.c( i ) ) );
            }
            problem_.apply( work, *stages_[i] );
            if ( !operations_.check_is_valid_number( *stages_[i] ) )
            {
                return false;
            }
        }
        // All derivatives are retained; the last stage state can become the candidate.
        operations_.assign( in, work );
        const bool primary_error = table_.is_embedded() || combined_error_;
        if ( primary_error )
        {
            operations_.assign_scalar( scalar_type( 0 ), *error_ );
        }
        if ( combined_error_ )
        {
            operations_.assign_scalar( scalar_type( 0 ), *secondary_error_ );
        }
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            if ( table_.b( i ) != 0 )
            {
                operations_.add_mul( dt_ * static_cast<scalar_type>( table_.b( i ) ), *stages_[i], work );
            }
            if ( primary_error && table_.error_b( i ) != 0 )
            {
                operations_.add_mul( dt_ * static_cast<scalar_type>( table_.error_b( i ) ), *stages_[i], *error_ );
            }
            if ( combined_error_ && table_.secondary_error_b( i ) != 0 )
            {
                operations_.add_mul(
                    dt_ * static_cast<scalar_type>( table_.secondary_error_b( i ) ), *stages_[i], *secondary_error_
                );
            }
        }
        return operations_.check_is_valid_number( work ) &&
               ( !primary_error || operations_.check_is_valid_number( *error_ ) ) &&
               ( !combined_error_ || operations_.check_is_valid_number( *secondary_error_ ) );
    }
};
}
}
}
#endif
