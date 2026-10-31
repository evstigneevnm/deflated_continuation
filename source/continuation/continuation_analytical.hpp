#ifndef __CONTINUATION_ANALYTICAL_HPP__
#define __CONTINUATION_ANALYTICAL_HPP__

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <common/scalar_math.h>
#include <continuation/continuation.hpp>

namespace continuation
{

template <
    class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator,
    class Knots, class LinearSolver, class Newton, class Curve,
    template <class, class, class, class, class> class SystemOperatorContinuation = system_operator_continuation,
    class NewtonTypes                                                             = void>
class continuation_analytical : public continuation<
                                    VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator,
                                    Knots, LinearSolver, Newton, Curve, SystemOperatorContinuation, NewtonTypes>
{
    using parent_t = continuation<
        VectorOperations, VectorFileOperations, Log, NonlinearOperations, LinearOperator, Knots, LinearSolver, Newton,
        Curve, SystemOperatorContinuation, NewtonTypes>;
    using T                 = typename parent_t::T;
    using T_vec             = typename parent_t::T_vec;
    using endpoint_reason_t = typename parent_t::endpoint_reason_t;

public:
    using parent_t::parent_t;

    // Zero samples only effective knots and boundaries; arclength settings are independent.
    void set_maximum_parameter_step( const T step )
    {
        if ( !common::scalar_math::isfinite( step ) || step < T( 0 ) )
            throw std::invalid_argument( "Analytical maximum parameter step must be finite and nonnegative" );
        maximum_parameter_step_ = step;
    }

    void set_exact_solution_provider( std::function<bool( const T &, T_vec & )> provider )
    {
        exact_solution_provider_ = std::move( provider );
    }

    void clear_exact_solution_provider()
    {
        exact_solution_provider_ = {};
    }

    bool continuate_curve( Curve *&curve, const T_vec &state, const T &parameter )
    {
        parent_t::update_knots();
        prepare_parameters( parameter );
        if ( !parent_t::vec_ops->is_valid_number( state ) )
            throw std::invalid_argument( "Analytical sampling requires a finite initial state" );
        parent_t::bif_diag     = curve;
        parent_t::direction    = parent_t::initial_direciton;
        parent_t::fail_flag    = false;
        parent_t::hard_failure = false;
        parent_t::endpoint_state.reset_curve();
        parent_t::break_semicurve = 0;
        parent_t::lambda_start    = parameter;
        parent_t::vec_ops->assign( state, parent_t::x_start );

        while ( parent_t::break_semicurve < 2 )
        {
            parent_t::vec_ops->assign( parent_t::x_start, parent_t::x0 );
            parent_t::lambda0                           = parameter;
            parent_t::current_semicurve_points          = 0;
            parent_t::current_semicurve_endpoint_reason = endpoint_reason_t::none;
            detail::start_new_curve_segment_if_available( curve );
            start_semicurve();
            parent_t::change_direction();
        }
        curve->print_curve();
        return !parent_t::hard_failure && !parent_t::endpoint_state.incomplete();
    }

private:
    T                                         maximum_parameter_step_ = T( 0 );
    std::vector<T>                            parameters_;
    std::function<bool( const T &, T_vec & )> exact_solution_provider_;

    void prepare_parameters( const T initial )
    {
        const T minimum = parent_t::lambda_min;
        const T maximum = parent_t::lambda_max;
        if ( !common::scalar_math::isfinite( initial ) || !common::scalar_math::isfinite( minimum ) ||
             !common::scalar_math::isfinite( maximum ) || !( minimum < maximum ) || initial < minimum ||
             initial > maximum || parent_t::max_S == 0 ||
             ( parent_t::initial_direciton != -1 && parent_t::initial_direciton != 1 ) )
            throw std::invalid_argument( "Invalid analytical sampling bounds, initial parameter, direction or limit" );

        parameters_ = { minimum, maximum };
        for ( const auto requested : *parent_t::knots )
        {
            T effective = requested;
            if ( parent_t::knot_resolver )
                parent_t::knot_resolver( requested, effective );
            if ( !common::scalar_math::isfinite( effective ) )
                throw std::invalid_argument( "Analytical sampling requires finite effective knots" );
            if ( effective >= minimum && effective <= maximum )
                parameters_.push_back( effective );
        }
        std::sort( parameters_.begin(), parameters_.end() );
        parameters_.erase( std::unique( parameters_.begin(), parameters_.end() ), parameters_.end() );
    }

    // Select the nearest mandatory point before adding optional samples inside that interval.
    T next_parameter( const T current, bool &mandatory ) const
    {
        auto target = std::upper_bound( parameters_.begin(), parameters_.end(), current );
        if ( parent_t::direction < 0 )
        {
            target = std::lower_bound( parameters_.begin(), parameters_.end(), current );
            --target;
        }
        T next = *target;
        if ( maximum_parameter_step_ > T( 0 ) )
        {
            const T proposed = current + T( parent_t::direction ) * maximum_parameter_step_;
            if ( parent_t::direction > 0 ? proposed < next : proposed > next )
                next = proposed;
        }
        if ( !common::scalar_math::isfinite( next ) || next == current )
            throw std::runtime_error( "Analytical parameter step makes no finite progress" );
        mandatory = next == *target;
        return next;
    }

    endpoint_reason_t boundary_reason( const T parameter ) const
    {
        if ( parameter == parent_t::lambda_min )
            return endpoint_reason_t::boundary_min;
        if ( parameter == parent_t::lambda_max )
            return endpoint_reason_t::boundary_max;
        return endpoint_reason_t::none;
    }

    void start_semicurve()
    {
        const T boundary = parent_t::direction < 0 ? parent_t::lambda_min : parent_t::lambda_max;
        parent_t::endpoint_state.reset_step();
        parent_t::add_solution_to_curve(
            parent_t::lambda0, parent_t::x0, true,
            parent_t::lambda0 == boundary ? boundary_reason( boundary ) : endpoint_reason_t::none
        );
        for ( unsigned int step = 0; step < parent_t::max_S && parent_t::lambda0 != boundary; ++step )
        {
            bool    mandatory;
            const T next = next_parameter( parent_t::lambda0, mandatory );
            if ( !evaluate_exact_solution( next, parent_t::x1 ) || !parent_t::vec_ops->is_valid_number( parent_t::x1 ) )
            {
                parent_t::mark_last_curve_point( endpoint_reason_t::knot_interpolation_failure );
                parent_t::hard_failure = true;
                parent_t::log->warning_f(
                    "continuation_analytical: exact solution unavailable or invalid at lambda = %le.", double( next )
                );
                break;
            }
            const auto reason = next == boundary              ? boundary_reason( boundary )
                                : step + 1 == parent_t::max_S ? endpoint_reason_t::max_steps
                                                              : endpoint_reason_t::none;
            parent_t::add_solution_to_curve( next, parent_t::x1, mandatory, reason );
            parent_t::vec_ops->assign( parent_t::x1, parent_t::x0 );
            parent_t::lambda0 = next;
        }
        ++parent_t::break_semicurve;
    }

    bool evaluate_exact_solution( const T &parameter, T_vec &state )
    {
        if ( exact_solution_provider_ )
            return exact_solution_provider_( parameter, state );
        parent_t::nonlin_op->exact_solution( parameter, state );
        return true;
    }
};

}

#endif
