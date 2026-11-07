#ifndef PERIODIC_ORBIT_IMPLICIT_STAGE_PROVIDER_H
#define PERIODIC_ORBIT_IMPLICIT_STAGE_PROVIDER_H

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/detail/affine_shift.h>
#include <nmfd/time_steppers/detail/problem_traits.h>

namespace periodic_orbit
{
template <class Space>
struct implicit_stage_context
{
    using T = typename Space::scalar_type;
    using X = typename Space::vector_type;
    T        time, a, b;
    const X &state, &previous;
};

namespace detail
{
template <class Sensitivity, class T, class X, class = void>
struct has_implicit_rate_tangent : std::false_type
{
};
template <class Sensitivity, class T, class X>
struct has_implicit_rate_tangent<
    Sensitivity, T, X,
    std::void_t<decltype( std::declval<const Sensitivity &>().rate_jacobian_apply(
        std::declval<T>(), std::declval<const X &>(), std::declval<const X &>(), std::declval<X &>()
    ) )>> : std::true_type
{
};
}

// Constructed only by derivative assembly, with a dedicated mutable problem.
// Sensitivity supplies -R_previous*dw, including zero algebraic mass rows.
template <class Space, class Problem, class NonlinearSolver, class LinearSolver, class Sensitivity>
class implicit_stage_provider
{
public:
    using T                             = typename Space::scalar_type;
    using X                             = typename Space::vector_type;
    using context_type                  = implicit_stage_context<Space>;
    static constexpr bool supports_rate = nmfd::time_steppers::detail::has_rate_evaluation<Problem, X>::value &&
                                          detail::has_implicit_rate_tangent<Sensitivity, T, X>::value;
    implicit_stage_provider(
        std::shared_ptr<Space> space, std::shared_ptr<Problem> problem, std::shared_ptr<NonlinearSolver> nonlinear,
        std::shared_ptr<LinearSolver> linear, std::shared_ptr<const Sensitivity> sensitivity, T reference_time = T( 1 )
    )
        : space_( std::move( space ) ), problem_( std::move( problem ) ), nonlinear_( std::move( nonlinear ) ),
          linear_( std::move( linear ) ), sensitivity_( std::move( sensitivity ) ), reference_time_( reference_time ),
          rhs_( checked_space() )
    {
        if ( !problem_ || !nonlinear_ || !linear_ || !sensitivity_ )
            throw std::invalid_argument( "Missing implicit sensitivity dependencies" );
        nmfd::time_steppers::detail::balanced_affine_shift( T( 0 ), reference_time );
        rhs_.init();
        rhs_.start_use();
        nonlinear_->set_nonlinear_operator( problem_ );
    }
    bool solve_stage( T time, T alpha, const X &previous, X &state )
    {
        const auto shift = nmfd::time_steppers::detail::balanced_affine_shift( alpha, reference_time_ );
        time_            = time;
        a_               = shift.a;
        b_               = shift.b;
        problem_->set_time( time );
        problem_->set_affine_shift( a_, b_ );
        space_->assign( previous, problem_->previous_state() );
        space_->assign_scalar( T( 0 ), problem_->known_rhs() );
        if ( !nonlinear_->solve( state ) || !space_->is_valid_number( state ) )
            return false;
        if ( !initialized_ )
        {
            // Warm up a separately supplied lazy linear solver during preparation,
            // never on the first published Jv. known_rhs remains the zero vector.
            problem_->set_linearization_point( state );
            linear_->set_operator( problem_->get_jacobi_operator() );
            space_->assign_scalar( T( 0 ), *rhs_ );
            if ( !linear_->solve( *rhs_, problem_->known_rhs() ) )
                return false;
            space_->assign_scalar( T( 0 ), problem_->known_rhs() );
            initialized_ = true;
        }
        return true;
    }
    bool solve_tangent( const X &state, const X &previous, const X &delta_previous, X &delta_state )
    {
        const context_type context{ time_, a_, b_, state, previous };
        if ( !sensitivity_->stage_tangent_rhs( context, delta_previous, *rhs_ ) || !space_->is_valid_number( *rhs_ ) )
            return false;
        problem_->set_linearization_point( state );
        linear_->set_operator( problem_->get_jacobi_operator() );
        space_->assign_scalar( T( 0 ), delta_state );
        return linear_->solve( *rhs_, delta_state );
    }
    bool evaluate_rate( T time, const X &state, X &rate )
    {
        if constexpr ( supports_rate )
        {
            problem_->set_time( time );
            return problem_->evaluate_rate( state, rate );
        }
        else
            return false;
    }
    bool apply_rate_tangent( T time, const X &state, const X &direction, X &out )
    {
        if constexpr ( supports_rate )
            return sensitivity_->rate_jacobian_apply( time, state, direction, out );
        else
            return false;
    }

private:
    std::shared_ptr<Space>                         space_;
    std::shared_ptr<Problem>                       problem_;
    std::shared_ptr<NonlinearSolver>               nonlinear_;
    std::shared_ptr<LinearSolver>                  linear_;
    std::shared_ptr<const Sensitivity>             sensitivity_;
    T                                              reference_time_, time_ = 0, a_ = 0, b_ = 1;
    bool                                           initialized_ = false;
    nmfd::detail::vector_wrap<Space, false, false> rhs_;
    const Space                                   &checked_space() const
    {
        if ( !space_ )
            throw std::invalid_argument( "Missing implicit sensitivity space" );
        return *space_;
    }
};

}
#endif
