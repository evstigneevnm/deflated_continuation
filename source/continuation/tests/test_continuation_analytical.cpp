#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <containers/knots.hpp>
#include <continuation/continuation_analytical.hpp>
#include <numerical_algos/tests/nmfd_cpu_reference_vector_space.h>

namespace
{
unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class Exception, class Action>
void require_throws( Action action, const char *message )
{
    bool caught = false;
    try
    {
        action();
    }
    catch ( const Exception & )
    {
        caught = true;
    }
    require( caught, message );
}

struct log_type
{
    template <class... Args>
    void info_f( const char *, Args... )
    {
    }
    template <class... Args>
    void warning_f( const char *, Args... )
    {
    }
};

struct unused_dependency
{
};
struct convergence_type
{
};
struct correction_type
{
    convergence_type  strategy;
    convergence_type *get_convergence_strategy_handle()
    {
        return &strategy;
    }
};
struct newton_types
{
    using tangent_solver_type           = void;
    using continuation_convergence_type = convergence_type;
    using continuation_correction_type  = correction_type;
};

template <class T>
struct exact_problem
{
    bool           zero    = true;
    T              minimum = T( 2 ), maximum = T( 30 );
    std::vector<T> evaluations;

    // Exact equilibria of F(q,lambda)=q-a(lambda): zero or a curved analytical branch.
    std::vector<T> value( const T parameter ) const
    {
        return zero ? std::vector<T>{ T( 0 ), T( 0 ) }
                    : std::vector<T>{ parameter * parameter, parameter * parameter * parameter };
    }
    void exact_solution( const T parameter, std::vector<T> &state )
    {
        require( parameter >= minimum && parameter <= maximum, "Exact provider evaluated outside its domain" );
        evaluations.push_back( parameter );
        state = value( parameter );
    }
};

template <class T>
struct recorded_curve
{
    using reason_type = container::curve_endpoint_reason;
    struct point_type
    {
        T              parameter;
        std::vector<T> state;
        bool           forced;
        reason_type    reason;
        unsigned int   segment;
    };
    std::vector<point_type> points;
    unsigned int            segment = 0, prints = 0;
    void                    start_new_segment()
    {
        ++segment;
    }
    void add( T parameter, const std::vector<T> &state, bool forced, reason_type reason )
    {
        points.push_back( { parameter, state, forced, reason, segment } );
    }
    void set_last_endpoint_reason( reason_type reason )
    {
        points.back().reason = reason;
    }
    void print_curve()
    {
        ++prints;
    }
};

template <class T>
struct fixture
{
    using space_type   = nmfd::tests::cpu_reference_vector_space<T>;
    using curve_type   = recorded_curve<T>;
    using sampler_type = continuation::continuation_analytical<
        space_type, unused_dependency, log_type, exact_problem<T>, unused_dependency, container::knots<T>,
        unused_dependency, unused_dependency, curve_type, continuation::system_operator_continuation, newton_types>;
    space_type                       space{ 2 };
    unused_dependency                files, linear_operator, linear_solver, newton;
    log_type                         log;
    exact_problem<T>                 problem;
    container::knots<T>              knots;
    std::shared_ptr<correction_type> correction = std::make_shared<correction_type>();
    sampler_type                     sampler{ &space, &files,         &log,    &problem,  &linear_operator,
                          &knots, &linear_solver, &newton, correction };
    curve_type                       curve;

    fixture()
    {
        knots.add_element(
            std::vector<T>{ T( 4.333 ), T( 5 ), T( 8 ), T( 8.005 ), T( 9 ), T( 17 ), T( 19 ), T( 30 ), T( 35 ) }
        );
        sampler.set_parameter_bounds( problem.minimum, problem.maximum );
        sampler.set_steps( 1500, T( 100 ), T( 250 ) );
        sampler.set_knot_resolver( []( const T requested, T &effective ) {
            effective = requested == T( 8 ) || requested == T( 8.005 ) ? T( 8.01 ) : requested;
            return true;
        } );
    }

    bool run( T initial )
    {
        curve = {};
        problem.evaluations.clear();
        auto       state  = problem.value( initial );
        const auto before = state;
        auto      *output = &curve;
        const bool result = sampler.continuate_curve( output, state, initial );
        require( state == before, "Analytical sampling changed its input state" );
        require( curve.prints == 1 && curve.segment == 2, "Analytical semicurves were not separated or printed" );
        return result;
    }
};

template <class T>
void test_knots_and_arclength_independence()
{
    fixture<T> f;
    require( f.run( T( 4.333 ) ), "Zero analytical branch did not finish" );
    const std::vector<T> expected{ T( 4.333 ), T( 2 ),  T( 4.333 ), T( 5 ), T( 8.01 ),
                                   T( 9 ),     T( 17 ), T( 19 ),    T( 30 ) };
    require(
        f.curve.points.size() == expected.size(), "Analytical sampler skipped knots or duplicated relocated knots"
    );
    for ( std::size_t i = 0; i < expected.size(); ++i )
    {
        const auto &point = f.curve.points[i];
        require( point.parameter == expected[i], "Analytical knots are not in traversal order" );
        require( point.state == f.problem.value( point.parameter ), "Zero sample is not an exact equilibrium" );
        require( point.forced, "Mandatory analytical sample was not forced to storage" );
        require( point.segment == ( i < 2 ? 1u : 2u ), "Parameter reset connects separate semicurves" );
    }
    require( f.curve.points[1].reason == container::curve_endpoint_reason::boundary_min, "Missing minimum endpoint" );
    require(
        f.curve.points.back().reason == container::curve_endpoint_reason::boundary_max, "Missing maximum endpoint"
    );
    f.sampler.set_steps( 1500, T( .01 ), T( .02 ) );
    require( f.run( T( 4.333 ) ), "Repeated analytical sampling failed" );
    for ( std::size_t i = 0; i < expected.size(); ++i )
        require(
            f.curve.points[i].parameter == expected[i], "Numerical arclength settings changed analytical sampling"
        );

    f.sampler.set_steps( 1500, T( 100 ), T( 250 ), 1 );
    f.sampler.set_maximum_parameter_step( T( 250 ) );
    require( f.run( T( 4.333 ) ), "Reversed analytical sampling failed" );
    require(
        f.curve.points[1].parameter == T( 5 ) && f.curve.points.back().parameter == T( 2 ),
        "Reversed traversal is incorrect"
    );
    require( f.run( T( 2 ) ), "Analytical branch starting at the minimum failed" );
    require(
        f.curve.points.back().reason == container::curve_endpoint_reason::boundary_min,
        "Initial boundary was not marked"
    );
    require( f.run( T( 30 ) ), "Analytical branch starting at the maximum failed" );
    require(
        f.curve.points.front().reason == container::curve_endpoint_reason::boundary_max,
        "Initial maximum boundary was not marked"
    );
}

template <class T>
void test_curved_branch_and_gap()
{
    fixture<T> f;
    f.problem.zero    = false;
    f.problem.minimum = T( 0 );
    f.problem.maximum = T( 3 );
    f.knots           = {};
    f.knots.add_element( std::vector<T>{ T( -1 ), T( .3 ), T( 2 ), T( 3 ), T( 8 ) } );
    f.sampler.set_parameter_bounds( T( 0 ), T( 3 ) );
    f.sampler.set_maximum_parameter_step( T( .7 ) );
    require( f.run( T( 1.1 ) ), "Curved analytical branch did not finish" );
    bool optional_sample = false;
    for ( std::size_t i = 0; i < f.curve.points.size(); ++i )
    {
        const auto &point = f.curve.points[i];
        require(
            point.state == f.problem.value( point.parameter ),
            "Nonconstant analytical state was interpolated, not evaluated"
        );
        optional_sample = optional_sample || !point.forced;
        if ( i > 0 && point.segment == f.curve.points[i - 1].segment )
        {
            const T gap = point.parameter - f.curve.points[i - 1].parameter;
            require(
                std::abs( gap ) <= T( .7 ) + T( 16 ) * std::numeric_limits<T>::epsilon(),
                "Analytical maximum gap was exceeded"
            );
            require( point.segment == 1 ? gap < T( 0 ) : gap > T( 0 ), "Analytical samples reversed direction" );
        }
    }
    require( optional_sample, "Explicit maximum gap did not add intermediate analytical samples" );
    for ( const T parameter : { T( 0 ), T( .3 ), T( 2 ), T( 3 ) } )
        require(
            std::any_of(
                f.curve.points.begin(), f.curve.points.end(),
                [&]( const auto &point ) { return point.parameter == parameter && point.forced; }
            ),
            "Maximum-gap sampling skipped a mandatory knot or boundary"
        );
}

template <class T>
void test_failure_and_validation()
{
    fixture<T> f;
    f.sampler.set_steps( 1, T( 100 ), T( 250 ) );
    require( !f.run( T( 4.333 ) ), "Exhausted analytical sampling falsely reported complete coverage" );
    require(
        f.curve.points.back().reason == container::curve_endpoint_reason::max_steps && f.curve.points.back().forced,
        "Sampling limit did not preserve a marked terminal sample"
    );
    f.sampler.set_steps( 1500, T( 100 ), T( 250 ) );
    for ( const bool nonfinite : { false, true } )
    {
        f.sampler.set_exact_solution_provider( [&]( const T &parameter, std::vector<T> &state ) {
            if ( parameter == T( 9 ) )
            {
                state.assign( 2, std::numeric_limits<T>::infinity() );
                return nonfinite;
            }
            f.problem.exact_solution( parameter, state );
            return true;
        } );
        require( !f.run( T( 4.333 ) ), "Unavailable or nonfinite analytical sample was accepted" );
        require(
            f.curve.points.back().parameter == T( 8.01 ) &&
                f.curve.points.back().reason == container::curve_endpoint_reason::knot_interpolation_failure,
            "Failed analytical evaluation overwrote the last valid sample"
        );
        for ( const auto &point : f.curve.points )
            require( f.space.is_valid_number( point.state ), "Invalid analytical sample entered the curve" );
    }
    f.sampler.clear_exact_solution_provider();
    require( f.run( T( 4.333 ) ), "Analytical sampler did not recover after a failed curve" );
    for ( const T invalid : { T( -1 ), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN() } )
        require_throws<std::invalid_argument>(
            [&] { f.sampler.set_maximum_parameter_step( invalid ); }, "Invalid parameter gap accepted"
        );
    require_throws<std::invalid_argument>( [&] { f.run( T( 1 ) ); }, "Out-of-bounds initial parameter accepted" );
    require_throws<std::invalid_argument>(
        [&] { f.run( std::numeric_limits<T>::quiet_NaN() ); }, "Nonfinite initial parameter accepted"
    );
    f.sampler.set_steps( 0, T( 100 ), T( 250 ) );
    require_throws<std::invalid_argument>( [&] { f.run( T( 4.333 ) ); }, "Zero sampling budget accepted" );
    f.sampler.set_steps( 1500, T( 100 ), T( 250 ) );
    f.sampler.set_maximum_parameter_step( std::numeric_limits<T>::min() );
    require_throws<std::runtime_error>(
        [&] { f.run( T( 4.333 ) ); }, "Nonprogressing parameter step was not diagnosed"
    );
    f.sampler.set_maximum_parameter_step( T( 0 ) );
    f.sampler.set_knot_resolver( []( const T, T &effective ) {
        effective = std::numeric_limits<T>::quiet_NaN();
        return true;
    } );
    require_throws<std::invalid_argument>( [&] { f.run( T( 4.333 ) ); }, "Nonfinite effective knot accepted" );
}
}

int main()
{
    try
    {
        test_knots_and_arclength_independence<float>();
        test_knots_and_arclength_independence<double>();
        test_knots_and_arclength_independence<long double>();
        test_curved_branch_and_gap<float>();
        test_curved_branch_and_gap<double>();
        test_curved_branch_and_gap<long double>();
        test_failure_and_validation<float>();
        test_failure_and_validation<double>();
        test_failure_and_validation<long double>();
        std::printf( "Analytical continuation: %u checks passed\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Analytical continuation failed after %u checks: %s\n", checks, error.what() );
        return 1;
    }
}
