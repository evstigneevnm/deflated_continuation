#include "continuation_newton_fixture.h"
#include <limits>
#include <scfd/utils/log_std.h>
#include <numerical_algos/detail/str_source_helper.h>
#include <continuation/system_operator_continuation.h>

namespace
{
using continuation::tests::test_report;

// A constant physical residual has the same RMS norm on every grid. The
// arclength residual is a separate scalar and must not be divided by sqrt(N).
template <class T>
void test_residual_norm( test_report &report )
{
    using operations_type = nmfd::operations::scfd_vector_space<continuation::tests::backend_type, T>;
    using scalar_space    = nmfd::operations::static_vector_space<T, 1, scfd::static_vec::vec<T, 1>>;
    using space_type      = nmfd::operations::pair_vector_space<operations_type, scalar_space>;
    using vector_type     = typename space_type::vector_type;
    struct constant_residual
    {
        operations_type &operations;
        T                state_value = 0, constraint = 0;
        unsigned int     calls = 0;
        void             apply( const vector_type &, vector_type &out )
        {
            ++calls;
            operations.assign_scalar( state_value, out.first );
            out.second[0] = constraint;
        }
    };
    struct values
    {
        T state, constraint;
    };
    const values cases[]   = { { std::ldexp( T( 1 ), -37 ), T( 0 ) }, { T( 0 ), T( .75 ) }, { T( .25 ), T( -.75 ) } };
    const T      tolerance = T( 32 ) * std::numeric_limits<T>::epsilon();
    for ( std::size_t size : { 1u, 16u, 1024u, 32766u } )
    {
        auto                                                operations = std::make_shared<operations_type>( size );
        space_type                                          space( operations, std::make_shared<scalar_space>() );
        nmfd::detail::vector_wrap<space_type, false, false> state( space ), residual( space );
        for ( auto *storage : { &state, &residual } )
        {
            storage->init();
            storage->start_use();
        }
        space.assign_scalar( T( 0 ), *state );
        for ( const auto &value : cases )
        {
            constant_residual op{ *operations, value.state, value.constraint };
            const T           norm = continuation::corrector_residual_norm::evaluate( space, op, *state, *residual );
            const T           expected = std::sqrt( value.state * value.state + value.constraint * value.constraint );
            report.require(
                std::abs( norm - expected ) <= tolerance * expected,
                "Corrector merit changed with state dimension or diluted the scalar constraint"
            );
            report.require(
                op.calls == 1 && ( *residual ).second[0] == value.constraint,
                "Corrector assessment reevaluated or rescaled the residual"
            );
            const T squared = T( size ) * value.state * value.state + value.constraint * value.constraint;
            report.require(
                std::abs( space.scalar_prod( *residual, *residual ) - squared ) <= tolerance * squared,
                "Corrector assessment changed the pair-space inner product"
            );
            if ( size == 32766 && value.constraint == T( 0 ) )
                report.require(
                    norm < T( 1e-9 ) && space.norm_l2( *residual ) > T( 1e-9 ),
                    "Large-state continuation still uses the overly strict Euclidean residual norm"
                );
        }
    }
}

// Circle F=u^2+lambda^2-1 with a linear arclength equation. The complete
// two-block Jacobian is checked against finite differences in both components.
template <class T>
void test_operator( test_report &report )
{
    using fixture_type = continuation::tests::circle_fixture<T>;
    fixture_type f;
    bool         unprepared = false;
    try
    {
        f.op->get_jacobi_operator();
    }
    catch ( const std::logic_error & )
    {
        unprepared = true;
    }
    report.require( unprepared, "Unprepared continuation Jacobian was exposed" );
    f.configure( T( .7 ), T( .04 ) );
    f.set( *f.state, T( .8 ), T( .6 ) );
    f.set( *f.direction, T( .3 ), T( -.4 ) );
    f.op->set_linearization_point( *f.state );
    const auto jacobian = f.op->get_jacobi_operator();
    report.require( jacobian == f.op->get_jacobi_operator(), "Prepared Jacobian handle was not reused" );
    jacobian->apply( *f.direction, *f.action );
    const T epsilon = std::is_same_v<T, float> ? T( .002 ) : T( 1e-5 );
    for ( bool parameter_only : { false, true } )
    {
        f.set( *f.direction, parameter_only ? T( 0 ) : T( .3 ), parameter_only ? T( -.4 ) : T( 0 ) );
        f.space->assign_lin_comb( T( 1 ), *f.state, epsilon, *f.direction, *f.plus );
        f.space->assign_lin_comb( T( 1 ), *f.state, -epsilon, *f.direction, *f.minus );
        f.op->apply( *f.plus, *f.rhs );
        f.op->apply( *f.minus, *f.tangent );
        f.space->assign_lin_comb( T( .5 ) / epsilon, *f.rhs, -T( .5 ) / epsilon, *f.tangent, *f.plus );
        jacobian->apply( *f.direction, *f.action );
        report.require(
            f.difference( *f.plus, *f.action ) < T( 200 ) * fixture_type::tolerance,
            "Continuation Jacobian failed component finite differences"
        );
    }
    jacobian->apply( *f.direction, *f.action );
    f.set( *f.plus, T( 2 ), T( -.2 ) );
    f.op->apply( *f.plus, *f.rhs );
    jacobian->apply( *f.direction, *f.plus );
    report.require( f.difference( *f.plus, *f.action ) == T( 0 ), "Residual trial changed the bordered Jacobian" );

    // Compare both historical S/O conventions with the existing implementation.
    struct unused_solver
    {
    };
    using old_constraint = continuation::system_operator_continuation<
        typename fixture_type::space_type, typename fixture_type::problem_type,
        typename fixture_type::state_operator_type, unused_solver, scfd::utils::log_std>;
    old_constraint old( f.operations.get(), nullptr, f.state_operator.get(), nullptr );
    old.set_verbose( false );
    for ( char type : { 'S', 'O' } )
    {
        f.configure( T( .7 ), T( .04 ), type );
        T parameter = ( *f.reference ).second[0], tangent_parameter = ( *f.tangent ).second[0], step = T( .04 );
        old.set_tangent_space( ( *f.reference ).first, parameter, ( *f.tangent ).first, tangent_parameter, step, type );
        f.op->apply( *f.state, *f.rhs );
        report.require(
            std::abs( ( *f.rhs ).second[0] - old.arclength_residual( ( *f.state ).first, ( *f.state ).second[0] ) ) <=
                fixture_type::tolerance,
            "Arclength residual changed its historical convention"
        );
    }
    const auto allocations = f.operations->allocations;
    for ( int i = 0; i < 10; ++i )
    {
        f.op->set_linearization_point( *f.state );
        f.op->apply( *f.plus, *f.rhs );
    }
    report.require( allocations == f.operations->allocations, "Continuation evaluation allocated vector storage" );
}

// Manufactured nonlinear chart q(u)=u^2, restricted to u>0. Its constraint row
// must include q'(u)=2u; reusing the original tangent as the row is incorrect.
template <class Space>
struct nonlinear_chart_constraint
{
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> operations;
    scalar_type            reference = 0, reference_parameter = 0, tangent = 0, tangent_parameter = 0, step = 0;
    explicit nonlinear_chart_constraint( std::shared_ptr<Space> space ) : operations( std::move( space ) )
    {
    }
    void
    set_tangent_space( const vector_type &x, scalar_type p, const vector_type &t, scalar_type tp, scalar_type h, char )
    {
        operations->get( x, &reference, 1 );
        operations->get( t, &tangent, 1 );
        reference_parameter = p;
        tangent_parameter   = tp;
        step                = h;
    }
    scalar_type apply( const vector_type &state, scalar_type parameter )
    {
        scalar_type u;
        operations->get( state, &u, 1 );
        return ( u * u - reference * reference ) * tangent + ( parameter - reference_parameter ) * tangent_parameter -
               step;
    }
    void linearize( const vector_type &state, scalar_type, vector_type &row, scalar_type &parameter_derivative )
    {
        scalar_type u;
        operations->get( state, &u, 1 );
        operations->assign_scalar( scalar_type( 2 ) * u * tangent, row );
        parameter_derivative = tangent_parameter;
    }
};

void test_chart_and_construction( test_report &report )
{
    using fixture_type  = continuation::tests::circle_fixture<double>;
    using operator_type = continuation::corrector_operator<
        fixture_type::space_type, fixture_type::problem_type, fixture_type::state_operator_type,
        nonlinear_chart_constraint<fixture_type::space_type>>;
    fixture_type  f;
    auto          constraint = std::make_shared<nonlinear_chart_constraint<fixture_type::space_type>>( f.operations );
    operator_type op( f.operations, f.problem, f.state_operator, constraint );
    f.set( *f.reference, 1.1, .1 );
    f.set( *f.tangent, .3, .7 );
    f.set( *f.state, 1.2, .2 );
    f.set( *f.direction, .4, -.5 );
    op.set_tangent_space( ( *f.reference ).first, .1, ( *f.tangent ).first, .7, .05 );
    op.set_linearization_point( *f.state );
    auto frozen = op.get_jacobi_operator();
    frozen->apply( *f.direction, *f.action );
    const double epsilon = 1e-5;
    f.space->assign_lin_comb( 1., *f.state, epsilon, *f.direction, *f.plus );
    f.space->assign_lin_comb( 1., *f.state, -epsilon, *f.direction, *f.minus );
    op.apply( *f.plus, *f.rhs );
    op.apply( *f.minus, *f.tangent );
    f.space->assign_lin_comb( .5 / epsilon, *f.rhs, -.5 / epsilon, *f.tangent, *f.plus );
    report.require( f.difference( *f.plus, *f.action ) < 1e-9, "Nonlinear chart derivative was omitted" );
    frozen->apply( *f.direction, *f.plus );
    report.require( f.difference( *f.plus, *f.action ) == 0, "Chart evaluation changed its frozen derivative" );

    for ( unsigned int activation = 1; activation <= 5; ++activation )
    {
        auto       operations       = std::make_shared<fixture_type::space_type>();
        auto       problem          = std::make_shared<fixture_type::problem_type>( 1., 1, operations );
        auto       state_operator   = std::make_shared<fixture_type::state_operator_type>( problem );
        const auto live             = operations->allocations - operations->frees;
        operations->fail_activation = operations->activations + activation;
        bool failed                 = false;
        try
        {
            auto invalid = std::make_shared<fixture_type::operator_type>( operations, problem, state_operator );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        report.require(
            failed && operations->allocations - operations->frees == live,
            "Partially constructed continuation workspace leaked"
        );
    }
}

// Reflection chart u>=0. Physical equations delegate to circle; raw entry
// points throw so projected and frozen-chart dispatch cannot pass silently.
template <class Space>
struct projected_circle : nonlinear_operators::circle<Space>
{
    using parent_type = nonlinear_operators::circle<Space>;
    using scalar_type = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> operations;
    unsigned int           projected_calls = 0, frozen_calls = 0, preparations = 0, projections = 0;
    scalar_type            last_reference = 0;
    explicit projected_circle( std::shared_ptr<Space> space )
        : parent_type( scalar_type( 1 ), 1, space ), operations( std::move( space ) )
    {
    }
    void F( const vector_type &, scalar_type, vector_type & )
    {
        throw std::logic_error( "Raw residual selected" );
    }
    void set_linearization_point( const vector_type &, scalar_type )
    {
        throw std::logic_error( "Raw linearization selected" );
    }
    void projected_F( const vector_type &x, scalar_type parameter, vector_type &out )
    {
        ++projected_calls;
        parent_type::F( x, parameter, out );
    }
    void projected_F_in_frozen_chart( const vector_type &x, scalar_type parameter, vector_type &out )
    {
        ++frozen_calls;
        parent_type::F( x, parameter, out );
    }
    void set_projected_linearization_point( const vector_type &x, scalar_type parameter )
    {
        ++preparations;
        parent_type::set_linearization_point( x, parameter );
    }
    void projected_jacobian_alpha( vector_type &out )
    {
        parent_type::jacobian_alpha( out );
    }
    void stabilize_corrector_trial( const vector_type &reference, scalar_type, vector_type &trial, scalar_type & )
    {
        ++projections;
        operations->get( reference, &last_reference, 1 );
        operations->make_abs( trial );
    }
};

void test_frozen_projection( test_report &report )
{
    using fixture_type  = continuation::tests::circle_fixture<double>;
    using problem_type  = projected_circle<fixture_type::space_type>;
    using linear_type   = nonlinear_operators::linear_operator_circle<fixture_type::space_type, problem_type>;
    using operator_type = continuation::corrector_operator<fixture_type::space_type, problem_type, linear_type>;
    fixture_type f;
    auto         problem = std::make_shared<problem_type>( f.operations );
    auto         linear  = std::make_shared<linear_type>( problem );
    auto         op      = std::make_shared<operator_type>( f.operations, problem, linear );
    f.configure( .6, .03 );
    op->set_tangent_space(
        ( *f.reference ).first, ( *f.reference ).second[0], ( *f.tangent ).first, ( *f.tangent ).second[0], .03
    );
    f.set( *f.state, .8, .6 );
    f.set( *f.direction, .3, -.2 );
    op->apply( *f.state, *f.rhs );
    op->set_linearization_point( *f.state );
    auto frozen = op->get_jacobi_operator();
    frozen->apply( *f.direction, *f.action );
    f.set( *f.plus, -1.2, .4 );
    op->apply( *f.plus, *f.rhs );
    typename operator_type::trial_projector projector( op );
    projector.apply( *f.plus );
    report.require(
        f.value( *f.plus ) == 1.2 && problem->projections == 1 && problem->last_reference == .8,
        "Trial projection did not use the unchanged Newton base iterate"
    );
    report.require(
        problem->projected_calls == 1 && problem->frozen_calls == 1 && problem->preparations == 1,
        "Residual trial refroze the prepared chart or bypassed projected entry points"
    );
    frozen->apply( *f.direction, *f.plus );
    report.require( f.difference( *f.plus, *f.action ) == 0, "Trial projection changed the frozen bordered operator" );
    bool rejected = false;
    try
    {
        projector.apply( *f.state );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    report.require( rejected && f.value( *f.state ) == .8, "Trial projection modified its borrowed base iterate" );
}

void test_action_failure( test_report &report )
{
    using fixture_type = continuation::tests::circle_fixture<double>;
    struct status_operator
    {
        using vector_type = typename fixture_type::space_type::vector_type;
        std::shared_ptr<const fixture_type::state_operator_type> original;
        bool                                                     fail = false;
        bool                                                     apply( const vector_type &in, vector_type &out ) const
        {
            if ( fail )
                return false;
            original->apply( in, out );
            return true;
        }
    };
    fixture_type f;
    auto         original = std::make_shared<status_operator>( status_operator{ f.state_operator } );
    continuation::corrector_operator<fixture_type::space_type, fixture_type::problem_type, status_operator> op(
        f.operations, f.problem, original
    );
    f.configure( .2, .03 );
    op.set_tangent_space(
        ( *f.reference ).first, ( *f.reference ).second[0], ( *f.tangent ).first, ( *f.tangent ).second[0], .03
    );
    op.set_linearization_point( *f.reference );
    report.require( op.get_jacobi_operator()->apply( *f.tangent, *f.action ), "Valid state-operator action failed" );
    original->fail = true;
    report.require(
        !op.get_jacobi_operator()->apply( *f.tangent, *f.action ), "Bordered action ignored a state-operator failure"
    );
    bool rejected = false;
    try
    {
        op.get_jacobi_operator()->apply( *f.tangent, *f.tangent );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    report.require( rejected, "Bordered Jacobian silently accepted aliased input and output" );
}
}

int main()
{
    continuation::tests::test_report report;
    try
    {
        test_residual_norm<float>( report );
        test_residual_norm<double>( report );
        test_operator<float>( report );
        test_operator<double>( report );
        test_chart_and_construction( report );
        test_frozen_projection( report );
        test_action_failure( report );
        std::printf( "Continuation operator: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Continuation operator failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
