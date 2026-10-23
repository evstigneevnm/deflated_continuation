#include "deflation_newton_fixture.h"

namespace
{
using deflation::tests::test_report;

// Circle equilibrium F(u,lambda)=u^2+lambda^2-1. Deflating a known nonzero
// root gives D=mF and J_D=mJ_F+F Dm, not merely a scaled physical Jacobian.
template <class T>
void test_operator( test_report &report )
{
    using fixture_type = deflation::tests::circle_fixture<T>;
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
    report.require( unprepared, "Unprepared deflated Jacobian was exposed" );
    f.space->assign_scalar( T( 1 ), *f.initial );
    f.solutions->set_known_solution( *f.initial );
    f.space->assign_scalar( T( -2 ), *f.initial );
    f.solutions->push_back( *f.initial );
    f.space->assign_scalar( T( .4 ), *f.state );
    f.space->assign_scalar( T( .7 ), *f.direction );
    f.op->set_linearization_point( *f.state );
    const auto jacobian    = f.op->get_jacobi_operator();
    const auto allocations = f.space->allocations;
    const T    epsilon     = std::is_same_v<T, float> ? T( .001 ) : T( 1e-5 );
    f.space->assign_lin_comb( T( 1 ), *f.state, epsilon, *f.direction, *f.plus );
    f.space->assign_lin_comb( T( 1 ), *f.state, -epsilon, *f.direction, *f.minus );
    f.op->apply( *f.plus, *f.residual );
    f.op->apply( *f.minus, *f.action );
    f.space->assign_lin_comb( T( .5 ) / epsilon, *f.residual, -T( .5 ) / epsilon, *f.action, *f.plus );
    report.require( jacobian->apply( *f.direction, *f.action ), "Deflated Jacobian action failed" );
    f.space->add_lin_comb( T( -1 ), *f.action, *f.plus );
    report.require(
        f.space->norm_l2( *f.plus ) < T( 1000 ) * fixture_type::tolerance,
        "Deflated Jacobian failed finite differences at a nonzero known solution"
    );
    f.op->apply( *f.state, *f.residual );
    const T physical_norm = f.op->original_residual_norm( *f.state, *f.plus );
    report.require(
        std::abs( physical_norm - T( .84 ) ) < fixture_type::tolerance,
        "Convergence assessment did not use the physical residual"
    );
    report.require(
        f.space->norm_l2( *f.residual ) > T( 2 ) * physical_norm,
        "Deflation test did not distinguish physical and deflated norms"
    );
    f.space->assign_scalar( T( -.4 ), *f.minus );
    f.op->apply( *f.minus, *f.residual );
    f.op->original_residual_norm( *f.minus, *f.residual );
    jacobian->apply( *f.direction, *f.plus );
    f.space->add_lin_comb( T( -1 ), *f.action, *f.plus );
    report.require( f.space->norm_l2( *f.plus ) == T( 0 ), "Trials changed the frozen deflated Jacobian" );
    report.require( f.op->get_jacobi_operator().get() == jacobian.get(), "Jacobian retrieval replaced storage" );
    report.require( f.space->allocations == allocations, "Deflation evaluation allocated vectors" );

    for ( T root : { T( 1 ), T( 1 ) - fixture_type::tolerance, T( -2 ) } )
    {
        f.space->assign_scalar( root, *f.state );
        report.require(
            !std::isfinite( f.op->original_residual_norm( *f.state, *f.residual ) ),
            "Excluded root or its tolerance neighborhood was accepted"
        );
    }
    f.solutions->clear();
    f.solutions->set_ignore_zero();
    f.space->assign_scalar( T( 1 ), *f.state );
    report.require(
        f.op->original_residual_norm( *f.state, *f.residual ) == T( 0 ), "Empty storage excluded an ordinary root"
    );
    f.op->set_parameter( T( .5 ) );
    unprepared = false;
    try
    {
        f.op->get_jacobi_operator();
    }
    catch ( const std::logic_error & )
    {
        unprepared = true;
    }
    report.require( unprepared && f.op->parameter() == T( .5 ), "Parameter change retained a prepared Jacobian" );
}

// In two dimensions norm_l2 is RMS but scalar_prod is unnormalized. With
// norm_weight=N the stored negative gradient has the correct Euclidean pairing.
void test_weighted_storage( test_report &report )
{
    using space_type                                          = deflation::tests::vector_space<double>;
    auto                                                space = std::make_shared<space_type>( 2 );
    nmfd::detail::vector_wrap<space_type, false, false> state( *space ), base( *space ), direction( *space ),
        gradient( *space ), plus( *space ), minus( *space );
    for ( auto *storage : { &state, &base, &direction, &gradient, &plus, &minus } )
    {
        storage->init();
        storage->start_use();
    }
    const double u[] = { .4, .7 }, root[] = { 1., -.3 }, v[] = { .2, -.4 };
    space->set( u, *state, 2 );
    space->set( root, *base, 2 );
    space->set( v, *direction, 2 );
    auto adapter = std::make_shared<deflation::identity_symmetry_adapter<space_type>>( space );
    deflation::solution_storage<space_type>          ordinary( space.get(), 2, 2. );
    deflation::symmetry_solution_storage<space_type> symmetric( space, 2, 2., 2., adapter );
    ordinary.set_known_solution( *base );
    symmetric.set_known_solution( *base );
    const double epsilon = 1e-5;
    space->assign_lin_comb( 1., *state, epsilon, *direction, *plus );
    space->assign_lin_comb( 1., *state, -epsilon, *direction, *minus );
    double beta, beta_plus, beta_minus;
    ordinary.calc_distance( *state, beta, *gradient );
    const double derivative = -space->scalar_prod( *gradient, *direction );
    ordinary.calc_distance( *plus, beta_plus, *gradient );
    ordinary.calc_distance( *minus, beta_minus, *gradient );
    report.require(
        std::abs( derivative - ( beta_plus - beta_minus ) / ( 2 * epsilon ) ) < 1e-9,
        "Normalized storage gradient failed finite differences"
    );
    symmetric.calc_distance( *state, beta_plus, *gradient );
    report.require(
        std::abs( beta - beta_plus ) < 1e-14 &&
            std::abs( derivative + space->scalar_prod( *gradient, *direction ) ) < 1e-12,
        "Ordinary and identity-symmetry storage disagree"
    );
}

// Canonical reflection u -> |u| identifies both signs as the same orbit.
template <class Space>
struct reflection_adapter
{
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    double                 sign( const vector_type &x ) const
    {
        double value;
        space->get( x, &value, 1 );
        return value < 0 ? -1. : 1.;
    }
    void stabilize( const vector_type &in, vector_type &out ) const
    {
        space->assign_mul( sign( in ), in, out );
    }
    void pullback_distance_gradient(
        const vector_type &source, const vector_type &, const vector_type &gradient, vector_type &out
    ) const
    {
        space->assign_mul( sign( source ), gradient, out );
    }
};

void test_symmetry_and_construction( test_report &report )
{
    using fixture_type   = deflation::tests::circle_fixture<double>;
    using space_type     = fixture_type::space_type;
    using adapter_type   = reflection_adapter<space_type>;
    using solutions_type = deflation::symmetry_solution_storage<space_type, adapter_type>;
    using operator_type  = deflation::deflated_operator<
         space_type, fixture_type::problem_type, fixture_type::state_operator_type, solutions_type>;
    fixture_type f;
    auto         adapter = std::make_shared<adapter_type>();
    adapter->space       = f.space;
    auto solutions       = std::make_shared<solutions_type>( f.space, 2, 1., 2., adapter );
    solutions->set_ignore_zero();
    f.space->assign_scalar( 1., *f.initial );
    solutions->push_back( *f.initial );
    operator_type op( f.space, f.problem, f.state_operator, solutions, 0., 1e-10 );
    f.space->assign_scalar( -1., *f.state );
    report.require(
        !std::isfinite( op.original_residual_norm( *f.state, *f.residual ) ),
        "Reflection-equivalent stored root was not excluded"
    );
    f.space->assign_scalar( -.4, *f.state );
    op.set_linearization_point( *f.state );
    f.space->assign_scalar( .7, *f.direction );
    op.get_jacobi_operator()->apply( *f.direction, *f.action );
    f.space->assign_lin_comb( 1., *f.state, 1e-5, *f.direction, *f.plus );
    f.space->assign_lin_comb( 1., *f.state, -1e-5, *f.direction, *f.minus );
    op.apply( *f.plus, *f.residual );
    op.apply( *f.minus, *f.plus );
    f.space->assign_lin_comb( 5e4, *f.residual, -5e4, *f.plus, *f.minus );
    f.space->add_lin_comb( -1., *f.action, *f.minus );
    report.require( f.space->norm_l2( *f.minus ) < 1e-8, "Symmetry pullback failed deflated finite differences" );

    for ( unsigned int failure = 1; failure <= 3; ++failure )
    {
        const auto allocations = f.space->allocations, frees = f.space->frees;
        f.space->fail_activation = f.space->activations + failure;
        bool caught              = false;
        try
        {
            operator_type broken( f.space, f.problem, f.state_operator, solutions, 0., 1e-10 );
        }
        catch ( const std::runtime_error & )
        {
            caught = true;
        }
        report.require(
            caught && f.space->allocations - allocations == f.space->frees - frees,
            "Partially constructed deflation workspace leaked"
        );
    }
    f.space->fail_activation = 0;
}

// Projected entry points must preserve the chart prepared for this correction.
template <class Space>
struct projected_circle : nonlinear_operators::circle<Space>
{
    using parent_type = nonlinear_operators::circle<Space>;
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> space;
    const vector_type     *base            = nullptr;
    double                 parameter       = 0;
    unsigned int           projected_calls = 0, frozen_calls = 0, projections = 0;
    explicit projected_circle( std::shared_ptr<Space> operations )
        : parent_type( 1., 1, operations ), space( std::move( operations ) )
    {
    }
    void F( const vector_type &, double, vector_type & )
    {
        throw std::logic_error( "Raw residual selected" );
    }
    void set_linearization_point( const vector_type &, double )
    {
        throw std::logic_error( "Raw linearization selected" );
    }
    void projected_F( const vector_type &state, double parameter, vector_type &out )
    {
        ++projected_calls;
        parent_type::F( state, parameter, out );
    }
    void projected_F_in_frozen_chart( const vector_type &state, double parameter, vector_type &out )
    {
        ++frozen_calls;
        parent_type::F( state, parameter, out );
    }
    void projected_F_at_linearization( vector_type &out )
    {
        parent_type::F( *base, parameter, out );
    }
    void set_projected_linearization_point( const vector_type &state, double value )
    {
        base      = &state;
        parameter = value;
        parent_type::set_linearization_point( state, value );
    }
    void project( vector_type &state )
    {
        ++projections;
        space->make_abs( state );
    }
};

template <class Space, class Problem>
struct checked_linear_operator : nonlinear_operators::linear_operator_circle<Space, Problem>
{
    using parent_type = nonlinear_operators::linear_operator_circle<Space, Problem>;
    using vector_type = typename Space::vector_type;
    bool fail         = false;
    explicit checked_linear_operator( std::shared_ptr<Problem> problem ) : parent_type( std::move( problem ) )
    {
    }
    bool apply( const vector_type &in, vector_type &out ) const
    {
        if ( fail )
            return false;
        parent_type::apply( in, out );
        return true;
    }
};

void test_projected_operator( test_report &report )
{
    using fixture_type = deflation::tests::circle_fixture<double>;
    using space_type   = fixture_type::space_type;
    using problem_type = projected_circle<space_type>;
    using linear_type  = checked_linear_operator<space_type, problem_type>;
    using operator_type =
        deflation::deflated_operator<space_type, problem_type, linear_type, fixture_type::solutions_type>;
    fixture_type  f;
    auto          problem = std::make_shared<problem_type>( f.space );
    auto          linear  = std::make_shared<linear_type>( problem );
    operator_type op( f.space, problem, linear, f.solutions, .2, 1e-10 );
    f.space->assign_scalar( .8, *f.state );
    f.space->assign_scalar( .3, *f.direction );
    op.apply( *f.state, *f.residual );
    op.set_linearization_point( *f.state );
    const auto jacobian = op.get_jacobi_operator();
    jacobian->apply( *f.direction, *f.action );
    f.space->assign_scalar( -1.2, *f.plus );
    typename operator_type::trial_projector projector( problem );
    projector.apply( *f.plus );
    op.apply( *f.plus, *f.residual );
    op.original_residual_norm( *f.plus, *f.residual );
    jacobian->apply( *f.direction, *f.minus );
    f.space->add_lin_comb( -1., *f.action, *f.minus );
    report.require(
        f.value( *f.plus ) == 1.2 && problem->projections == 1, "Deflation trial projector bypassed the problem"
    );
    report.require(
        problem->projected_calls == 1 && problem->frozen_calls == 2 && f.space->norm_l2( *f.minus ) == 0,
        "Projected trials changed the frozen deflated operator"
    );
    linear->fail = true;
    report.require( !jacobian->apply( *f.direction, *f.action ), "Deflated Jacobian hid a failed state action" );
    bool rejected = false;
    try
    {
        jacobian->apply( *f.state, *f.state );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    report.require( rejected, "Aliased deflated Jacobian action was accepted" );
}
}

int main()
{
    test_report report;
    try
    {
        test_operator<float>( report );
        test_operator<double>( report );
        test_weighted_storage( report );
        test_symmetry_and_construction( report );
        test_projected_operator( report );
        std::printf( "Deflation operator: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Deflation operator failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
