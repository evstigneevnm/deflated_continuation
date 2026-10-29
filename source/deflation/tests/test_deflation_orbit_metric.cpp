#include "deflation_newton_fixture.h"

#include <array>
#include <symmetry/finite_quotient_adapter.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

namespace
{

// Three even Fourier modes admit a residual quarter shift (-,+,-). Additional
// odd modes model weak incompatible noise. Global negation exercises composition
// with a finite action. RMS distances pair with Euclidean gradients using weight N.
class space_type : public deflation::tests::vector_space<double>
{
public:
    using base_type = deflation::tests::vector_space<double>;
    space_type() : base_type( 17 )
    {
    }
    void get( const vector_type &, double * ) const              = delete;
    void get( const vector_type &, double *, std::size_t ) const = delete;
    void set( const double *, vector_type & ) const              = delete;
    void set( const double *, vector_type &, std::size_t ) const = delete;
};

using buffer_type   = nmfd::detail::vector_wrap<space_type, false, false>;
using identity_type = deflation::identity_symmetry_adapter<space_type>;
using actions_type  = symmetry::finite_action_registry<space_type>;
using aligner_type  = symmetry::fourier::residual_translation_orbit_aligner_2d<space_type>;
using quotient_type = symmetry::finite_quotient_adapter<space_type, identity_type, aligner_type>;
using storage_type  = deflation::symmetry_solution_storage<
     space_type, quotient_type, void, deflation::symmetry_deflation_metric::nearest_orbit>;
static_assert( storage_type::deflation_metric == deflation::symmetry_deflation_metric::nearest_orbit );
static_assert(
    deflation::symmetry_solution_storage<space_type, quotient_type>::deflation_metric ==
    deflation::symmetry_deflation_metric::canonical
);

void test_metric( deflation::tests::test_report &report )
{
    auto space = std::make_shared<space_type>();
    space->use_high_precision();
    buffer_type root( *space ), second( *space ), query( *space ), shifted( *space ), signs( *space ),
        gradient( *space ), saved_gradient( *space ), work( *space ), direction( *space ), plus( *space ),
        minus( *space );
    for ( auto *buffer :
          { &root, &second, &query, &shifted, &signs, &gradient, &saved_gradient, &work, &direction, &plus, &minus } )
    {
        buffer->init();
        buffer->start_use();
    }
    const std::array<double, 17> r{ 2., 1., .5 }, q{ 1.6, 1.2, .6 }, r2{ -.4, .8, 1.2 }, v{ .2, -.3, .1 };
    std::array<double, 17>       sign;
    sign.fill( 1. );
    sign[0] = sign[2] = -1.;
    space->base_type::set( r.data(), *root );
    space->base_type::set( r2.data(), *second );
    space->base_type::set( q.data(), *query );
    space->base_type::set( v.data(), *direction );
    space->base_type::set( sign.data(), *signs );

    std::vector<symmetry::fourier::mode_index<2>> modes( 17, { 1, 0 } );
    modes[0] = { 2, 0 };
    modes[1] = { 4, 0 };
    modes[2] = { 6, 0 };
    aligner_type::policy_type policy;
    policy.absolute_active_tolerance = 1e-8;
    auto aligner                     = std::make_shared<aligner_type>( space, modes, policy );
    auto actions                     = std::make_shared<actions_type>( space );
    actions->add( "negation", [space]( const auto &in, auto &out ) { space->assign_mul( -1., in, out ); } );
    auto quotient =
        std::make_shared<quotient_type>( space, std::make_shared<identity_type>( space ), actions, aligner );
    storage_type storage( space, 4, 17., 2., quotient, 1e-10 );
    storage.set_ignore_zero();
    double multiplier;
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        multiplier == 1. && space->norm( *gradient ) == 0. && std::isinf( storage.last_minimum_distance() ),
        "Empty orbit storage changed deflation"
    );
    storage.push_back( *root );
    storage.calc_distance( *query, multiplier, *saved_gradient );
    const double base_multiplier   = multiplier;
    const double expected_distance = std::sqrt( .21 / 17. );
    report.require( std::abs( multiplier - ( 1. + 17. / .21 ) ) < 2e-12, "Orbit multiplier changed its formula" );
    report.require(
        std::abs( storage.last_minimum_distance() - expected_distance ) < 1e-14, "Orbit exclusion distance is not RMS"
    );
    const auto allocations = space->allocations;

    for ( double finite_sign : { 1., -1. } )
    {
        space->mul_pointwise( finite_sign, *query, 1., *signs, *shifted );
        storage.calc_distance( *shifted, multiplier, *gradient );
        const auto minimum = storage.last_minimum_distance();
        report.require(
            std::abs( multiplier - base_multiplier ) < 2e-12, "Equivalent queries have different multipliers"
        );
        report.require(
            std::abs( minimum - storage.nearest_stabilized_distance( *shifted ) ) < 1e-14,
            "Multiplier and duplicate detection use different orbit distances"
        );
        space->mul_pointwise( finite_sign, *saved_gradient, 1., *signs, *work );
        space->add_mul( -1., *gradient, *work );
        report.require( space->norm( *work ) < 1e-10, "Orbit multiplier gradient is not equivariant" );
    }

    // A noisy odd mode crosses the active-mode threshold without exceeding the permitted RMS defect.
    space->mul_pointwise( 1., *query, 1., *signs, *shifted );
    space->set_value_at_point( 1.5e-8, 16, *shifted );
    storage.calc_distance( *shifted, multiplier, *gradient );
    const auto noisy_distance = storage.last_minimum_distance();
    report.require(
        std::abs( multiplier - base_multiplier ) < 2e-11 &&
            std::abs( noisy_distance - storage.nearest_stabilized_distance( *shifted ) ) < 1e-14,
        "Weak incompatible noise changed the deflation orbit"
    );

    space->mul_pointwise( 1., *root, 1., *signs, *shifted );
    storage.calc_distance( *shifted, multiplier, *gradient );
    report.require(
        std::isinf( multiplier ) && storage.last_minimum_distance() == 0. && space->is_valid_number( *gradient ) &&
            space->norm( *gradient ) == 0.,
        "Exact stored orbit did not return an explicit singular multiplier"
    );
    space->assign_mul( .99, *shifted, *query );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        std::isfinite( multiplier ) && storage.last_minimum_distance() > 1e-4, "A nearby distinct state was excluded"
    );
    report.require( space->allocations == allocations, "Orbit evaluation allocated solution vectors" );

    // Multiple roots preserve the averaged inverse-power multiplier and differentiate each selected action.
    space->base_type::set( q.data(), *query );
    const double second_distance = storage.canonical_distance( *second, *query );
    storage.push_back( *second );
    storage.calc_distance( *query, multiplier, *saved_gradient );
    const double combined_multiplier = 1. + .5 * ( base_multiplier - 1. + 1. / ( second_distance * second_distance ) );
    report.require( std::abs( multiplier - combined_multiplier ) < 2e-12, "Orbit deflation changed root averaging" );
    const auto check_derivative = [&]( const space_type::vector_type &state, const space_type::vector_type &v ) {
        storage.calc_distance( state, multiplier, *gradient );
        const double derivative     = -space->scalar_prod( *gradient, v );
        double       previous_error = std::numeric_limits<double>::infinity();
        for ( double h : { 1e-3, 5e-4, 2.5e-4 } )
        {
            space->assign_mul( 1., state, h, v, *plus );
            space->assign_mul( 1., state, -h, v, *minus );
            double m_plus, m_minus;
            storage.calc_distance( *plus, m_plus, *gradient );
            storage.calc_distance( *minus, m_minus, *gradient );
            const double error = std::abs( ( m_plus - m_minus ) / ( 2 * h ) - derivative );
            report.require( error < .4 * previous_error, "Orbit gradient does not converge in finite differences" );
            previous_error = error;
        }
        report.require( previous_error < 1e-4, "Orbit negative gradient failed RMS/Euclidean pairing" );
    };
    check_derivative( *query, *direction );
    for ( double finite_sign : { 1., -1. } )
    {
        space->mul_pointwise( finite_sign, *query, 1., *signs, *shifted );
        space->mul_pointwise( finite_sign, *direction, 1., *signs, *work );
        check_derivative( *shifted, *work );
    }

    storage.clear();
    storage.set_known_solution( *root );
    storage.push_back( *second );
    storage.calc_distance( *query, multiplier, *gradient );
    space->add_mul( -1., *saved_gradient, *gradient );
    report.require(
        std::abs( multiplier - combined_multiplier ) < 2e-12 && space->norm( *gradient ) < 1e-10,
        "Optional base root uses a different orbit metric"
    );

    // On a tie the scalar distance remains defined; no unique gradient is asserted.
    space->assign_scalar( 0., *query );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        std::isfinite( multiplier ) && space->is_valid_number( *gradient ), "Orbit tie produced invalid output"
    );
    const double tie_multiplier = multiplier;
    storage.calc_distance( *query, multiplier, *work );
    space->add_mul( -1., *gradient, *work );
    report.require(
        multiplier == tie_multiplier && space->norm( *work ) == 0., "Orbit tie breaking is not deterministic"
    );
}

} // namespace

int main()
{
    deflation::tests::test_report report;
    try
    {
        test_metric( report );
        std::printf( "Orbit deflation metric: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Orbit deflation metric failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
