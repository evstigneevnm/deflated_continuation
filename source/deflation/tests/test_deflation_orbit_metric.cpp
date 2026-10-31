#include "deflation_newton_fixture.h"

#include <array>
#include <continuation/tangent_normalization.h>
#include <main/deflation_continuation/rejected_candidate_cache.h>
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
    explicit space_type( std::size_t size = 17 ) : base_type( size )
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

struct counting_aligner : aligner_type
{
    using aligner_type::aligner_type;
    std::size_t          preparations = 0;
    prepared_source_type prepare_source_orbit( const vector_type &source )
    {
        ++preparations;
        return aligner_type::prepare_source_orbit( source );
    }
};
static_assert( storage_type::deflation_metric == deflation::symmetry_deflation_metric::nearest_orbit );
static_assert(
    deflation::symmetry_solution_storage<space_type, quotient_type>::deflation_metric ==
    deflation::symmetry_deflation_metric::canonical
);

// State residuals use RMS norms, but endpoint directions use the same
// augmented Euclidean pairing as continuation's tangent normalization.
void test_tangent_similarity( deflation::tests::test_report &report )
{
    space_type space( 4 );
    space.use_high_precision();
    buffer_type reference( space ), source( space );
    for ( auto *buffer : { &reference, &source } )
    {
        buffer->init();
        buffer->start_use();
        space.assign_scalar( 1., **buffer );
    }
    const auto similarity = [&]( double reference_parameter = 0., double source_parameter = 0. ) {
        return continuation::rank1_tangent_line_similarity(
            &space, *reference, reference_parameter, *source, source_parameter
        );
    };
    report.require(
        space.norm_l2( *reference ) == 1. && space.scalar_prod( *reference, *reference ) == 4.,
        "Tangent fixture must use RMS residuals and an unnormalized dot product"
    );
    report.require( std::abs( similarity() - 1. ) < 1e-14, "Parallel tangent lines did not match" );
    space.assign_mul( -3., *reference, *source );
    report.require( std::abs( similarity() - 1. ) < 1e-14, "Opposite rescaled tangent lines did not match" );
    space.assign_scalar( 1., *source );
    space.set_value_at_point( -1., 3, *source );
    report.require(
        std::abs( similarity() - .5 ) < 1e-14, "RMS normalization falsely joined transverse tangent lines"
    );
    report.require( std::abs( similarity( 2., 2. ) - .75 ) < 1e-14, "Parameter tangent has a different metric weight" );
    space.scale( 3., *source );
    report.require( std::abs( similarity( 2., 6. ) - .75 ) < 1e-14, "Extended tangent rescaling changed similarity" );
    space.assign_scalar( 1., *source );
    space.set_value_at_point( -1., 2, *source );
    space.set_value_at_point( -1., 3, *source );
    report.require( similarity() == 0., "Orthogonal tangent lines were joined" );
    space.assign_scalar( 0., *source );
    report.require( similarity() == 0., "Zero extended tangent was joined" );
    report.require(
        similarity( std::numeric_limits<double>::quiet_NaN(), 1. ) == 0., "Nonfinite parameter was joined"
    );
    space.set_value_at_point( std::numeric_limits<double>::infinity(), 0, *source );
    report.require( similarity() == 0., "Nonfinite state tangent was joined" );
}

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
    {
        main_classes::deflation_continuation_detail::rejected_candidate_cache<space_type> rejected(
            space.get(), [&storage]( const auto &stored, const auto &candidate ) {
                return storage.canonical_distance( stored, candidate );
            }
        );
        rejected.add( 17., *root );
        const auto cache_allocations = space->allocations;
        space->mul_pointwise( -1., *root, 1., *signs, *shifted );
        space->assign_mul( 1., *root, -1., *shifted, *work );
        report.require( space->norm_l2( *work ) > .1, "Rejection fixture must need finite and translation actions" );
        double distance;
        report.require(
            rejected.nearest_distance( 17., *shifted, distance ) && distance < 1e-10,
            "Session cache missed a finite-action/quarter-shift duplicate"
        );
        report.require(
            std::abs( distance - storage.nearest_stabilized_distance( *shifted ) ) < 1e-14,
            "Session rejection and known-root checks use different orbit metrics"
        );
        space->scale( .99, *shifted );
        report.require(
            rejected.nearest_distance( 17., *shifted, distance ) && distance > 1e-4,
            "Session cache rejected a nearby distinct root"
        );
        report.require( !rejected.nearest_distance( 19., *shifted, distance ), "Session cache mixed different knots" );
        report.require( space->allocations == cache_allocations, "Session orbit queries allocated state vectors" );
    }
    storage.calc_distance( *query, multiplier, *saved_gradient );
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

// A zero equation isolates known-root exclusion from residual acceptance.
struct zero_equation
{
    std::shared_ptr<space_type> space;
    explicit zero_equation( std::shared_ptr<space_type> operations ) : space( std::move( operations ) )
    {
    }
    void F( const space_type::vector_type &, double, space_type::vector_type &out )
    {
        space->assign_scalar( 0., out );
    }
    void apply( const space_type::vector_type &, space_type::vector_type &out ) const
    {
        space->assign_scalar( 0., out );
    }
};

// The stored even-mode orbit admits a quarter shift. Odd-mode noise belongs to
// the query, which is not transformed, and must not make that stored image
// disappear when the old two-vector compatibility threshold is crossed.
void test_source_orbit_cutoff( deflation::tests::test_report &report, bool rank_two )
{
    auto space = std::make_shared<space_type>();
    space->use_high_precision();
    buffer_type root( *space ), query( *space ), gradient( *space ), work( *space ), plus( *space ), minus( *space );
    for ( auto *buffer : { &root, &query, &gradient, &work, &plus, &minus } )
    {
        buffer->init();
        buffer->start_use();
    }
    const std::array<double, 17> values{ 2., 1., .5 }, translated{ -2., 1., -.5 };
    space->base_type::set( values.data(), *root );
    space->base_type::set( translated.data(), *query );
    auto modes = std::vector<symmetry::fourier::mode_index<2>>( 17, { 1, 0 } );
    modes[0]   = { 2, 0 };
    modes[1]   = rank_two ? symmetry::fourier::mode_index<2>{ 0, 2 } : symmetry::fourier::mode_index<2>{ 4, 0 };
    modes[2]   = rank_two ? symmetry::fourier::mode_index<2>{ 2, 2 } : symmetry::fourier::mode_index<2>{ 6, 0 };
    aligner_type::policy_type policy;
    policy.absolute_active_tolerance = 1e-8;
    auto actions                     = std::make_shared<actions_type>( space );
    auto quotient                    = std::make_shared<quotient_type>(
        space, std::make_shared<identity_type>( space ), actions, std::make_shared<aligner_type>( space, modes, policy )
    );
    auto storage = std::make_shared<storage_type>( space, 2, 17., 2., quotient, 1e-10 );
    storage->set_ignore_zero();
    storage->push_back( *root );
    // A zero residual must not bypass the independent known-root exclusion.
    auto problem = std::make_shared<zero_equation>( space );
    deflation::deflated_operator<space_type, zero_equation, zero_equation, storage_type> op(
        space, problem, problem, storage, 0., 1e-6
    );
    const auto   allocations = space->allocations;
    const double boundary    = .5 * std::sqrt( 17. ) * policy.absolute_active_tolerance;
    for ( double factor : { 1. - 1e-6, 1. + 1e-6, 2. } )
    {
        const double noise = boundary * factor;
        space->set_value_at_point( noise, 16, *query );
        double multiplier;
        storage->calc_distance( *query, multiplier, *gradient );
        const double expected = noise / std::sqrt( 17. );
        report.require(
            std::abs( storage->last_minimum_distance() - expected ) < 1e-20,
            "Query cutoff changed the stored orbit distance"
        );
        report.require(
            std::abs( storage->nearest_stabilized_distance( *query ) - expected ) < 1e-20 &&
                std::abs( storage->canonical_distance( *root, *query ) - expected ) < 1e-20,
            "Cutoff multiplier and novelty queries disagree"
        );
        report.require(
            std::abs( multiplier / ( 1. + 1. / ( expected * expected ) ) - 1. ) < 1e-12,
            "Query cutoff changed the deflation multiplier"
        );
        report.require(
            std::isinf( op.deflated_residual_norm( *query, *work ) ),
            "A near known orbit passed exclusion across the query cutoff"
        );
        const double h = noise * 1e-4;
        space->assign( *query, *plus );
        space->assign( *query, *minus );
        space->set_value_at_point( noise + h, 16, *plus );
        space->set_value_at_point( noise - h, 16, *minus );
        space->assign_mul( .5 / h, *plus, -.5 / h, *minus, *work );
        const double derivative = -space->scalar_prod( *gradient, *work );
        double       upper, lower;
        storage->calc_distance( *plus, upper, *gradient );
        storage->calc_distance( *minus, lower, *gradient );
        report.require(
            std::abs( ( upper - lower ) / ( 2 * h ) / derivative - 1. ) < 1e-6,
            "Deflation gradient is discontinuous across the query cutoff"
        );
    }
    space->set_value_at_point( 1e-4, 16, *query );
    report.require( storage->nearest_stabilized_distance( *query ) > 1e-6, "A distinct query was excluded" );
    report.require( space->allocations == allocations, "Cutoff queries allocated state vectors" );
}

// Reproduce the fresh-root noise defect at the benchmark dimension, without FFTs.
// Only a few even Fourier coefficients are nonzero; weak (1,1) noise must not
// suppress an admissible y quarter shift. Prepared metadata must survive queries,
// but not root replacement, adapter rebinding, registry changes, or clear().
void test_stored_source_preparation( deflation::tests::test_report &report )
{
    constexpr std::size_t size = 32766;
    using prepared_quotient    = symmetry::finite_quotient_adapter<space_type, identity_type, counting_aligner>;
    using prepared_storage     = deflation::symmetry_solution_storage<
            space_type, prepared_quotient, void, deflation::symmetry_deflation_metric::nearest_orbit>;
    auto space = std::make_shared<space_type>( size );
    space->use_high_precision();
    buffer_type root( *space ), query( *space ), gradient( *space );
    for ( auto *buffer : { &root, &query, &gradient } )
    {
        buffer->init();
        buffer->start_use();
        space->assign_scalar( 0., **buffer );
    }
    const auto put = [&]( auto &x, std::size_t i, double value ) { space->set_value_at_point( value, i, x ); };
    put( *root, 0, 2. );
    put( *root, 1, 1. );
    put( *root, 2, .5 );
    put( *root, 3, 1.2e-8 );
    put( *query, 0, 2. );
    put( *query, 1, -1. );
    put( *query, 2, -.5 );
    std::vector<symmetry::fourier::mode_index<2>> modes( size, { 1, 1 } );
    modes[0] = { 2, 0 };
    modes[1] = { 0, 2 };
    modes[2] = { 2, 2 };
    aligner_type::policy_type policy;
    policy.absolute_active_tolerance = 1e-8;
    auto aligner                     = std::make_shared<counting_aligner>( space, modes, policy );
    auto actions                     = std::make_shared<actions_type>( space );
    actions->add( "negation", [space]( const auto &in, auto &out ) { space->assign_mul( -1., in, out ); } );
    auto quotient =
        std::make_shared<prepared_quotient>( space, std::make_shared<identity_type>( space ), actions, aligner );
    prepared_storage storage( space, 4, double( size ), 2., quotient, 1e-6 );
    storage.set_ignore_zero();
    storage.push_back( *root );
    const auto   preparations = aligner->preparations;
    const auto   allocations  = space->allocations;
    const double expected     = 1.2e-8 / std::sqrt( double( size ) );
    double       multiplier;
    for ( int i = 0; i < 3; ++i )
    {
        storage.calc_distance( *query, multiplier, *gradient );
        report.require(
            std::abs( storage.last_minimum_distance() - expected ) < 1e-22, "Fresh stored-root noise hid a duplicate"
        );
        report.require(
            std::abs( storage.nearest_stabilized_distance( *query ) - expected ) < 1e-22,
            "Cached novelty and multiplier distances disagree"
        );
    }
    report.require(
        aligner->preparations == preparations && space->allocations == allocations,
        "Cached queries rebuilt support or allocated state vectors"
    );
    storage.push_back( *query );
    report.require(
        storage.get_size() == 1 && storage.duplicates_skipped_since_clear() == 1,
        "Fresh-root symmetry duplicate was inserted"
    );
    put( *query, 0, 2.001 );
    report.require( storage.nearest_stabilized_distance( *query ) > 1e-6, "Nearby distinct root was excluded" );
    put( *query, 0, 2. );
    actions->add( "identity_again", [space]( const auto &in, auto &out ) { space->assign( in, out ); } );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        aligner->preparations == preparations + actions->size() &&
            std::abs( storage.last_minimum_distance() - expected ) < 1e-22,
        "Registry mutation retained stale root metadata"
    );
    actions->clear();
    bool empty_registry_rejected = false;
    try
    {
        storage.calc_distance( *query, multiplier, *gradient );
    }
    catch ( const std::logic_error & )
    {
        empty_registry_rejected = true;
    }
    report.require( empty_registry_rejected && storage.get_size() == 1, "An empty registry reused a stale root image" );
    actions->reset_to_identity();
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        std::abs( storage.last_minimum_distance() - expected ) < 1e-22,
        "Registry restoration did not recover prepared queries"
    );
    auto replacement = std::make_shared<counting_aligner>( space, modes, policy );
    auto rebound =
        std::make_shared<prepared_quotient>( space, std::make_shared<identity_type>( space ), actions, replacement );
    storage.set_symmetry_adapter( rebound );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require( replacement->preparations == actions->size(), "Adapter rebinding retained stale preparation" );
    storage.clear();
    storage.set_known_solution( *root );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require(
        std::abs( storage.last_minimum_distance() - expected ) < 1e-22, "Base-root preparation changed the metric"
    );
    put( *root, 3, .1 );
    storage.set_known_solution( *root );
    storage.calc_distance( *query, multiplier, *gradient );
    report.require( storage.last_minimum_distance() > 1e-3, "Base-root replacement reused stale compatibility" );
    storage.clear();
    storage.set_ignore_zero();
    storage.calc_distance( *query, multiplier, *gradient );
    report.require( multiplier == 1. && space->norm( *gradient ) == 0., "Cleared root preparation remained active" );
    auto limited_policy                        = policy;
    limited_policy.maximum_proposal_characters = 4;
    auto limited_aligner                       = std::make_shared<counting_aligner>( space, modes, limited_policy );
    auto limited_adapter                       = std::make_shared<prepared_quotient>(
        space, std::make_shared<identity_type>( space ), actions, limited_aligner
    );
    prepared_storage limited_storage( space, 2, double( size ), 2., limited_adapter, 1e-6 );
    put( *root, 3, 1.2e-8 );
    bool rejected = false;
    try
    {
        limited_storage.push_back( *root );
    }
    catch ( const std::runtime_error & )
    {
        rejected = true;
    }
    report.require( rejected && limited_storage.get_size() == 0, "Failed preparation inserted a partial root" );
    limited_storage.set_symmetry_adapter( rebound );
    limited_storage.push_back( *root );
    report.require( limited_storage.get_size() == 1, "Failed preparation prevented subsequent root insertion" );
}

} // namespace

int main()
{
    deflation::tests::test_report report;
    try
    {
        test_tangent_similarity( report );
        test_metric( report );
        test_source_orbit_cutoff( report, false );
        test_source_orbit_cutoff( report, true );
        test_stored_source_preparation( report );
        std::printf( "Orbit deflation metric: %u checks passed\n", report.checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "Orbit deflation metric failed after %u checks: %s\n", report.checks, error.what() );
        return 1;
    }
}
