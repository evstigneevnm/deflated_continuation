#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/scfd_vector_space.h>
#if defined( TEST_ISOTROPY_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_ISOTROPY_HIP )
#    include <scfd/backend/hip.h>
#elif defined( TEST_ISOTROPY_OMP )
#    include <scfd/backend/omp.h>
#else
#    include <scfd/backend/serial_cpu.h>
#endif
#include <symmetry/continuation/isotropy_transition_detector.h>
#include <symmetry/fourier/translation_isotropy_geometry.h>
#include <symmetry/finite_isotropy_geometry.h>

namespace
{

#if defined( TEST_ISOTROPY_CUDA )
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_ISOTROPY_HIP )
using backend_type = scfd::backend::hip<std::ptrdiff_t>;
#elif defined( TEST_ISOTROPY_OMP )
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif

std::size_t reduced_records = 0, returned_summary_bytes = 0;
struct checked_reduce
{
    template <class T, class Operation>
    T operator()( std::ptrdiff_t count, const T *input, T identity, Operation operation ) const
    {
        static_assert( sizeof( T ) <= 256, "A classification result must have bounded size" );
        if ( count > 256 )
            throw std::runtime_error( "Classification downloaded or reduced an unbounded record collection" );
        reduced_records        = std::max( reduced_records, static_cast<std::size_t>( count ) );
        returned_summary_bytes = std::max( returned_summary_bytes, sizeof( T ) );
        return typename backend_type::reduce_type{}( count, input, identity, operation );
    }
};
struct classification_backend : backend_type
{
    using reduce_type = checked_reduce;
};

struct space_type : nmfd::operations::scfd_vector_space<backend_type, double>
{
    using base_type = nmfd::operations::scfd_vector_space<backend_type, double>;
    using base_type::base_type;
    using backend_type                                           = classification_backend;
    void get( const vector_type &, double *, std::size_t ) const = delete;
    void set( const double *, vector_type &, std::size_t ) const = delete;
    struct memory_type : base_type::memory_type
    {
        static void copy_to_host( std::size_t, const void *, void * ) = delete;
    };
};

int  checks = 0;
void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

void test_bounded_reductions_and_packed_fields()
{
    using geometry_type                 = symmetry::fourier::translation_isotropy_geometry<space_type, 2>;
    using coefficient_type              = geometry_type::coefficient_type;
    constexpr std::size_t         count = 8192;
    space_type                    space( count );
    std::vector<coefficient_type> coefficients;
    for ( std::size_t i = 0; i < count; i += 2 )
        coefficients.push_back( { symmetry::fourier::mode_index<2>{ 1, 2 }, i, i + 1 } );
    coefficients.back().mode  = symmetry::fourier::mode_index<2>{ 2, 1 };
    coefficients.front().mode = symmetry::fourier::mode_index<2>{ 1, 0 };
    geometry_type                         geometry( space, coefficients, "two_field_translation_torus" );
    nmfd::detail::vector_wrap<space_type> before( space ), after( space );
    before.start_use();
    after.start_use();
    std::vector<double> values( count, 1. );
    values[0] = values[1] = 1e-3;
    space.base_type::set( values.data(), *before, count );
    values[0] = values[1] = 1e-14;
    space.base_type::set( values.data(), *after, count );
    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled = true;
    space.use_high_precision();
    const auto result = symmetry::continuation::detect_isotropy_transition( geometry, *before, *after, policy );
    require( result.detected && result.candidate_order == 3, "Packed multi-field diagonal C3 transition" );
    require(
        space.last_reduction_used_high_precision(), "Isotropy transverse norms ignored selected reduction precision"
    );
    require( reduced_records == 256 && returned_summary_bytes <= 256, "Per-state metadata grew with N" );
    require(
        result.candidate_stabilizer.lattice_basis == std::vector<std::int64_t>( { 1, 2, 0, 3 } ),
        "Diagonal C3 embedding was lost"
    );
}

void test_invalid_layouts()
{
    using geometry_type    = symmetry::fourier::translation_isotropy_geometry<space_type, 1>;
    using coefficient_type = geometry_type::coefficient_type;
    space_type                             space( 2 );
    const auto                             missing = std::numeric_limits<std::size_t>::max();
    const symmetry::fourier::mode_index<1> mode{ 1 };
    for ( const auto &layout : std::vector<std::vector<coefficient_type>>{
              { { mode, 0, 0 } },
              { { mode, 0, missing } },
              { { mode, 0, 2 } },
              { { mode, missing, 0 }, { mode, 1, missing } }
          } )
    {
        bool rejected = false;
        try
        {
            geometry_type geometry( space, layout, "layout_validation" );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Invalid isotropy coefficient layout was accepted" );
    }
}

void test_finite_groups()
{
    auto space   = std::make_shared<space_type>( 2 );
    auto actions = std::make_shared<symmetry::finite_action_registry<space_type>>( space );
    actions->clear();
    for ( unsigned g = 0; g < 4; ++g )
        actions->add(
            "sign_" + std::to_string( g ),
            [g]( const space_type::vector_type &source, space_type::vector_type &destination ) {
                typename space_type::for_each_type each;
                each(
                    [=] __DEVICE_TAG__( std::ptrdiff_t i ) {
                        destination( i ) = ( g & ( 1u << i ) ) ? -source( i ) : source( i );
                    },
                    2
                );
                each.wait();
            }
        );
    std::vector<std::size_t> products;
    for ( std::size_t a = 0; a < 4; ++a )
        for ( std::size_t b = 0; b < 4; ++b )
            products.push_back( a ^ b );
    symmetry::finite_isotropy_geometry<space_type> geometry( space, actions, products, "independent_sign_changes" );
    nmfd::detail::vector_wrap<space_type>          before( *space ), after( *space );
    before.start_use();
    after.start_use();
    double values[] = { 1., 1e-3 };
    space->base_type::set( values, *before, 2 );
    values[1] = 0.;
    space->base_type::set( values, *after, 2 );
    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled    = true;
    const auto result = symmetry::continuation::detect_isotropy_transition( geometry, *before, *after, policy );
    require( result.detected && result.candidate_order == 2, "Finite action transition" );
    require(
        result.candidate_stabilizer.finite_elements == std::vector<std::size_t>( { 0, 2 } ), "Finite subgroup embedding"
    );
    values[0] = 0.;
    values[1] = 1.;
    space->base_type::set( values, *before, 2 );
    require(
        !symmetry::continuation::detect_isotropy_transition( geometry, *before, *after, policy ).detected,
        "Equal-order finite subgroups are not nested"
    );
    actions->reset_to_identity();
    bool rejected = false;
    try
    {
        geometry.classify( *before, 1e-10 );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    require( rejected, "Changed registry was reused without rebinding" );
}

template <std::size_t D>
void test_lattice()
{
    using lattice_type = symmetry::translation::integer_lattice<D>;
    std::mt19937                             random( 17 );
    std::vector<std::array<std::int64_t, D>> generators( 30 );
    for ( auto &row : generators )
        for ( std::size_t j = 0; j < D; ++j )
            row[j] = ( static_cast<int>( random() % 21 ) - 10 ) * static_cast<int>( j + 2 );
    for ( int iteration = 0; iteration < 50; ++iteration )
    {
        std::shuffle( generators.begin(), generators.end(), random );
        lattice_type all, even, odd;
        for ( std::size_t i = 0; i < generators.size(); ++i )
        {
            all.insert( generators[i].data() );
            ( i % 2 == 0 ? even : odd ).insert( generators[i].data() );
        }
        const auto merged = symmetry::translation::lattice_sum<D>{}( even, odd );
        require(
            !all.overflow && all.contains( merged ) && merged.contains( all ), "Lattice merge changed its integer span"
        );
        for ( std::size_t i = 0; i < D; ++i )
            for ( std::size_t j = 0; j < D; ++j )
                require( all.basis[i][j] == merged.basis[i][j], "Hermite reduction depends on grouping" );
    }
    lattice_type scaled;
    for ( std::size_t i = 0; i < D; ++i )
    {
        std::int64_t row[D]{};
        row[i] = 2;
        scaled.insert( row );
    }
    const auto orbit = symmetry::describe_orbit( scaled );
    require(
        orbit.active_rank == D && orbit.finite_isotropy_order() == ( 1u << D ), "Smith invariants of scaled identity"
    );
}

// An extra field breaks translation symmetry even if the first field is invariant.
template <std::size_t D>
void test_geometry()
{
    using mode_type     = symmetry::fourier::mode_index<D>;
    using geometry_type = symmetry::fourier::translation_isotropy_geometry<space_type, D>;
    std::vector<mode_type> modes( D + 2 );
    for ( std::size_t i = 0; i < D; ++i )
    {
        typename mode_type::storage_type values{};
        values[i] = i == 0 ? 3 : 1;
        modes[i]  = mode_type( values );
    }
    typename mode_type::storage_type breaking{};
    breaking[0]  = 1;
    modes[D]     = mode_type( breaking );
    modes[D + 1] = modes[0];
    space_type                            space( modes.size() );
    geometry_type                         geometry( space, modes, "manufactured_translation_torus" );
    nmfd::detail::vector_wrap<space_type> previous( space ), candidate( space );
    previous.start_use();
    candidate.start_use();
    std::vector<double> values( modes.size(), 1. );
    values[D] = 1e-3;
    space.base_type::set( values.data(), *previous, values.size() );
    values[D] = 1e-14;
    space.base_type::set( values.data(), *candidate, values.size() );
    symmetry::continuation::isotropy_transition_policy<double> policy;
    policy.enabled                 = true;
    policy.relative_mode_tolerance = 1e-10;
    const auto result = symmetry::continuation::detect_isotropy_transition( geometry, *previous, *candidate, policy );
    if ( !result.detected )
        std::cerr << "C3 fixture: dimension=" << D << " orders=" << result.previous_order << ','
                  << result.candidate_order << " ratios=" << result.previous_transverse_ratio << ','
                  << result.candidate_transverse_ratio << '\n';
    require(
        result.detected && result.previous_order == 1 && result.candidate_order == 3,
        "Generic C3 transition not detected"
    );
    require( result.candidate_stabilizer.dimension == D, "Transition lost spatial dimension" );
    require(
        !symmetry::continuation::detect_isotropy_transition( geometry, *candidate, *candidate, policy ).detected,
        "Unchanged stabilizer caused an event"
    );
    require(
        !symmetry::continuation::detect_isotropy_transition( geometry, *candidate, *previous, policy ).detected,
        "Symmetry loss caused an increase event"
    );
    if constexpr ( D > 1 )
    {
        std::fill( values.begin(), values.end(), 0. );
        values[0] = 1.;
        space.base_type::set( values.data(), *candidate, values.size() );
        const auto continuous =
            symmetry::continuation::detect_isotropy_transition( geometry, *previous, *candidate, policy );
        require(
            continuous.detected && continuous.candidate_orbit_type.continuous_isotropy_dimension == D - 1,
            "Rank loss did not produce the correct continuous stabilizer"
        );
    }
    space.assign_scalar( 0., *candidate );
    require(
        !symmetry::continuation::detect_isotropy_transition( geometry, *previous, *candidate, policy ).detected,
        "Zero state created an artificial finite symmetry event"
    );
    values[0] = std::numeric_limits<double>::infinity();
    space.base_type::set( values.data(), *candidate, values.size() );
    bool rejected = false;
    try
    {
        geometry.classify( *candidate, 1e-10 );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Nonfinite state accepted" );
}

void test_embedding_and_rank()
{
    using lattice_type = symmetry::translation::integer_lattice<2>;
    lattice_type       x, y;
    const std::int64_t x0[] = { 2, 0 }, x1[] = { 0, 1 }, y0[] = { 1, 0 }, y1[] = { 0, 2 };
    x.insert( x0 );
    x.insert( x1 );
    y.insert( y0 );
    y.insert( y1 );
    require(
        symmetry::describe_orbit( x ).finite_isotropy_order() == symmetry::describe_orbit( y ).finite_isotropy_order(),
        "Equal-order test fixture"
    );
    require( !x.contains( y ) && !y.contains( x ), "Different embedded C2 groups were treated as nested" );
    lattice_type       rank_one;
    const std::int64_t diagonal[] = { 2, 2 };
    rank_one.insert( diagonal );
    const auto orbit = symmetry::describe_orbit( rank_one );
    require(
        orbit.continuous_isotropy_dimension == 1 && orbit.finite_isotropy_order() == 2, "Continuous isotropy was lost"
    );
    lattice_type       overflow;
    const std::int64_t a[] = { 1, std::numeric_limits<std::int64_t>::max() }, b[] = { 2, -1 };
    overflow.insert( a );
    overflow.insert( b );
    require( overflow.overflow, "Integer overflow was not reported" );
    const std::int64_t minimum[] = { std::numeric_limits<std::int64_t>::min(), 0 };
    require( !x.contains( minimum ), "Unsupported minimum integer passed lattice membership" );
}

}

int main()
{
    try
    {
        test_lattice<1>();
        test_lattice<2>();
        test_lattice<3>();
        test_geometry<1>();
        test_geometry<2>();
        test_geometry<3>();
        test_embedding_and_rank();
        test_bounded_reductions_and_packed_fields();
        test_invalid_layouts();
        test_finite_groups();
        std::cout << checks << " isotropy checks passed on " << backend_type::name() << '\n';
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
