#include "section_test_support.h"
#include <periodic_orbit/sections/affine_chart.h>
using namespace section_tests;

// Manufactured affine hyperplanes, including normals almost parallel to either
// coordinate axis. Round trips and directional differences test an isometric
// N-1 coordinate representation without any dense basis or host state views.
struct pattern
{
    double             *out;
    int                 mode;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        if ( mode == 0 || mode == 1 )
            out[i] = i == 0 ? ( mode == 0 ? 1. : -1. ) : 1e-14;
        else if ( mode == 2 )
            out[i] = i == 0 ? 0. : 1.;
        else
            out[i] = .2 * std::sin( .17 * ( i + 1 ) + mode );
    }
};
void initialize( X &out, int mode )
{
    if ( out.size() )
        backend::for_each_type{}( pattern{ out.raw_ptr(), mode }, out.size() );
}

void check_chart( std::size_t count, int normal_mode )
{
    auto ops    = std::make_shared<operations>( count );
    auto coords = std::make_shared<operations>( count - 1 );
    auto center = buffer( ops ), normal = buffer( ops ), state = buffer( ops ), delta = buffer( ops ),
         saved = buffer( ops ), expected = buffer( ops );
    auto z = buffer( coords ), v = buffer( coords ), recovered = buffer( coords );
    initialize( *center, 3 );
    initialize( *normal, normal_mode );
    initialize( *z, 4 );
    initialize( *v, 5 );
    auto                                     section = std::make_shared<plane>( ops, *center, *normal );
    periodic_orbit::affine_chart<operations> chart( ops, coords, section );
    plane::evaluation_workspace              work( ops );
    const double                             tol        = 1e-11 * ( 1 + coords->norm( *z ) );
    const auto                               host_bytes = guarded_memory::host_bytes;
    chart.lift( *z, *state );
    require( std::abs( section->bind( work ).value( 0, *state ) ) < tol, "Lifted state is off section" );
    chart.restrict( *state, *recovered );
    coords->add_lin_comb( -1, *z, *recovered );
    require( coords->norm( *recovered ) < tol, "Section coordinate round trip" );
    chart.lift_increment( *v, *delta );
    require( std::abs( ops->scalar_prod( section->normal(), *delta ) ) < tol, "Coordinate increment is not tangent" );
    require(
        std::abs( ops->norm_sq( *delta ) - coords->norm_sq( *v ) ) < tol * ( 1 + coords->norm( *v ) ),
        "Chart is not isometric"
    );
    chart.restrict_increment( *delta, *recovered );
    coords->add_lin_comb( -1, *v, *recovered );
    require( coords->norm( *recovered ) < tol, "Tangent coordinate round trip" );

    initialize( *state, 6 );
    ops->assign_lin_comb( 1, *state, -1, *center, *expected );
    ops->add_lin_comb( -ops->scalar_prod( section->normal(), *expected ), section->normal(), *expected );
    ops->add_lin_comb( 1, *center, *expected );
    chart.restrict( *state, *recovered );
    chart.lift( *recovered, *saved );
    ops->add_lin_comb( -1, *expected, *saved );
    require( ops->norm( *saved ) < tol, "Off-section restriction is not orthogonal projection" );

    coords->assign_lin_comb( 1, *z, 1e-5, *v, *recovered );
    chart.lift( *recovered, *state );
    coords->assign_lin_comb( 1, *z, -1e-5, *v, *recovered );
    chart.lift( *recovered, *saved );
    ops->assign_lin_comb( 5e4, *state, -5e4, *saved, *expected );
    ops->add_lin_comb( -1, *delta, *expected );
    require( ops->norm( *expected ) < 1e-8 * ( 1 + ops->norm( *delta ) ), "Chart derivative finite difference" );

    if ( count > 1 )
    {
        // Overlapping views are safe because the chart stages data on its backend.
        X alias;
        alias.init_by_raw_data( ( *state ).raw_ptr() + 1, count - 1 );
        coords->assign( *z, alias );
        chart.lift( alias, *state );
        chart.restrict( *state, alias );
        coords->assign_lin_comb( 1, alias, -1, *z, *recovered );
        require( coords->norm( *recovered ) < tol, "Overlapping chart buffers were corrupted" );
    }
    require( guarded_memory::host_bytes == host_bytes, "Chart conversion transferred array data to host" );
    auto       wrong  = buffer( ops );
    const auto before = ops->norm( *state );
    require_throws<std::invalid_argument>(
        [&] { chart.lift( *wrong, *state ); }, "Wrong coordinate dimension accepted"
    );
    require( ops->norm( *state ) == before, "Rejected chart call changed output" );
    require_throws<std::invalid_argument>(
        [&] { periodic_orbit::affine_chart<operations> bad( ops, ops, section ); },
        "Wrong chart-space dimension accepted"
    );
    section.reset();
    chart.lift( *z, *state );
    require( ops->is_valid_number( *state ), "Chart lost its section owner" );
}

struct allocation_space : operations
{
    using operations::operations;
    mutable std::size_t live = 0, allocations = 0;
    std::size_t         fail_allocation = 0;
    void                start_use_vector( X &x ) const override
    {
        const bool fresh = x.is_free();
        operations::start_use_vector( x );
        if ( fresh )
        {
            ++live;
            ++allocations;
            if ( allocations == fail_allocation )
                throw std::runtime_error( "Injected chart activation failure" );
        }
    }
    void free_vector( X &x ) const override
    {
        if ( !x.is_free() )
            --live;
        operations::free_vector( x );
    }
};

void check_construction_failure()
{
    auto ops         = std::make_shared<allocation_space>( 8 );
    auto coordinates = std::make_shared<allocation_space>( 7 );
    using storage    = nmfd::detail::vector_wrap<allocation_space, false, false>;
    storage center( *ops ), normal( *ops );
    center.init();
    center.start_use();
    normal.init();
    normal.start_use();
    ops->assign_scalar( 0, *center );
    ops->assign_scalar( 1, *normal );
    auto       section   = std::make_shared<periodic_orbit::affine_section<allocation_space>>( ops, *center, *normal );
    const auto before    = ops->live;
    ops->fail_allocation = ops->allocations + 2;
    require_throws<std::runtime_error>(
        [&] { periodic_orbit::affine_chart<allocation_space> bad( ops, coordinates, section ); },
        "Chart activation failure ignored"
    );
    require( ops->live == before, "Partially constructed chart leaked a backend buffer" );
    ops->fail_allocation = 0;
    periodic_orbit::affine_chart<allocation_space> good( ops, coordinates, section );
    require( ops->live == before + 2, "Chart storage is not bounded to two state buffers" );
}

void test()
{
    for ( auto count : { 2u, 3u, 32u, 32768u } )
        for ( int mode = 0; mode < 3; ++mode )
            check_chart( count, mode );
    check_chart( 1, 0 );
    check_chart( 1, 1 );
    check_construction_failure();
    require( guarded_memory::host_bytes == 0, "Section charts performed host transfers" );
}
int main()
{
    return run( test );
}
