#include "section_test_support.h"
using namespace section_tests;

struct scaled_metric : periodic_orbit::euclidean_section_metric<operations>
{
    using base = periodic_orbit::euclidean_section_metric<operations>;
    using base::base;
    double product( const X &x, const X &y ) const
    {
        return 4 * base::product( x, y );
    }
    double norm( const X &x ) const
    {
        return std::sqrt( product( x, x ) );
    }
};

void test()
{
    auto ops    = std::make_shared<operations>( 2 );
    auto center = buffer( ops ), normal = buffer( ops ), state = buffer( ops ), direction = buffer( ops );
    fill( *ops, *center, 1e12, 1e12 );
    fill( *ops, *normal, 3, 4 );
    auto                        section = std::make_shared<plane>( ops, *center, *normal );
    plane::evaluation_workspace work( ops ), other( ops );
    auto                        evaluator = section->bind( work ), independent = section->bind( other );
    fill( *ops, *state, 1e12 + 1, 1e12 + 2 );
    fill( *ops, *direction, 5, 0 );
    require( std::abs( evaluator.value( 0, *state ) - 2.2 ) < 1e-14, "Difference-first plane evaluation" );
    require(
        std::abs( evaluator.state_derivative( 0, *state, *direction ) - 3 ) < 1e-14, "Section directional derivative"
    );
    require( independent.value( 0, *center ) == 0, "Independent evaluation workspace" );
    using weighted_plane = periodic_orbit::affine_section<operations, scaled_metric>;
    weighted_plane                       weighted( ops, *center, *normal, std::make_shared<scaled_metric>( ops ) );
    weighted_plane::evaluation_workspace weighted_work( ops );
    require(
        std::abs( weighted.bind( weighted_work ).value( 0, *state ) - 4.4 ) < 1e-14, "Supplied metric was ignored"
    );
    section.reset();
    require( std::abs( evaluator.value( 0, *state ) - 2.2 ) < 1e-14, "Evaluator did not retain immutable geometry" );
    auto                        wrong = std::make_shared<operations>( 2 );
    plane::evaluation_workspace wrong_work( wrong );
    auto                        ordinary = make_plane( ops, 0, 0, 1, 0 );
    require_throws<std::invalid_argument>(
        [&] { ordinary->bind( wrong_work ); }, "Foreign section workspace accepted"
    );
    ops->assign_scalar( 0, *normal );
    require_throws<std::invalid_argument>( [&] { plane bad( ops, *center, *normal ); }, "Zero normal accepted" );
    ops->assign_scalar( std::numeric_limits<double>::infinity(), *normal );
    require_throws<std::invalid_argument>( [&] { plane bad( ops, *center, *normal ); }, "Nonfinite normal accepted" );
    periodic_orbit::event_params<double> invalid;
    invalid.subdivisions = 0;
    require_throws<std::invalid_argument>(
        [&] { periodic_orbit::event_locator<operations> bad( ops, invalid ); }, "Unbounded event scan"
    );
    auto three       = std::make_shared<operations>( 3 );
    auto three_state = buffer( three );
    three->assign_scalar( 0, *three_state );
    require_throws<std::invalid_argument>(
        [&] { plane bad( ops, *three_state, *state ); }, "Mismatched plane dimensions accepted"
    );

    // Transfer guard is exercised, not just declared. Initialization above used kernels.
    double host[2]{};
    require_throws<std::runtime_error>(
        [&] { guarded_memory::copy_to_host( sizeof( host ), ( *state ).raw_ptr(), host ); },
        "Transfer guard did not reject a full vector"
    );
}
int main()
{
    return run( test );
}
