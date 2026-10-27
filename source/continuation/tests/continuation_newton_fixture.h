#ifndef __CONTINUATION_TESTS_NEWTON_FIXTURE_H__
#define __CONTINUATION_TESTS_NEWTON_FIXTURE_H__

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <scfd/backend/hip.h>
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
#else
#    include <scfd/backend/serial_cpu.h>
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <continuation/corrector_operator.h>

namespace continuation
{
namespace tests
{
#if defined( TEST_VECTOR_BACKEND_CUDA )
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_HIP )
using backend_type = scfd::backend::hip<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif

struct test_report
{
    unsigned int checks = 0;
    void         require( bool passed, const char *message )
    {
        ++checks;
        if ( !passed )
            throw std::runtime_error( message );
    }
};

template <class T>
class vector_space : public nmfd::operations::scfd_vector_space<backend_type, T>
{
    using parent_type = nmfd::operations::scfd_vector_space<backend_type, T>;

public:
    using typename parent_type::vector_type;
    mutable unsigned int allocations = 0, frees = 0, activations = 0;
    unsigned int         fail_activation = 0;
    vector_space() : parent_type( 1 )
    {
    }
    void init_vector( vector_type &x ) const
    {
        parent_type::init_vector( x );
        ++allocations;
    }
    void free_vector( vector_type &x ) const
    {
        parent_type::free_vector( x );
        ++frees;
    }
    void start_use_vector( vector_type &x ) const
    {
        if ( ++activations == fail_activation )
            throw std::runtime_error( "Injected activation failure" );
        parent_type::start_use_vector( x );
    }
};

template <class T>
struct circle_fixture
{
    using space_type          = vector_space<T>;
    using problem_type        = nonlinear_operators::circle<space_type>;
    using state_operator_type = nonlinear_operators::linear_operator_circle<space_type, problem_type>;
    using operator_type       = corrector_operator<space_type, problem_type, state_operator_type>;
    using pair_space_type     = typename operator_type::vector_space_type;
    using vector_type         = typename pair_space_type::vector_type;
    using storage_type        = nmfd::detail::vector_wrap<pair_space_type, false, false>;
    std::shared_ptr<space_type>          operations     = std::make_shared<space_type>();
    std::shared_ptr<problem_type>        problem        = std::make_shared<problem_type>( T( 1 ), 1, operations );
    std::shared_ptr<state_operator_type> state_operator = std::make_shared<state_operator_type>( problem );
    std::shared_ptr<operator_type>       op    = std::make_shared<operator_type>( operations, problem, state_operator );
    std::shared_ptr<pair_space_type>     space = op->space();
    storage_type reference{ *space }, tangent{ *space }, state{ *space }, rhs{ *space }, direction{ *space },
        action{ *space }, plus{ *space }, minus{ *space };
    static constexpr T tolerance = std::is_same_v<T, float> ? T( 3e-6 ) : T( 3e-12 );

    circle_fixture()
    {
        for ( auto *storage : { &reference, &tangent, &state, &rhs, &direction, &action, &plus, &minus } )
        {
            storage->init();
            storage->start_use();
            space->assign_scalar( T( 0 ), **storage );
        }
    }
    void set( vector_type &x, T state_value, T parameter )
    {
        operations->assign_scalar( state_value, x.first );
        x.second[0] = parameter;
    }
    T value( const vector_type &x ) const
    {
        T result;
        operations->get( x.first, &result, 1 );
        return result;
    }
    void configure( T angle, T step, char type = 'S' )
    {
        set( *reference, std::cos( angle ), std::sin( angle ) );
        set( *tangent, -std::sin( angle ), std::cos( angle ) );
        op->set_tangent_space(
            ( *reference ).first, ( *reference ).second[0], ( *tangent ).first, ( *tangent ).second[0], step, type
        );
    }
    T difference( const vector_type &a, const vector_type &b )
    {
        space->assign_lin_comb( T( 1 ), a, T( -1 ), b, *minus );
        return space->norm_l2( *minus );
    }
};

}
}

#endif
