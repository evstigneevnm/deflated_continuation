#ifndef __DEFLATION_TESTS_NEWTON_FIXTURE_H__
#define __DEFLATION_TESTS_NEWTON_FIXTURE_H__

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#if defined( TEST_VECTOR_BACKEND_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
#else
#    include <scfd/backend/serial_cpu.h>
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <deflation/deflated_operator.h>
#include <deflation/solution_storage.h>
#include <deflation/symmetry_solution_storage.h>

namespace deflation
{
namespace tests
{
#if defined( TEST_VECTOR_BACKEND_CUDA )
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
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
    explicit vector_space( std::size_t size = 1 ) : parent_type( size )
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
    using solutions_type      = solution_storage<space_type>;
    using operator_type       = deflated_operator<space_type, problem_type, state_operator_type, solutions_type>;
    using vector_type         = typename space_type::vector_type;
    using storage_type        = nmfd::detail::vector_wrap<space_type, false, false>;
    std::shared_ptr<space_type>          space          = std::make_shared<space_type>();
    std::shared_ptr<problem_type>        problem        = std::make_shared<problem_type>( T( 1 ), 1, space );
    std::shared_ptr<state_operator_type> state_operator = std::make_shared<state_operator_type>( problem );
    std::shared_ptr<solutions_type>      solutions      = std::make_shared<solutions_type>( space.get(), 4, T( 1 ) );
    static constexpr T                   tolerance      = std::is_same_v<T, float> ? T( 3e-6 ) : T( 3e-12 );
    std::shared_ptr<operator_type>       op =
        std::make_shared<operator_type>( space, problem, state_operator, solutions, T( 0 ), T( 10 ) * tolerance );
    storage_type state{ *space }, residual{ *space }, direction{ *space }, action{ *space }, plus{ *space },
        minus{ *space }, initial{ *space };

    circle_fixture()
    {
        solutions->set_ignore_zero();
        for ( auto *storage : { &state, &residual, &direction, &action, &plus, &minus, &initial } )
        {
            storage->init();
            storage->start_use();
            space->assign_scalar( T( 0 ), **storage );
        }
    }
    T value( const vector_type &state ) const
    {
        T result;
        space->get( state, &result, 1 );
        return result;
    }
};

}
}

#endif
