#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <scfd/backend/serial_cpu.h>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
#    include <common/cuda_init_scfd.h>
#else
#    include <scfd/backend/omp.h>
#endif

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/detail/static_vector_traits.h>
#include <nmfd/operations/scfd_vector_operations.h>

namespace
{

unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class Operations, class = void>
struct has_factory : std::false_type
{
};

template <class Operations>
struct has_factory<
    Operations, std::void_t<decltype( std::declval<Operations &>()
                                          .init_vector( std::declval<typename Operations::vector_type &>() ) )>>
    : std::true_type
{
};

template <class Backend, class T>
class inspected_operations : public nmfd::operations::scfd_vector_operations<Backend, T>
{
    using parent_t = nmfd::operations::scfd_vector_operations<Backend, T>;

public:
    using parent_t::parent_t;
    const void *scratch() const
    {
        return this->helper_.raw_ptr();
    }
};

template <class Backend, class T>
void test_hierarchy()
{
    using operations_type    = inspected_operations<Backend, T>;
    using traits_type        = nmfd::operations::detail::scfd_array_traits<T, typename Backend::memory_type>;
    using space_type         = nmfd::operations::dense_vector_space<traits_type, Backend>;
    using project_space_type = nmfd::operations::scfd_vector_space<Backend, T>;
    using vector_type        = typename space_type::vector_type;
    using norm_type          = typename space_type::norm_type;

    static_assert( !has_factory<operations_type>::value, "Operations must not require a space factory" );
    static_assert( has_factory<space_type>::value, "A space must provide its factory" );
    static_assert( std::is_base_of_v<typename space_type::parent_t, space_type> );
    static_assert( std::is_same_v<scfd_vector_operations<Backend, T>, project_space_type> );
    static_assert( std::is_floating_point_v<norm_type> );
    static_assert( !std::is_copy_constructible_v<operations_type> );

    operations_type operations( 1 );
    const void     *largest_scratch = nullptr;
    for ( const std::size_t size : { 1, 31, 7, 31 } )
    {
        space_type                            space( size );
        nmfd::detail::vector_wrap<space_type> x( space ), y( space );
        x.start_use();
        y.start_use();
        operations.assign_scalar( T( 2 ), *x );
        operations.assign_scalar( T( 3 ), *y );
        require( space.size() == size && space.get_size( *x ) == size, "Space owns vector dimensions" );
        require( operations.scalar_prod( *x, *y ) == T( 6 * size ), "Operations use supplied vector extents" );
        require( operations.norm_sq( *x ) == norm_type( 4 * size ), "Real norm type and unscaled squared norm" );
        require( operations.norm_inf( *x ) == norm_type( 2 ), "Infinity norm is not an absolute sum" );

        if ( size == 31 && largest_scratch == nullptr )
            largest_scratch = operations.scratch();
        if ( largest_scratch != nullptr )
            require( operations.scratch() == largest_scratch, "Scratch capacity survives smaller vectors" );

        operations.use_high_precision();
        require(
            operations.scalar_prod( *x, *y ) == T( 6 * size ) && operations.last_reduction_used_high_precision(),
            "Compensated reduction follows vector extent"
        );
        operations.set_regular_precision();
        const norm_type tolerance = norm_type( 16 ) * std::numeric_limits<norm_type>::epsilon();
        require(
            std::abs( space.norm_l2( *x ) - space.norm( *x ) ) <= tolerance * space.norm( *x ),
            "Dense space keeps its Euclidean norm convention"
        );

        project_space_type project_space( size );
        require(
            std::abs( project_space.norm_l2( *x ) - norm_type( 2 ) ) <= tolerance,
            "Project SCFD space keeps its RMS norm convention"
        );
        require( project_space.norm2_sq( *x ) == norm_type( 4 ), "Project normalized squared norm" );
        operations.assign_scalar( T( std::numeric_limits<norm_type>::max() / norm_type( 2 ) ), *x );
        require( operations.check_is_valid_number( *x ), "Finite validation does not square large values" );
        operations.set_value_at_point( T( std::numeric_limits<norm_type>::infinity() ), 0, *x );
        require( !operations.check_is_valid_number( *x ), "Finite validation catches nonfinite values" );
    }

    space_type                            space( 2 );
    nmfd::detail::vector_wrap<space_type> x( space ), y( space );
    x.start_use();
    y.start_use();
    space.assign_scalar( T( 1 ), *x );
    typename space_type::operations_base_type &algebra = space;
    typename space_type::factory_base_type    &factory = space;
    algebra.assign_lin_comb( T( 4 ), *x, *y );
    require( algebra.norm_sq( *y ) == norm_type( 32 ), "Common operation base dispatches correctly" );
    algebra.set_high_precision();
    algebra.scalar_prod( *x, *y );
    require( space.last_reduction_used_high_precision(), "Base precision control reaches the implementation" );
    algebra.set_regular_precision();
    algebra.scalar_prod( *x, *y );
    require( !space.last_reduction_used_high_precision(), "Base precision control restores ordinary reduction" );
    vector_type extra;
    factory.init_vector( extra );
    factory.start_use_vector( extra );
    require( space.get_size( extra ) == 2, "Common factory base allocates in the same space" );
    factory.stop_use_vector( extra );
    factory.free_vector( extra );

    space_type                            other_space( 3 );
    nmfd::detail::vector_wrap<space_type> other( other_space );
    other.start_use();
    bool rejected = false;
    try
    {
        operations.scalar_prod( *x, *other );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Scalar product rejects mismatched vector extents" );

    typename space_type::multivector_type basis;
    rejected = false;
    try
    {
        space.init_multivector( basis, -1 );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected && basis.empty(), "Negative multivector count leaves storage intact" );
    space.init_multivector( basis, 1 );
    rejected = false;
    try
    {
        space.start_use_multivector( basis, 2 );
    }
    catch ( const std::out_of_range & )
    {
        rejected = true;
    }
    require( rejected && basis.size() == 1, "Activation rejects an invalid multivector count" );
    rejected = false;
    try
    {
        operations.assign( basis, 1, 1, *x );
    }
    catch ( const std::out_of_range & )
    {
        rejected = true;
    }
    require( rejected, "Multivector algebra rejects an invalid basis index" );
    space.free_multivector( basis, 1 );
}

template <class T>
void test_static_traits()
{
    using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
    using traits_type  = nmfd::operations::detail::static_vector_traits<T, 3>;
    using space_type   = nmfd::operations::dense_vector_space<traits_type, backend_type>;
    space_type                       space;
    typename space_type::vector_type x{}, y{};
    space.assign_scalar( T( 2 ), x );
    space.assign_scalar( T( 3 ), y );
    require( space.size() == 3 && space.scalar_prod( x, y ) == T( 18 ), "Static storage uses the same operations" );
    space.set_value_at_point( T( 5 ), 1, x );
    require( space.get_value_at_point( 1, x ) == T( 5 ), "Static storage uses trait-based component access" );
    space.make_abs_copy( x, y );
    require( space.norm_inf( y ) == T( 5 ), "Static real-valued mappings use rebound storage traits" );
}

}

int main()
{
    try
    {
#ifdef TEST_VECTOR_BACKEND_CUDA
        common::init_cuda_from_scfd_selector( "auto" );
        using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#else
        using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#endif
        test_hierarchy<scfd::backend::serial_cpu<std::ptrdiff_t>, float>();
        test_hierarchy<scfd::backend::serial_cpu<std::ptrdiff_t>, double>();
        test_hierarchy<backend_type, float>();
        test_hierarchy<backend_type, double>();
        using complex_type = common::scfd_backend_ext::complex_t<backend_type, double>;
        test_hierarchy<backend_type, complex_type>();
        test_static_traits<float>();
        test_static_traits<double>();
        std::cout << "Checks: " << checks << ", failures: 0\nPASSED\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
