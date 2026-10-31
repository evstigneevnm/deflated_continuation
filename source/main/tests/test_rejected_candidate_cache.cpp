#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <main/deflation_continuation/rejected_candidate_cache.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = std::vector<double>;

    void init_vector( vector_type & )
    {
    }
    void start_use_vector( vector_type & )
    {
        ++active_vectors;
    }
    void stop_use_vector( vector_type & )
    {
        --active_vectors;
    }
    void free_vector( vector_type &value )
    {
        value.clear();
    }
    void assign( const vector_type &source, vector_type &destination )
    {
        destination = source;
    }
    void assign_mul(
        double left_scale, const vector_type &left, double right_scale, const vector_type &right, vector_type &output
    )
    {
        output.resize( left.size() );
        for ( std::size_t index = 0; index < left.size(); ++index )
        {
            output[index] = left_scale * left[index] + right_scale * right[index];
        }
    }
    double norm_l2( const vector_type &value )
    {
        double sum = 0.0;
        for ( const double entry : value )
        {
            sum += entry * entry;
        }
        return std::sqrt( sum );
    }

    int active_vectors = 0;
};

void require( bool condition, const std::string &message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

}

int main()
{
    try
    {
        fake_vector_operations vector_operations;
        {
            main_classes::deflation_continuation_detail::rejected_candidate_cache<fake_vector_operations> cache(
                &vector_operations
            );
            cache.add( 4.0, { 1.0, 2.0 } );
            cache.add( 4.0, { 4.0, 6.0 } );
            cache.add( 5.0, { 20.0, 20.0 } );
            require( cache.size() == 3, "cache size" );

            double distance = 0.0;
            require( cache.nearest_distance( 4.0, { 2.0, 2.0 }, distance ), "matching knot" );
            require( std::abs( distance - 1.0 ) < 1.0e-14, "nearest distance" );
            require( !cache.nearest_distance( 6.0, { 2.0, 2.0 }, distance ), "different knot" );

            cache.clear();
            require( cache.size() == 0, "clear" );
            require( vector_operations.active_vectors == 1, "difference workspace retained" );
        }
        require( vector_operations.active_vectors == 0, "all vectors released" );

        // Sign-related states represent the same orbit; the stored state must be
        // the first callback argument, as in the persistent rejection registry.
        unsigned int distance_calls = 0;
        {
            main_classes::deflation_continuation_detail::rejected_candidate_cache<fake_vector_operations> cache(
                &vector_operations, [&distance_calls]( const auto &stored, const auto &candidate ) {
                    ++distance_calls;
                    require( stored == std::vector<double>{ 1., 2. }, "distance callback argument order" );
                    return std::min(
                        std::hypot( stored[0] - candidate[0], stored[1] - candidate[1] ),
                        std::hypot( stored[0] + candidate[0], stored[1] + candidate[1] )
                    );
                }
            );
            cache.add( 4., { 1., 2. } );
            double distance;
            require( cache.nearest_distance( 4., { -1., -2. }, distance ) && distance == 0., "orbit duplicate" );
            require(
                cache.nearest_distance( 4., { -1.001, -2. }, distance ) && distance > 1e-4,
                "nearby distinct orbit was rejected"
            );
            require( !cache.nearest_distance( 5., { -1., -2. }, distance ), "orbit comparison crossed knots" );
            require( distance_calls == 2, "callback used for a different knot" );
            cache.clear();
            require( !cache.nearest_distance( 4., { -1., -2. }, distance ), "clear retained rejected states" );
        }
        require( vector_operations.active_vectors == 0, "orbit cache vectors released" );

        for ( const double invalid : { -1., std::numeric_limits<double>::quiet_NaN() } )
        {
            main_classes::deflation_continuation_detail::rejected_candidate_cache<fake_vector_operations> cache(
                &vector_operations, [invalid]( const auto &, const auto & ) { return invalid; }
            );
            cache.add( 4., { 1., 2. } );
            double distance;
            bool   rejected = false;
            try
            {
                cache.nearest_distance( 4., { -1., -2. }, distance );
            }
            catch ( const std::logic_error & )
            {
                rejected = true;
            }
            require( rejected && cache.size() == 1, "invalid orbit distance silently accepted" );
        }
        require( vector_operations.active_vectors == 0, "invalid query released cache vectors" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
