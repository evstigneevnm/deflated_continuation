#include <cmath>
#include <iostream>

#include <nmfd/operations/scfd_vector_space.h>
#include <scfd/backend/serial_cpu.h>

using vec_ops_t = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<std::ptrdiff_t>, double>;
using T         = typename vec_ops_t::scalar_type;
using vec_t     = typename vec_ops_t::vector_type;
using mvec_t    = typename vec_ops_t::multivector_type;


int main( int argc, char const *argv[] )
{
    mvec_t    V;
    bool      ok_flag = true;
    vec_ops_t vec_ops( 100 );
    vec_ops.init_multivector( V, 10 );
    vec_ops.start_use_multivector( V, 10 );

    for ( int j = 0; j < 10; j++ )
    {
        auto &x = V[j];
        for ( int k = 0; k < 100; k++ )
        {

            x( k ) = ( j + 1 ) * k;
        }
    }


    for ( int j = 0; j < 10; j++ )
    {
        auto &x = V[j];
        for ( int k = 0; k < 100; k++ )
        {

            if ( std::abs( x( k ) - ( j + 1 ) * k ) > 1.0e-10 )
            {
                std::cout << "failed at: " << j << ", " << k << ". Expexted: " << ( j + 1 ) * k << ", got: " << x( k )
                          << std::endl;
                ok_flag = false;
            }
        }
    }

    vec_ops.stop_use_multivector( V, 10 );
    vec_ops.free_multivector( V, 10 );

    if ( ok_flag )
    {
        std::cout << "PASSED" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "FAILED" << std::endl;
    }
}
