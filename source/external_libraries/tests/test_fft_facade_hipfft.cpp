#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <common/hip_init_scfd.h>
#include <external_libraries/fft_facade_hipfft.h>
#include <scfd/arrays/array.h>
#include <scfd/memory/hip.h>

namespace
{

int checks = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

// A constant plus one Fourier mode has known DC/mode amplitudes in any rank.
// The inverse is unnormalized: inverse(forward(u)) = physical_size * u.
template <class T>
void test_transform( const external_libraries::fft::dimensions &dims )
{
    using transform_type = external_libraries::fft::r2c<external_libraries::fft::hipfft_backend, T>;
    using complex_type   = typename transform_type::complex_type;
    transform_type                                   transform( dims );
    const auto                                       size      = dims.physical_size();
    const T                                          tolerance = std::is_same_v<T, float> ? T( 2e-5 ) : T( 1e-11 );
    const T                                          pi        = std::acos( T( -1 ) );
    scfd::arrays::array<T, scfd::memory::hip_device> input, output;
    scfd::arrays::array<complex_type, scfd::memory::hip_device> spectrum;
    input.init( size );
    output.init( size );
    spectrum.init( dims.r2c_complex_size() );
    auto values = input.create_view( false );
    for ( std::size_t i = 0; i < size; ++i )
        values( i ) = T( 1 ) + T( .25 ) * std::cos( T( 2 ) * pi * T( i % dims.last_size() ) / T( dims.last_size() ) );
    values.release( true );
    require( transform.get_reduced_size() == dims.reduced_size(), "Reduced FFT extent" );
    require( transform.complex_size() == dims.r2c_complex_size(), "Complex FFT extent" );
    require( transform.normalization_factor() == T( size ), "FFT normalization" );
    transform.fft( input.raw_ptr(), spectrum.raw_ptr() );
    auto modes = spectrum.create_view( true );
    require( std::abs( modes( 0 ).real() - T( size ) ) < tolerance * T( size ), "FFT DC amplitude" );
    require( std::abs( modes( 0 ).imag() ) < tolerance * T( size ), "FFT DC imaginary part" );
    require( std::abs( modes( 1 ).real() - T( size ) / T( 8 ) ) < tolerance * T( size ), "FFT mode amplitude" );
    modes.release( false );
    transform.ifft( spectrum.raw_ptr(), output.raw_ptr() );
    auto back = output.create_view( true );
    for ( std::size_t i = 0; i < size; ++i )
    {
        const T expected =
            T( 1 ) + T( .25 ) * std::cos( T( 2 ) * pi * T( i % dims.last_size() ) / T( dims.last_size() ) );
        require( std::abs( back( i ) / T( size ) - expected ) < tolerance, "FFT roundtrip" );
    }
    back.release( false );
}

}

int main( int argc, char **argv )
{
    try
    {
        common::init_hip_from_scfd_selector( argc > 1 ? argv[1] : "auto" );
        namespace fft = external_libraries::fft;
        for ( const auto &dims :
              { fft::dimensions( 8 ), fft::dimensions( 9 ), fft::dimensions( 5, 7 ), fft::dimensions( 4, 5, 6 ) } )
        {
            test_transform<float>( dims );
            test_transform<double>( dims );
        }
        for ( const std::size_t size : { std::size_t( 0 ), std::size_t( std::numeric_limits<int>::max() ) + 1 } )
        {
            bool rejected = false;
            try
            {
                fft::r2c<fft::hipfft_backend, double> invalid( size );
            }
            catch ( const std::exception & )
            {
                rejected = true;
            }
            require( rejected, "Invalid FFT dimensions accepted" );
        }
        std::cout << "hipFFT R2C: " << checks << " checks, 0 failures\n";
    }
    catch ( const std::exception &error )
    {
        std::cerr << "hipFFT facade failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
