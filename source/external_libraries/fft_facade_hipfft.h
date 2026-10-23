#ifndef __EXTERNAL_LIBRARIES_FFT_FACADE_HIPFFT_H__
#define __EXTERNAL_LIBRARIES_FFT_FACADE_HIPFFT_H__

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <hipfft/hipfft.h>
#include <thrust/complex.h>
#include <external_libraries/fft_facade.h>

namespace external_libraries
{
namespace fft
{
namespace detail
{

inline void check_hipfft( hipfftResult result, const char *operation )
{
    if ( result != HIPFFT_SUCCESS )
        throw std::runtime_error( std::string( operation ) + ": hipFFT status " + std::to_string( result ) );
}

class hipfft_plan
{
public:
    hipfft_plan( const dimensions &dims, hipfftType type )
    {
        validate_dimensions( dims );
        const auto maximum = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        if ( dims.size_x > maximum || dims.size_y > maximum || dims.size_z > maximum )
            throw std::invalid_argument( "hipFFT dimensions exceed the integer interface" );
        check_hipfft( hipfftCreate( &handle_ ), "Create FFT handle" );
        try
        {
            std::size_t  workspace;
            hipfftResult result;
            if ( dims.rank == 1 )
                result = hipfftMakePlan1d( handle_, static_cast<int>( dims.size_x ), type, 1, &workspace );
            else if ( dims.rank == 2 )
                result = hipfftMakePlan2d(
                    handle_, static_cast<int>( dims.size_x ), static_cast<int>( dims.size_y ), type, &workspace
                );
            else
                result = hipfftMakePlan3d(
                    handle_, static_cast<int>( dims.size_x ), static_cast<int>( dims.size_y ),
                    static_cast<int>( dims.size_z ), type, &workspace
                );
            check_hipfft( result, "Prepare FFT plan" );
        }
        catch ( ... )
        {
            hipfftDestroy( handle_ );
            throw;
        }
    }

    hipfft_plan( const hipfft_plan & )            = delete;
    hipfft_plan &operator=( const hipfft_plan & ) = delete;

    ~hipfft_plan()
    {
        hipfftDestroy( handle_ );
    }

    hipfftHandle get() const
    {
        return handle_;
    }

private:
    hipfftHandle handle_{};
};

}

// Unnormalized, out-of-place R2C/C2R transforms, matching FFTW and cuFFT.
template <class T>
class r2c<hipfft_backend, T>
{
    static_assert( std::is_same_v<T, float> || std::is_same_v<T, double>, "hipFFT supports float and double" );

public:
    using backend_type = hipfft_backend;
    using real_type    = T;
    using complex_type = thrust::complex<T>;

    explicit r2c( std::size_t size_x ) : r2c( dimensions( size_x ) )
    {
    }

    r2c( std::size_t size_x, std::size_t size_y ) : r2c( dimensions( size_x, size_y ) )
    {
    }

    r2c( std::size_t size_x, std::size_t size_y, std::size_t size_z ) : r2c( dimensions( size_x, size_y, size_z ) )
    {
    }

    explicit r2c( const dimensions &dims )
        : dims_( dims ), forward_( dims, std::is_same_v<T, float> ? HIPFFT_R2C : HIPFFT_D2Z ),
          inverse_( dims, std::is_same_v<T, float> ? HIPFFT_C2R : HIPFFT_Z2D )
    {
        static_assert( sizeof( complex_type ) == 2 * sizeof( T ), "hipFFT requires interleaved complex storage" );
    }

    void forward( T *source, complex_type *destination )
    {
        if constexpr ( std::is_same_v<T, float> )
            detail::check_hipfft(
                hipfftExecR2C( forward_.get(), source, reinterpret_cast<hipfftComplex *>( destination ) ), "R2C FFT"
            );
        else
            detail::check_hipfft(
                hipfftExecD2Z( forward_.get(), source, reinterpret_cast<hipfftDoubleComplex *>( destination ) ),
                "D2Z FFT"
            );
    }

    void inverse( complex_type *source, T *destination )
    {
        if constexpr ( std::is_same_v<T, float> )
            detail::check_hipfft(
                hipfftExecC2R( inverse_.get(), reinterpret_cast<hipfftComplex *>( source ), destination ), "C2R FFT"
            );
        else
            detail::check_hipfft(
                hipfftExecZ2D( inverse_.get(), reinterpret_cast<hipfftDoubleComplex *>( source ), destination ),
                "Z2D FFT"
            );
    }

    void fft( T *source, complex_type *destination )
    {
        forward( source, destination );
    }

    void ifft( complex_type *source, T *destination )
    {
        inverse( source, destination );
    }

    std::size_t get_reduced_size() const
    {
        return dims_.reduced_size();
    }

    std::size_t reduced_size() const
    {
        return dims_.reduced_size();
    }

    std::size_t physical_size() const
    {
        return dims_.physical_size();
    }

    std::size_t complex_size() const
    {
        return dims_.r2c_complex_size();
    }

    T normalization_factor() const
    {
        return static_cast<T>( physical_size() );
    }

    const dimensions &dims() const
    {
        return dims_;
    }

    static const char *backend_name()
    {
        return "hipfft";
    }

private:
    dimensions          dims_;
    detail::hipfft_plan forward_, inverse_;
};

}
}

#endif
