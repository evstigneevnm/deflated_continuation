#ifndef __LINSOLVER_EXTERNAL_FFT_FACADE_H__
#define __LINSOLVER_EXTERNAL_FFT_FACADE_H__

#include <cstddef>
#include <stdexcept>

namespace external_libraries
{
namespace fft
{

struct fftw_backend
{
};

struct cufft_backend
{
};

struct hipfft_backend
{
};

struct dimensions
{
    int rank = 1;
    std::size_t size_x = 0;
    std::size_t size_y = 1;
    std::size_t size_z = 1;

    explicit dimensions(std::size_t size_x_):
        rank(1),
        size_x(size_x_)
    {
    }

    dimensions(std::size_t size_x_, std::size_t size_y_):
        rank(2),
        size_x(size_x_),
        size_y(size_y_)
    {
    }

    dimensions(std::size_t size_x_, std::size_t size_y_, std::size_t size_z_):
        rank(3),
        size_x(size_x_),
        size_y(size_y_),
        size_z(size_z_)
    {
    }

    std::size_t last_size() const
    {
        return rank == 1 ? size_x : (rank == 2 ? size_y : size_z);
    }

    std::size_t reduced_size() const
    {
        return last_size()/2 + 1;
    }

    std::size_t physical_size() const
    {
        return size_x*size_y*size_z;
    }

    std::size_t r2c_complex_size() const
    {
        if(rank == 1)
        {
            return reduced_size();
        }
        if(rank == 2)
        {
            return size_x*reduced_size();
        }
        return size_x*size_y*reduced_size();
    }
};

inline void validate_dimensions(const dimensions& dims)
{
    if(dims.rank < 1 || dims.rank > 3)
    {
        throw std::runtime_error("fft facade supports only ranks 1, 2, and 3.");
    }
    if(dims.size_x == 0 || dims.size_y == 0 || dims.size_z == 0)
    {
        throw std::runtime_error("fft facade dimensions must be positive.");
    }
}

template<class Backend, class T>
class r2c;

template<class Backend, class T>
class c2c;

} // namespace fft
} // namespace external_libraries

#endif
