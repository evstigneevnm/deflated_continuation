#ifndef __LINSOLVER_EXTERNAL_FFT_FACADE_CUFFT_H__
#define __LINSOLVER_EXTERNAL_FFT_FACADE_CUFFT_H__

#include <external_libraries/cufft_wrap.h>
#include <external_libraries/fft_facade.h>

#include <memory>

namespace external_libraries
{
namespace fft
{

template<class T>
class r2c<cufft_backend, T>
{
public:
    using backend_type = cufft_backend;
    using real_type = T;
    using complex_type = typename cufft_wrap_R2C<T>::complex_type;

    explicit r2c(std::size_t size_x):
        r2c(dimensions(size_x))
    {
    }

    r2c(std::size_t size_x, std::size_t size_y):
        r2c(dimensions(size_x, size_y))
    {
    }

    r2c(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        r2c(dimensions(size_x, size_y, size_z))
    {
    }

    explicit r2c(const dimensions& dims):
        dims_(dims),
        implementation_(make_implementation(dims))
    {
        validate_dimensions(dims_);
    }

    void fft(real_type* source, complex_type* destination)
    {
        forward(source, destination);
    }

    void ifft(complex_type* source, real_type* destination)
    {
        inverse(source, destination);
    }

    void forward(real_type* source, complex_type* destination)
    {
        implementation_->fft(source, destination);
    }

    void inverse(complex_type* source, real_type* destination)
    {
        implementation_->ifft(source, destination);
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

    real_type normalization_factor() const
    {
        return static_cast<real_type>(physical_size());
    }

    const dimensions& dims() const
    {
        return dims_;
    }

    static const char* backend_name()
    {
        return "cufft";
    }

private:
    static std::unique_ptr<cufft_wrap_R2C<T>> make_implementation(const dimensions& dims)
    {
        validate_dimensions(dims);
        if(dims.rank == 1)
        {
            return std::unique_ptr<cufft_wrap_R2C<T>>(new cufft_wrap_R2C<T>(dims.size_x));
        }
        if(dims.rank == 2)
        {
            return std::unique_ptr<cufft_wrap_R2C<T>>(new cufft_wrap_R2C<T>(dims.size_x, dims.size_y));
        }
        return std::unique_ptr<cufft_wrap_R2C<T>>(new cufft_wrap_R2C<T>(dims.size_x, dims.size_y, dims.size_z));
    }

    dimensions dims_;
    std::unique_ptr<cufft_wrap_R2C<T>> implementation_;
};

template<class T>
class c2c<cufft_backend, T>
{
public:
    using backend_type = cufft_backend;
    using real_type = T;
    using complex_type = typename cufft_wrap_C2C<T>::complex_type;

    explicit c2c(std::size_t size_x):
        c2c(dimensions(size_x))
    {
    }

    c2c(std::size_t size_x, std::size_t size_y):
        c2c(dimensions(size_x, size_y))
    {
    }

    c2c(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        c2c(dimensions(size_x, size_y, size_z))
    {
    }

    explicit c2c(const dimensions& dims):
        dims_(dims),
        implementation_(make_implementation(dims))
    {
        validate_dimensions(dims_);
    }

    void fft(complex_type* source, complex_type* destination)
    {
        forward(source, destination);
    }

    void ifft(complex_type* source, complex_type* destination)
    {
        inverse(source, destination);
    }

    void forward(complex_type* source, complex_type* destination)
    {
        implementation_->fft(source, destination);
    }

    void inverse(complex_type* source, complex_type* destination)
    {
        implementation_->ifft(source, destination);
    }

    std::size_t physical_size() const
    {
        return dims_.physical_size();
    }

    std::size_t complex_size() const
    {
        return dims_.physical_size();
    }

    real_type normalization_factor() const
    {
        return static_cast<real_type>(physical_size());
    }

    const dimensions& dims() const
    {
        return dims_;
    }

    static const char* backend_name()
    {
        return "cufft";
    }

private:
    static std::unique_ptr<cufft_wrap_C2C<T>> make_implementation(const dimensions& dims)
    {
        validate_dimensions(dims);
        if(dims.rank == 1)
        {
            return std::unique_ptr<cufft_wrap_C2C<T>>(new cufft_wrap_C2C<T>(dims.size_x));
        }
        if(dims.rank == 2)
        {
            return std::unique_ptr<cufft_wrap_C2C<T>>(new cufft_wrap_C2C<T>(dims.size_x, dims.size_y));
        }
        return std::unique_ptr<cufft_wrap_C2C<T>>(new cufft_wrap_C2C<T>(dims.size_x, dims.size_y, dims.size_z));
    }

    dimensions dims_;
    std::unique_ptr<cufft_wrap_C2C<T>> implementation_;
};

} // namespace fft
} // namespace external_libraries

#endif
