#ifndef __LINSOLVER_EXTERNAL_HIPFFT_WRAPPER_H__
#define __LINSOLVER_EXTERNAL_HIPFFT_WRAPPER_H__

#include <cstddef>
#include <stdexcept>
#include <string>

#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#elif __has_include(<hipfft.h>)
#include <hipfft.h>
#else
#error "hipfft_wrapper.h requires hipFFT headers. Install hipFFT or include this header only in HIP FFT builds."
#endif

namespace linsolver_external_libraries
{
namespace detail
{

inline std::string hipfft_error_name(hipfftResult status)
{
    if(status == HIPFFT_SUCCESS)
    {
        return "HIPFFT_SUCCESS";
    }
    return "HIPFFT_ERROR_" + std::to_string(static_cast<int>(status));
}

inline void hipfft_safe_call(hipfftResult status, const char* expression, const char* file, int line)
{
    if(status != HIPFFT_SUCCESS)
    {
        throw std::runtime_error(
            std::string("HIPFFT_SAFE_CALL ") + file + " " + std::to_string(line) + " : " + expression +
            " failed: " + hipfft_error_name(status)
        );
    }
}

#define LINSOLVER_HIPFFT_SAFE_CALL(expr) \
    ::linsolver_external_libraries::detail::hipfft_safe_call((expr), #expr, __FILE__, __LINE__)

template<class T>
struct hipfft_traits;

template<>
struct hipfft_traits<float>
{
    using real_type = float;
    using complex_type = hipfftComplex;

    static constexpr hipfftType c2c_type = HIPFFT_C2C;
    static constexpr hipfftType r2c_type = HIPFFT_R2C;
    static constexpr hipfftType c2r_type = HIPFFT_C2R;

    static hipfftResult exec_c2c(hipfftHandle plan, complex_type* source, complex_type* destination, int direction)
    {
        return hipfftExecC2C(plan, source, destination, direction);
    }

    static hipfftResult exec_r2c(hipfftHandle plan, real_type* source, complex_type* destination)
    {
        return hipfftExecR2C(plan, source, destination);
    }

    static hipfftResult exec_c2r(hipfftHandle plan, complex_type* source, real_type* destination)
    {
        return hipfftExecC2R(plan, source, destination);
    }
};

template<>
struct hipfft_traits<double>
{
    using real_type = double;
    using complex_type = hipfftDoubleComplex;

    static constexpr hipfftType c2c_type = HIPFFT_Z2Z;
    static constexpr hipfftType r2c_type = HIPFFT_D2Z;
    static constexpr hipfftType c2r_type = HIPFFT_Z2D;

    static hipfftResult exec_c2c(hipfftHandle plan, complex_type* source, complex_type* destination, int direction)
    {
        return hipfftExecZ2Z(plan, source, destination, direction);
    }

    static hipfftResult exec_r2c(hipfftHandle plan, real_type* source, complex_type* destination)
    {
        return hipfftExecD2Z(plan, source, destination);
    }

    static hipfftResult exec_c2r(hipfftHandle plan, complex_type* source, real_type* destination)
    {
        return hipfftExecZ2D(plan, source, destination);
    }
};

inline void create_hipfft_plan(hipfftHandle* plan, int rank, std::size_t size_x, std::size_t size_y, std::size_t size_z, hipfftType type)
{
    if(rank == 1)
    {
        LINSOLVER_HIPFFT_SAFE_CALL(hipfftPlan1d(plan, static_cast<int>(size_x), type, 1));
    }
    else if(rank == 2)
    {
        LINSOLVER_HIPFFT_SAFE_CALL(hipfftPlan2d(plan, static_cast<int>(size_x), static_cast<int>(size_y), type));
    }
    else
    {
        LINSOLVER_HIPFFT_SAFE_CALL(
            hipfftPlan3d(plan, static_cast<int>(size_x), static_cast<int>(size_y), static_cast<int>(size_z), type)
        );
    }
}

} // namespace detail

template<class T>
class hipfft_wrap_R2C
{
public:
    using real_type = T;
    using complex_type = typename detail::hipfft_traits<T>::complex_type;

    explicit hipfft_wrap_R2C(std::size_t size_x):
        rank_(1),
        size_x_(size_x),
        size_y_(1),
        size_z_(1),
        reduced_size_(size_x / 2 + 1)
    {
        create_plans();
    }

    hipfft_wrap_R2C(std::size_t size_x, std::size_t size_y):
        rank_(2),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(1),
        reduced_size_(size_y / 2 + 1)
    {
        create_plans();
    }

    hipfft_wrap_R2C(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        rank_(3),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(size_z),
        reduced_size_(size_z / 2 + 1)
    {
        create_plans();
    }

    hipfft_wrap_R2C(const hipfft_wrap_R2C&) = delete;
    hipfft_wrap_R2C& operator=(const hipfft_wrap_R2C&) = delete;

    hipfft_wrap_R2C(hipfft_wrap_R2C&& other) noexcept
    {
        move_from(other);
    }

    hipfft_wrap_R2C& operator=(hipfft_wrap_R2C&& other) noexcept
    {
        if(this != &other)
        {
            destroy_plans();
            move_from(other);
        }
        return *this;
    }

    ~hipfft_wrap_R2C()
    {
        destroy_plans();
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
        LINSOLVER_HIPFFT_SAFE_CALL(detail::hipfft_traits<T>::exec_r2c(planR2C_, source, destination));
    }

    void inverse(complex_type* source, real_type* destination)
    {
        LINSOLVER_HIPFFT_SAFE_CALL(detail::hipfft_traits<T>::exec_c2r(planC2R_, source, destination));
    }

    std::size_t get_reduced_size() const
    {
        return reduced_size_;
    }

    std::size_t reduced_size() const
    {
        return reduced_size_;
    }

    std::size_t physical_size() const
    {
        return size_x_ * size_y_ * size_z_;
    }

    std::size_t complex_size() const
    {
        if(rank_ == 1)
        {
            return reduced_size_;
        }
        if(rank_ == 2)
        {
            return size_x_ * reduced_size_;
        }
        return size_x_ * size_y_ * reduced_size_;
    }

    real_type normalization_factor() const
    {
        return static_cast<real_type>(physical_size());
    }

private:
    void create_plans()
    {
        detail::create_hipfft_plan(
            &planR2C_,
            rank_,
            size_x_,
            size_y_,
            size_z_,
            detail::hipfft_traits<T>::r2c_type
        );
        detail::create_hipfft_plan(
            &planC2R_,
            rank_,
            size_x_,
            size_y_,
            size_z_,
            detail::hipfft_traits<T>::c2r_type
        );
    }

    void destroy_plans()
    {
        if(planR2C_)
        {
            hipfftDestroy(planR2C_);
            planR2C_ = 0;
        }
        if(planC2R_)
        {
            hipfftDestroy(planC2R_);
            planC2R_ = 0;
        }
    }

    void move_from(hipfft_wrap_R2C& other)
    {
        rank_ = other.rank_;
        size_x_ = other.size_x_;
        size_y_ = other.size_y_;
        size_z_ = other.size_z_;
        reduced_size_ = other.reduced_size_;
        planR2C_ = other.planR2C_;
        planC2R_ = other.planC2R_;
        other.planR2C_ = 0;
        other.planC2R_ = 0;
    }

    int rank_ = 1;
    std::size_t size_x_ = 0;
    std::size_t size_y_ = 1;
    std::size_t size_z_ = 1;
    std::size_t reduced_size_ = 0;
    hipfftHandle planR2C_ = 0;
    hipfftHandle planC2R_ = 0;
};

template<class T>
class hipfft_wrap_C2C
{
public:
    using real_type = T;
    using complex_type = typename detail::hipfft_traits<T>::complex_type;

    explicit hipfft_wrap_C2C(std::size_t size_x):
        rank_(1),
        size_x_(size_x),
        size_y_(1),
        size_z_(1)
    {
        create_plan();
    }

    hipfft_wrap_C2C(std::size_t size_x, std::size_t size_y):
        rank_(2),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(1)
    {
        create_plan();
    }

    hipfft_wrap_C2C(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        rank_(3),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(size_z)
    {
        create_plan();
    }

    hipfft_wrap_C2C(const hipfft_wrap_C2C&) = delete;
    hipfft_wrap_C2C& operator=(const hipfft_wrap_C2C&) = delete;

    ~hipfft_wrap_C2C()
    {
        if(planC2C_)
        {
            hipfftDestroy(planC2C_);
            planC2C_ = 0;
        }
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
        LINSOLVER_HIPFFT_SAFE_CALL(detail::hipfft_traits<T>::exec_c2c(planC2C_, source, destination, HIPFFT_FORWARD));
    }

    void inverse(complex_type* source, complex_type* destination)
    {
        LINSOLVER_HIPFFT_SAFE_CALL(detail::hipfft_traits<T>::exec_c2c(planC2C_, source, destination, HIPFFT_BACKWARD));
    }

    std::size_t physical_size() const
    {
        return size_x_ * size_y_ * size_z_;
    }

    std::size_t complex_size() const
    {
        return physical_size();
    }

    real_type normalization_factor() const
    {
        return static_cast<real_type>(physical_size());
    }

private:
    void create_plan()
    {
        detail::create_hipfft_plan(
            &planC2C_,
            rank_,
            size_x_,
            size_y_,
            size_z_,
            detail::hipfft_traits<T>::c2c_type
        );
    }

    int rank_ = 1;
    std::size_t size_x_ = 0;
    std::size_t size_y_ = 1;
    std::size_t size_z_ = 1;
    hipfftHandle planC2C_ = 0;
};

} // namespace linsolver_external_libraries

template<class T>
using hipfft_wrap_R2C = linsolver_external_libraries::hipfft_wrap_R2C<T>;

template<class T>
using hipfft_wrap_C2C = linsolver_external_libraries::hipfft_wrap_C2C<T>;

#endif
