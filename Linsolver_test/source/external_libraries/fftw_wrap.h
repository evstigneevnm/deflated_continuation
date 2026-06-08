#ifndef __LINSOLVER_EXTERNAL_FFTW_WRAP_H__
#define __LINSOLVER_EXTERNAL_FFTW_WRAP_H__

#include <complex>
#include <cstddef>
#include <fftw3.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace external_libraries
{
namespace detail
{

inline void ensure_fftw_plan(bool valid, const char* label)
{
    if(!valid)
    {
        throw std::runtime_error(std::string("fftw_wrap: failed to create plan: ") + label);
    }
}

template<class T>
struct fftw_traits;

template<>
struct fftw_traits<double>
{
    using real_type = double;
    using complex_type = std::complex<double>;
    using fftw_complex_type = fftw_complex;
    using plan_type = fftw_plan;

    static plan_type plan_r2c_1d(int n0, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_1d(n0, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_1d(int n0, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_1d(n0, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_2d(int n0, int n1, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_2d(int n0, int n1, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_2d(n0, n1, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_3d(int n0, int n1, int n2, real_type* in, complex_type* out)
    {
        return fftw_plan_dft_r2c_3d(n0, n1, n2, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_3d(int n0, int n1, int n2, complex_type* in, real_type* out)
    {
        return fftw_plan_dft_c2r_3d(n0, n1, n2, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2c_1d(int n0, complex_type* in, complex_type* out, int sign)
    {
        return fftw_plan_dft_1d(
            n0,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static plan_type plan_c2c_2d(int n0, int n1, complex_type* in, complex_type* out, int sign)
    {
        return fftw_plan_dft_2d(
            n0,
            n1,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static plan_type plan_c2c_3d(int n0, int n1, int n2, complex_type* in, complex_type* out, int sign)
    {
        return fftw_plan_dft_3d(
            n0,
            n1,
            n2,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static void execute(plan_type plan)
    {
        fftw_execute(plan);
    }

    static void destroy(plan_type plan)
    {
        fftw_destroy_plan(plan);
    }
};

template<>
struct fftw_traits<float>
{
    using real_type = float;
    using complex_type = std::complex<float>;
    using fftw_complex_type = fftwf_complex;
    using plan_type = fftwf_plan;

    static plan_type plan_r2c_1d(int n0, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_1d(n0, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_1d(int n0, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_1d(n0, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_2d(int n0, int n1, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_2d(int n0, int n1, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_2d(n0, n1, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_r2c_3d(int n0, int n1, int n2, real_type* in, complex_type* out)
    {
        return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, reinterpret_cast<fftw_complex_type*>(out), FFTW_ESTIMATE);
    }

    static plan_type plan_c2r_3d(int n0, int n1, int n2, complex_type* in, real_type* out)
    {
        return fftwf_plan_dft_c2r_3d(n0, n1, n2, reinterpret_cast<fftw_complex_type*>(in), out, FFTW_ESTIMATE);
    }

    static plan_type plan_c2c_1d(int n0, complex_type* in, complex_type* out, int sign)
    {
        return fftwf_plan_dft_1d(
            n0,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static plan_type plan_c2c_2d(int n0, int n1, complex_type* in, complex_type* out, int sign)
    {
        return fftwf_plan_dft_2d(
            n0,
            n1,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static plan_type plan_c2c_3d(int n0, int n1, int n2, complex_type* in, complex_type* out, int sign)
    {
        return fftwf_plan_dft_3d(
            n0,
            n1,
            n2,
            reinterpret_cast<fftw_complex_type*>(in),
            reinterpret_cast<fftw_complex_type*>(out),
            sign,
            FFTW_ESTIMATE
        );
    }

    static void execute(plan_type plan)
    {
        fftwf_execute(plan);
    }

    static void destroy(plan_type plan)
    {
        fftwf_destroy_plan(plan);
    }
};

} // namespace detail

template<class T>
class fftw_wrap_R2C
{
public:
    using real_type = T;
    using complex_type = typename detail::fftw_traits<T>::complex_type;
    using plan_type = typename detail::fftw_traits<T>::plan_type;

    explicit fftw_wrap_R2C(std::size_t size_x):
        rank_(1),
        size_x_(size_x),
        size_y_(1),
        size_z_(1),
        reduced_size_(size_x / 2 + 1)
    {
    }

    fftw_wrap_R2C(std::size_t size_x, std::size_t size_y):
        rank_(2),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(1),
        reduced_size_(size_y / 2 + 1)
    {
    }

    fftw_wrap_R2C(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        rank_(3),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(size_z),
        reduced_size_(size_z / 2 + 1)
    {
    }

    fftw_wrap_R2C(const fftw_wrap_R2C&) = delete;
    fftw_wrap_R2C& operator=(const fftw_wrap_R2C&) = delete;

    fftw_wrap_R2C(fftw_wrap_R2C&& other) noexcept
    {
        move_from(other);
    }

    fftw_wrap_R2C& operator=(fftw_wrap_R2C&& other) noexcept
    {
        if(this != &other)
        {
            destroy_plans();
            move_from(other);
        }
        return *this;
    }

    ~fftw_wrap_R2C()
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
        ensure_forward_plan(source, destination);
        detail::fftw_traits<T>::execute(planR2C_);
    }

    void inverse(complex_type* source, real_type* destination)
    {
        ensure_inverse_plan(source, destination);
        detail::fftw_traits<T>::execute(planC2R_);
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
    void ensure_forward_plan(real_type* source, complex_type* destination)
    {
        if(planR2C_ && forward_source_ == source && forward_destination_ == destination)
        {
            return;
        }
        if(planR2C_)
        {
            detail::fftw_traits<T>::destroy(planR2C_);
            planR2C_ = nullptr;
        }
        if(rank_ == 1)
        {
            planR2C_ = detail::fftw_traits<T>::plan_r2c_1d(static_cast<int>(size_x_), source, destination);
        }
        else if(rank_ == 2)
        {
            planR2C_ = detail::fftw_traits<T>::plan_r2c_2d(
                static_cast<int>(size_x_),
                static_cast<int>(size_y_),
                source,
                destination
            );
        }
        else
        {
            planR2C_ = detail::fftw_traits<T>::plan_r2c_3d(
                static_cast<int>(size_x_),
                static_cast<int>(size_y_),
                static_cast<int>(size_z_),
                source,
                destination
            );
        }
        detail::ensure_fftw_plan(planR2C_ != nullptr, "R2C");
        forward_source_ = source;
        forward_destination_ = destination;
    }

    void ensure_inverse_plan(complex_type* source, real_type* destination)
    {
        if(planC2R_ && inverse_source_ == source && inverse_destination_ == destination)
        {
            return;
        }
        if(planC2R_)
        {
            detail::fftw_traits<T>::destroy(planC2R_);
            planC2R_ = nullptr;
        }
        if(rank_ == 1)
        {
            planC2R_ = detail::fftw_traits<T>::plan_c2r_1d(static_cast<int>(size_x_), source, destination);
        }
        else if(rank_ == 2)
        {
            planC2R_ = detail::fftw_traits<T>::plan_c2r_2d(
                static_cast<int>(size_x_),
                static_cast<int>(size_y_),
                source,
                destination
            );
        }
        else
        {
            planC2R_ = detail::fftw_traits<T>::plan_c2r_3d(
                static_cast<int>(size_x_),
                static_cast<int>(size_y_),
                static_cast<int>(size_z_),
                source,
                destination
            );
        }
        detail::ensure_fftw_plan(planC2R_ != nullptr, "C2R");
        inverse_source_ = source;
        inverse_destination_ = destination;
    }

    void destroy_plans()
    {
        if(planR2C_)
        {
            detail::fftw_traits<T>::destroy(planR2C_);
            planR2C_ = nullptr;
        }
        forward_source_ = nullptr;
        forward_destination_ = nullptr;
        if(planC2R_)
        {
            detail::fftw_traits<T>::destroy(planC2R_);
            planC2R_ = nullptr;
        }
        inverse_source_ = nullptr;
        inverse_destination_ = nullptr;
    }

    void move_from(fftw_wrap_R2C& other)
    {
        rank_ = other.rank_;
        size_x_ = other.size_x_;
        size_y_ = other.size_y_;
        size_z_ = other.size_z_;
        reduced_size_ = other.reduced_size_;
        planR2C_ = other.planR2C_;
        planC2R_ = other.planC2R_;
        forward_source_ = other.forward_source_;
        forward_destination_ = other.forward_destination_;
        inverse_source_ = other.inverse_source_;
        inverse_destination_ = other.inverse_destination_;
        other.planR2C_ = nullptr;
        other.planC2R_ = nullptr;
        other.forward_source_ = nullptr;
        other.forward_destination_ = nullptr;
        other.inverse_source_ = nullptr;
        other.inverse_destination_ = nullptr;
    }

    int rank_ = 1;
    std::size_t size_x_ = 0;
    std::size_t size_y_ = 1;
    std::size_t size_z_ = 1;
    std::size_t reduced_size_ = 0;
    plan_type planR2C_ = nullptr;
    plan_type planC2R_ = nullptr;
    real_type* forward_source_ = nullptr;
    complex_type* forward_destination_ = nullptr;
    complex_type* inverse_source_ = nullptr;
    real_type* inverse_destination_ = nullptr;
};

template<class T>
class fftw_wrap_C2C
{
public:
    using real_type = T;
    using complex_type = typename detail::fftw_traits<T>::complex_type;
    using plan_type = typename detail::fftw_traits<T>::plan_type;

    explicit fftw_wrap_C2C(std::size_t size_x):
        rank_(1),
        size_x_(size_x),
        size_y_(1),
        size_z_(1)
    {
    }

    fftw_wrap_C2C(std::size_t size_x, std::size_t size_y):
        rank_(2),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(1)
    {
    }

    fftw_wrap_C2C(std::size_t size_x, std::size_t size_y, std::size_t size_z):
        rank_(3),
        size_x_(size_x),
        size_y_(size_y),
        size_z_(size_z)
    {
    }

    fftw_wrap_C2C(const fftw_wrap_C2C&) = delete;
    fftw_wrap_C2C& operator=(const fftw_wrap_C2C&) = delete;

    ~fftw_wrap_C2C()
    {
        destroy_plans();
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
        ensure_forward_plan(source, destination);
        detail::fftw_traits<T>::execute(planC2C_forward_);
    }

    void inverse(complex_type* source, complex_type* destination)
    {
        ensure_inverse_plan(source, destination);
        detail::fftw_traits<T>::execute(planC2C_inverse_);
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
    void ensure_forward_plan(complex_type* source, complex_type* destination)
    {
        if(planC2C_forward_ && forward_source_ == source && forward_destination_ == destination)
        {
            return;
        }
        if(planC2C_forward_)
        {
            detail::fftw_traits<T>::destroy(planC2C_forward_);
            planC2C_forward_ = nullptr;
        }
        planC2C_forward_ = create_plan(source, destination, FFTW_FORWARD);
        detail::ensure_fftw_plan(planC2C_forward_ != nullptr, "C2C forward");
        forward_source_ = source;
        forward_destination_ = destination;
    }

    void ensure_inverse_plan(complex_type* source, complex_type* destination)
    {
        if(planC2C_inverse_ && inverse_source_ == source && inverse_destination_ == destination)
        {
            return;
        }
        if(planC2C_inverse_)
        {
            detail::fftw_traits<T>::destroy(planC2C_inverse_);
            planC2C_inverse_ = nullptr;
        }
        planC2C_inverse_ = create_plan(source, destination, FFTW_BACKWARD);
        detail::ensure_fftw_plan(planC2C_inverse_ != nullptr, "C2C inverse");
        inverse_source_ = source;
        inverse_destination_ = destination;
    }

    plan_type create_plan(complex_type* source, complex_type* destination, int sign)
    {
        if(rank_ == 1)
        {
            return detail::fftw_traits<T>::plan_c2c_1d(static_cast<int>(size_x_), source, destination, sign);
        }
        if(rank_ == 2)
        {
            return detail::fftw_traits<T>::plan_c2c_2d(
                static_cast<int>(size_x_),
                static_cast<int>(size_y_),
                source,
                destination,
                sign
            );
        }
        return detail::fftw_traits<T>::plan_c2c_3d(
            static_cast<int>(size_x_),
            static_cast<int>(size_y_),
            static_cast<int>(size_z_),
            source,
            destination,
            sign
        );
    }

    void destroy_plans()
    {
        if(planC2C_forward_)
        {
            detail::fftw_traits<T>::destroy(planC2C_forward_);
            planC2C_forward_ = nullptr;
        }
        forward_source_ = nullptr;
        forward_destination_ = nullptr;
        if(planC2C_inverse_)
        {
            detail::fftw_traits<T>::destroy(planC2C_inverse_);
            planC2C_inverse_ = nullptr;
        }
        inverse_source_ = nullptr;
        inverse_destination_ = nullptr;
    }

    int rank_ = 1;
    std::size_t size_x_ = 0;
    std::size_t size_y_ = 1;
    std::size_t size_z_ = 1;
    plan_type planC2C_forward_ = nullptr;
    plan_type planC2C_inverse_ = nullptr;
    complex_type* forward_source_ = nullptr;
    complex_type* forward_destination_ = nullptr;
    complex_type* inverse_source_ = nullptr;
    complex_type* inverse_destination_ = nullptr;
};

} // namespace external_libraries

template<class T>
using fftw_wrap_R2C = external_libraries::fftw_wrap_R2C<T>;

template<class T>
using fftw_wrap_C2C = external_libraries::fftw_wrap_C2C<T>;

#endif
