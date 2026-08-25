#ifndef __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_1D_H__
#define __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_1D_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#include <cufft.h>
#endif

#include <common/scalar_math.h>
#include <external_libraries/fft_facade.h>
#include <scfd/arrays/array.h>
#include <scfd/utils/device_tag.h>
#include <symmetry/fourier/real_packed_fourier_actions_1d.h>

namespace nonlinear_operators
{

namespace ks1d_detail
{

template<class VectorOperations, class = void>
struct vector_access
{
    using vector_type = typename VectorOperations::vector_type;
    using scalar_type = typename VectorOperations::scalar_type;
    using ordinal_type = typename VectorOperations::ordinal_type;

    static scalar_type* data(vector_type& x)
    {
        return x.data();
    }

    static const scalar_type* data(const vector_type& x)
    {
        return x.data();
    }

    template<class Function>
    static void for_each(Function&& function, ordinal_type n)
    {
        for(ordinal_type i = 0; i < n; ++i)
        {
            function(i);
        }
    }
};

template<class VectorOperations>
struct vector_access<VectorOperations, std::void_t<typename VectorOperations::for_each_type>>
{
    using vector_type = typename VectorOperations::vector_type;
    using scalar_type = typename VectorOperations::scalar_type;
    using ordinal_type = typename VectorOperations::ordinal_type;
    using for_each_type = typename VectorOperations::for_each_type;

    static auto data(vector_type& x) -> decltype(x.raw_ptr())
    {
        return x.raw_ptr();
    }

    static auto data(const vector_type& x) -> decltype(x.raw_ptr())
    {
        return x.raw_ptr();
    }

    template<class Function>
    static void for_each(Function&& function, ordinal_type n)
    {
        for_each_type for_each;
        for_each(std::forward<Function>(function), n);
        for_each.wait();
    }
};

template<class Complex>
struct complex_access;

template<class T>
struct complex_access<std::complex<T>>
{
    using real_type = T;

    __DEVICE_TAG__ static std::complex<T> make(const T real, const T imag)
    {
        return std::complex<T>(real, imag);
    }

    __DEVICE_TAG__ static T real(const std::complex<T>& value)
    {
        return value.real();
    }

    __DEVICE_TAG__ static T imag(const std::complex<T>& value)
    {
        return value.imag();
    }
};

#if defined(__CUDACC__)
template<>
struct complex_access<cufftComplex>
{
    using real_type = float;

    __DEVICE_TAG__ static cufftComplex make(const float real, const float imag)
    {
        cufftComplex value;
        value.x = real;
        value.y = imag;
        return value;
    }

    __DEVICE_TAG__ static float real(const cufftComplex& value)
    {
        return value.x;
    }

    __DEVICE_TAG__ static float imag(const cufftComplex& value)
    {
        return value.y;
    }
};

template<>
struct complex_access<cufftDoubleComplex>
{
    using real_type = double;

    __DEVICE_TAG__ static cufftDoubleComplex make(const double real, const double imag)
    {
        cufftDoubleComplex value;
        value.x = real;
        value.y = imag;
        return value;
    }

    __DEVICE_TAG__ static double real(const cufftDoubleComplex& value)
    {
        return value.x;
    }

    __DEVICE_TAG__ static double imag(const cufftDoubleComplex& value)
    {
        return value.y;
    }
};
#endif

template<class T>
T abs(const T& value)
{
    return common::scalar_math::abs(value);
}

template<class T>
T sqrt(const T& value)
{
    return common::scalar_math::sqrt(value);
}

} // namespace ks1d_detail

template<class VectorOperations, class FFTBackend, unsigned int BLOCK_SIZE_x = 64>
class kuramoto_sivashinskiy_1d
{
public:
    struct is_periodic_orbit_reprojected
    {
        static const bool value = false;
    };

    using vector_operations_real = VectorOperations;
    using fft_backend_type = FFTBackend;
    using fft_type = external_libraries::fft::r2c<fft_backend_type, typename VectorOperations::scalar_type>;
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using ordinal_type = typename VectorOperations::ordinal_type;
    using access_type = ks1d_detail::vector_access<VectorOperations>;
    using complex_type = typename fft_type::complex_type;
    using complex_access_type = ks1d_detail::complex_access<complex_type>;
    using memory_type = typename VectorOperations::memory_type;
    using complex_vector_type = scfd::arrays::array<complex_type, memory_type>;

    kuramoto_sivashinskiy_1d(
        const T& a_val_,
        const T& b_val_,
        std::size_t physical_size_,
        VectorOperations* vec_ops_
    ):
        vec_ops(vec_ops_),
        physical_size_(physical_size_),
        complex_size_(physical_size_/2 + 1),
        mode_count_(complex_size_ > 1 ? complex_size_ - 2 : 0),
        fft_plan(physical_size_),
        a_val(a_val_),
        b_val(b_val_)
    {
        if(physical_size_ < 4 || physical_size_%2 != 0)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d expects an even physical grid with at least 4 points.");
        }
        if(vec_ops->get_default_size() != mode_count_)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d vector size must be physical_size/2 - 1.");
        }
        common_constructor_operation();
    }

    ~kuramoto_sivashinskiy_1d()
    {
        vec_ops->stop_use_vector(u_0);
        vec_ops->free_vector(u_0);
        vec_ops->stop_use_vector(physical_u);
        vec_ops->free_vector(physical_u);
        vec_ops->stop_use_vector(physical_ux);
        vec_ops->free_vector(physical_ux);
        vec_ops->stop_use_vector(physical_du);
        vec_ops->free_vector(physical_du);
        vec_ops->stop_use_vector(physical_dux);
        vec_ops->free_vector(physical_dux);
        vec_ops->stop_use_vector(physical_nonlin);
        vec_ops->free_vector(physical_nonlin);
        vec_ops->stop_use_vector(physical_out);
        vec_ops->free_vector(physical_out);
        free_complex(u_hat);
        free_complex(ux_hat);
        free_complex(du_hat);
        free_complex(dux_hat);
        free_complex(nonlin_hat);
        free_complex(u0_hat);
        free_complex(nonlin0_hat);
        free_complex(ifft_work_hat);
    }

    std::size_t size() const
    {
        return mode_count_;
    }

    std::size_t physical_size() const
    {
        return physical_size_;
    }

    std::size_t complex_size() const
    {
        return complex_size_;
    }

    T linear_multiplier(const std::size_t mode, const T lambda) const
    {
        const T k = static_cast<T>(mode);
        const T k2 = k*k;
        return lambda*(-k2) + b_val*k2*k2;
    }

    template<class FiniteActionRegistry>
    void configure_finite_symmetry_actions(FiniteActionRegistry& registry) const
    {
        registry.reset_to_identity();
        symmetry::fourier::add_sine_half_period_shift_action(registry);
    }

    void F(const T_vec& u, const T lambda, T_vec& v)
    {
        reduced_to_complex(u, u_hat);
        compute_nonlinearity(u_hat, physical_u, physical_ux, ux_hat, physical_nonlin, nonlin_hat);
        assemble_reduced_rhs(u_hat, nonlin_hat, lambda, v);
    }

    void linear_residual(const T_vec& u, const T lambda, T_vec& v) const
    {
        const auto up = access_type::data(u);
        auto vp = access_type::data(v);
        const T b = b_val;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const T k = static_cast<T>(i + 1);
            const T k2 = k*k;
            vp[i] = (lambda*(-k2) + b*k2*k2)*up[i];
        }, static_cast<ordinal_type>(mode_count_));
    }

    void nonlinear_residual(const T_vec& u, const T lambda, T_vec& v)
    {
        reduced_to_complex(u, u_hat);
        compute_nonlinearity(u_hat, physical_u, physical_ux, ux_hat, physical_nonlin, nonlin_hat);
        const auto np = nonlin_hat.raw_ptr();
        auto vp = access_type::data(v);
        const T a = a_val;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t mode = static_cast<std::size_t>(i) + 1;
            vp[i] = lambda*a*complex_access_type::imag(np[mode]);
        }, static_cast<ordinal_type>(mode_count_));
    }

    void set_linearization_point(const T_vec& u_0_, const T lambda_0_)
    {
        vec_ops->assign(u_0_, u_0);
        lambda_0 = lambda_0_;
        reduced_to_complex(u_0, u0_hat);
        compute_nonlinearity(u0_hat, physical_u, physical_ux, ux_hat, physical_nonlin, nonlin0_hat);
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        reduced_to_complex(du, du_hat);
        apply_gradient(du_hat, dux_hat);
        inverse_to_physical(du_hat, physical_du);
        inverse_to_physical(dux_hat, physical_dux);

        const auto dup = access_type::data(physical_du);
        const auto duxp = access_type::data(physical_dux);
        const auto u0p = access_type::data(physical_u);
        const auto ux0p = access_type::data(physical_ux);
        auto nonlinp = access_type::data(physical_nonlin);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            nonlinp[i] = dup[i]*ux0p[i] + u0p[i]*duxp[i];
        }, static_cast<ordinal_type>(physical_size_));
        fft_plan.forward(access_type::data(physical_nonlin), nonlin_hat.raw_ptr());
        assemble_reduced_rhs(du_hat, nonlin_hat, lambda_0, dv);
    }

    void jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, lambda_0, dv);
    }

    void jacobian_alpha(const T_vec& u, const T&, T_vec& dv)
    {
        reduced_to_complex(u, u_hat);
        compute_nonlinearity(u_hat, physical_u, physical_ux, ux_hat, physical_nonlin, nonlin_hat);
        const auto up = u_hat.raw_ptr();
        const auto np = nonlin_hat.raw_ptr();
        auto dvp = access_type::data(dv);
        const T a = a_val;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t mode = static_cast<std::size_t>(i) + 1;
            const T k = static_cast<T>(mode);
            dvp[i] = a*complex_access_type::imag(np[mode]) - k*k*complex_access_type::imag(up[mode]);
        }, static_cast<ordinal_type>(mode_count_));
    }

    void preconditioner_jacobian_u(T_vec& rhs_to_solution) const
    {
        solve_jacobian_system(rhs_to_solution);
    }

    void solve_jacobian_system(T_vec& rhs_to_solution) const
    {
        preconditioner_jacobian_affine_u(
            rhs_to_solution,
            T(1),
            T(0));
    }

    void preconditioner_jacobian_affine_u(
        T_vec& rhs_to_solution,
        const T jacobian_scale,
        const T identity_shift) const
    {
        auto xp = access_type::data(rhs_to_solution);
        const T lambda = lambda_0;
        const T b = b_val;
        const T pole_relative_tolerance =
            std::sqrt(std::numeric_limits<T>::epsilon());
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const T k = static_cast<T>(i + 1);
            const T k2 = k*k;
            const T linear_term =
                jacobian_scale*lambda*(-k2);
            const T biharmonic_term =
                jacobian_scale*b*k2*k2;
            const T diag =
                linear_term + biharmonic_term + identity_shift;
            const T abs_diag = diag < T(0) ? -diag : diag;
            const T abs_linear =
                linear_term < T(0) ? -linear_term : linear_term;
            const T abs_biharmonic =
                biharmonic_term < T(0)
                    ? -biharmonic_term
                    : biharmonic_term;
            const T abs_shift =
                identity_shift < T(0)
                    ? -identity_shift
                    : identity_shift;
            const T diagonal_scale =
                abs_linear + abs_biharmonic + abs_shift;
            const T pole_threshold =
                pole_relative_tolerance*
                (diagonal_scale > T(1) ? diagonal_scale : T(1));
            if(abs_diag > pole_threshold)
            {
                xp[i] /= diag;
            }
        }, static_cast<ordinal_type>(mode_count_));
    }

    std::pair<T, T>
    preconditioner_jacobian_affine_diagonal_range(
        const T jacobian_scale,
        const T identity_shift) const
    {
        T minimum = std::numeric_limits<T>::infinity();
        T maximum = T(0);
        for(std::size_t mode = 1;
            mode <= mode_count_;
            ++mode)
        {
            const T k = static_cast<T>(mode);
            const T k2 = k*k;
            const T diagonal =
                jacobian_scale*
                    (lambda_0*(-k2) + b_val*k2*k2) +
                identity_shift;
            const T absolute =
                diagonal < T(0) ? -diagonal : diagonal;
            minimum = std::min(minimum, absolute);
            maximum = std::max(maximum, absolute);
        }
        return {minimum, maximum};
    }

    T preconditioner_jacobian_affine_min_relative_diagonal(
        const T jacobian_scale,
        const T identity_shift) const
    {
        T minimum = std::numeric_limits<T>::infinity();
        for(std::size_t mode = 1;
            mode <= mode_count_;
            ++mode)
        {
            const T k = static_cast<T>(mode);
            const T k2 = k*k;
            const T jacobian_diagonal =
                jacobian_scale*
                    (lambda_0*(-k2) + b_val*k2*k2);
            const T diagonal =
                jacobian_diagonal + identity_shift;
            const T absolute =
                diagonal < T(0) ? -diagonal : diagonal;
            const T jacobian_absolute =
                jacobian_diagonal < T(0)
                ? -jacobian_diagonal
                : jacobian_diagonal;
            const T shift_absolute =
                identity_shift < T(0)
                ? -identity_shift
                : identity_shift;
            const T scale =
                jacobian_absolute + shift_absolute;
            minimum = std::min(
                minimum,
                scale > T(0) ? absolute/scale : absolute);
        }
        return minimum;
    }

    void physical_solution(T_vec& u_in, T_vec& u_out)
    {
        physical_solution(static_cast<const T_vec&>(u_in), u_out);
    }

    void physical_solution(const T_vec& u_in, T_vec& u_out)
    {
        if(static_cast<std::size_t>(u_out.size()) != physical_size_)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d::physical_solution: output vector has wrong size.");
        }
        reduced_to_complex(u_in, u_hat);
        inverse_to_physical(u_hat, u_out);
    }

    void project(T_vec&)
    {
    }

    void exact_solution(const T&, T_vec& u_out)
    {
        vec_ops->assign_scalar(T(0), u_out);
    }

    T check_solution_quality(const T_vec& u)
    {
        T_vec residual;
        vec_ops->init_vector(residual);
        vec_ops->start_use_vector(residual);
        F(u, lambda_0, residual);
        const T quality = vec_ops->norm_l2(residual);
        vec_ops->stop_use_vector(residual);
        vec_ops->free_vector(residual);
        return quality;
    }

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res) const
    {
        std::vector<T> host_u(mode_count_, T(0));
        vec_ops->get(u_in, host_u.data(), mode_count_);
        res.clear();
        res.reserve(3);
        res.push_back(vec_ops->norm_l2(u_in));
        res.push_back(mode_count_ > 0 ? host_u[0] : T(0));
        res.push_back(mode_count_ > 1 ? host_u[1] : T(0));
    }

    std::vector<std::string> norm_bifurcation_diagram_labels() const
    {
        return {"l2_norm", "mode_1", "mode_2"};
    }

    void randomize_vector(T_vec& u_out)
    {
        std::vector<T> host_values(mode_count_, T(0));
        const unsigned int profile_id = random_profile_counter++%8;
        const T amplitude = T(0.2) + T(0.1)*static_cast<T>(profile_id%4);
        for(std::size_t i = 0; i < mode_count_; ++i)
        {
            const T sign = ((i + profile_id)%2 == 0) ? T(1) : T(-1);
            host_values[i] = sign*amplitude/static_cast<T>((i + 1)*(i + 1));
        }
        vec_ops->set(host_values.data(), u_out, mode_count_);
    }

    const VectorOperations* get_vec_ops_ref() const
    {
        return vec_ops;
    }

    VectorOperations* get_vec_ops_ref()
    {
        return vec_ops;
    }

private:
    VectorOperations* vec_ops;
    std::size_t physical_size_;
    std::size_t complex_size_;
    std::size_t mode_count_;
    fft_type fft_plan;
    T a_val;
    T b_val;
    T lambda_0 = T(0);
    unsigned int random_profile_counter = 0;

    T_vec u_0;
    T_vec physical_u;
    T_vec physical_ux;
    T_vec physical_du;
    T_vec physical_dux;
    T_vec physical_nonlin;
    T_vec physical_out;

    complex_vector_type u_hat;
    complex_vector_type ux_hat;
    complex_vector_type du_hat;
    complex_vector_type dux_hat;
    complex_vector_type nonlin_hat;
    complex_vector_type u0_hat;
    complex_vector_type nonlin0_hat;
    complex_vector_type ifft_work_hat;

    static void free_complex(complex_vector_type& x)
    {
        if(!x.is_free())
        {
            x.free();
        }
    }

    void init_complex(complex_vector_type& x)
    {
        if(x.is_free())
        {
            x.init(static_cast<ordinal_type>(complex_size_));
        }
    }

public:
    void common_constructor_operation()
    {
        vec_ops->init_vector(u_0);
        vec_ops->start_use_vector(u_0);
        vec_ops->assign_scalar(T(0), u_0);

        vec_ops->init_vector(physical_u);
        vec_ops->start_use_vector(physical_u, physical_size_);
        vec_ops->init_vector(physical_ux);
        vec_ops->start_use_vector(physical_ux, physical_size_);
        vec_ops->init_vector(physical_du);
        vec_ops->start_use_vector(physical_du, physical_size_);
        vec_ops->init_vector(physical_dux);
        vec_ops->start_use_vector(physical_dux, physical_size_);
        vec_ops->init_vector(physical_nonlin);
        vec_ops->start_use_vector(physical_nonlin, physical_size_);
        vec_ops->init_vector(physical_out);
        vec_ops->start_use_vector(physical_out, physical_size_);

        init_complex(u_hat);
        init_complex(ux_hat);
        init_complex(du_hat);
        init_complex(dux_hat);
        init_complex(nonlin_hat);
        init_complex(u0_hat);
        init_complex(nonlin0_hat);
        init_complex(ifft_work_hat);
    }

    void assign_zero_complex(complex_vector_type& z)
    {
        auto zp = z.raw_ptr();
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            zp[i] = complex_access_type::make(T(0), T(0));
        }, static_cast<ordinal_type>(complex_size_));
    }

    void copy_complex(const complex_vector_type& source, complex_vector_type& destination)
    {
        const auto sp = source.raw_ptr();
        auto dp = destination.raw_ptr();
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            dp[i] = sp[i];
        }, static_cast<ordinal_type>(complex_size_));
    }

    void reduced_to_complex(const T_vec& reduced, complex_vector_type& spectrum)
    {
        assign_zero_complex(spectrum);
        const auto rp = access_type::data(reduced);
        auto sp = spectrum.raw_ptr();
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t mode = static_cast<std::size_t>(i) + 1;
            sp[mode] = complex_access_type::make(T(0), rp[i]);
        }, static_cast<ordinal_type>(mode_count_));
    }

    void inverse_to_physical(const complex_vector_type& spectrum, T_vec& physical)
    {
        copy_complex(spectrum, ifft_work_hat);
        fft_plan.inverse(ifft_work_hat.raw_ptr(), access_type::data(physical));
        vec_ops->scale(T(1)/static_cast<T>(physical_size_), physical);
    }

    void apply_gradient(const complex_vector_type& source, complex_vector_type& destination)
    {
        const auto sp = source.raw_ptr();
        auto dp = destination.raw_ptr();
        const std::size_t complex_size_l = complex_size_;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t k_idx = static_cast<std::size_t>(i);
            if(k_idx == 0 || k_idx + 1 == complex_size_l)
            {
                dp[k_idx] = complex_access_type::make(T(0), T(0));
            }
            else
            {
                const T k = static_cast<T>(k_idx);
                const T real = complex_access_type::real(sp[k_idx]);
                const T imag = complex_access_type::imag(sp[k_idx]);
                dp[k_idx] = complex_access_type::make(-k*imag, k*real);
            }
        }, static_cast<ordinal_type>(complex_size_));
    }

    void compute_nonlinearity(
        const complex_vector_type& spectrum,
        T_vec& physical,
        T_vec& physical_derivative,
        complex_vector_type& derivative_spectrum,
        T_vec& physical_nonlinearity,
        complex_vector_type& nonlinearity_spectrum)
    {
        apply_gradient(spectrum, derivative_spectrum);
        inverse_to_physical(spectrum, physical);
        inverse_to_physical(derivative_spectrum, physical_derivative);
        const auto up = access_type::data(physical);
        const auto uxp = access_type::data(physical_derivative);
        auto np = access_type::data(physical_nonlinearity);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            np[i] = up[i]*uxp[i];
        }, static_cast<ordinal_type>(physical_size_));
        fft_plan.forward(access_type::data(physical_nonlinearity), nonlinearity_spectrum.raw_ptr());
    }

    void assemble_reduced_rhs(
        const complex_vector_type& source_spectrum,
        const complex_vector_type& nonlinear_spectrum,
        const T lambda,
        T_vec& reduced_rhs)
    {
        const auto up = source_spectrum.raw_ptr();
        const auto np = nonlinear_spectrum.raw_ptr();
        auto rp = access_type::data(reduced_rhs);
        const T a = a_val;
        const T b = b_val;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t mode = static_cast<std::size_t>(i) + 1;
            const T k = static_cast<T>(mode);
            const T k2 = k*k;
            const T linear = lambda*(-k2) + b*k2*k2;
            rp[i] = linear*complex_access_type::imag(up[mode])
                + lambda*a*complex_access_type::imag(np[mode]);
        }, static_cast<ordinal_type>(mode_count_));
    }
};

} // namespace nonlinear_operators

#endif
