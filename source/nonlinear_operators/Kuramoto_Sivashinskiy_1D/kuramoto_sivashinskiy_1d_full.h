#ifndef __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_1D_FULL_H__
#define __NONLINEAR_OPERATORS_KURAMOTO_SIVASHINSKIY_1D_FULL_H__

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>
#include <symmetry/fourier/real_packed_fourier_slice_1d_adapter.h>

namespace nonlinear_operators
{

template<class VectorOperations, class FFTBackend, unsigned int BLOCK_SIZE_x = 64>
class kuramoto_sivashinskiy_1d_full
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
    using symmetry_adapter_type = symmetry::fourier::real_packed_fourier_slice_1d_adapter<VectorOperations>;

    kuramoto_sivashinskiy_1d_full(
        const T& a_val_,
        const T& b_val_,
        std::size_t physical_size_,
        VectorOperations* vec_ops_
    ):
        vec_ops(vec_ops_),
        physical_size_(physical_size_),
        complex_size_(physical_size_/2 + 1),
        mode_count_(complex_size_ > 1 ? complex_size_ - 2 : 0),
        state_size_(2*mode_count_),
        fft_plan(physical_size_),
        symmetry_adapter(vec_ops_, mode_count_),
        a_val(a_val_),
        b_val(b_val_)
    {
        if(physical_size_ < 4 || physical_size_%2 != 0)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d_full expects an even physical grid with at least 4 points.");
        }
        if(vec_ops->get_default_size() != state_size_)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d_full vector size must be 2*(physical_size/2 - 1).");
        }
        common_constructor_operation();
    }

    ~kuramoto_sivashinskiy_1d_full()
    {
        vec_ops->stop_use_vector(u_0);
        vec_ops->free_vector(u_0);
        vec_ops->stop_use_vector(projected_state);
        vec_ops->free_vector(projected_state);
        vec_ops->stop_use_vector(projected_vector);
        vec_ops->free_vector(projected_vector);
        vec_ops->stop_use_vector(projected_residual);
        vec_ops->free_vector(projected_residual);
        vec_ops->stop_use_vector(projected_field);
        vec_ops->free_vector(projected_field);
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
        return state_size_;
    }

    std::size_t state_size() const
    {
        return state_size_;
    }

    std::size_t mode_count() const
    {
        return mode_count_;
    }

    std::size_t positive_modes() const
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

    void F(const T_vec& u, const T lambda, T_vec& v)
    {
        reduced_to_complex(u, u_hat);
        compute_nonlinearity_spectral(u_hat, nonlin_hat);
        assemble_reduced_rhs(u_hat, nonlin_hat, lambda, v);
    }

    void set_linearization_point(const T_vec& u_0_, const T lambda_0_)
    {
        vec_ops->assign(u_0_, u_0);
        lambda_0 = lambda_0_;
        reduced_to_complex(u_0, u0_hat);
        compute_nonlinearity_spectral(u0_hat, nonlin0_hat);
    }

    void set_projected_linearization_point(const T_vec& u_0_, const T lambda_0_)
    {
        symmetry_adapter.stabilize(u_0_, projected_state);
        set_linearization_point(projected_state, lambda_0_);
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        reduced_to_complex(du, du_hat);
        compute_jacobian_nonlinearity_spectral(u0_hat, du_hat, nonlin_hat);
        assemble_reduced_rhs(du_hat, nonlin_hat, lambda_0, dv);
    }

    void jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, lambda_0, dv);
    }

    void jacobian_alpha(const T_vec& u, const T&, T_vec& dv)
    {
        reduced_to_complex(u, u_hat);
        compute_nonlinearity_spectral(u_hat, nonlin_hat);
        const auto up = u_hat.raw_ptr();
        const auto np = nonlin_hat.raw_ptr();
        auto dvp = access_type::data(dv);
        const T a = a_val;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const std::size_t mode = static_cast<std::size_t>(i) + 1;
            const T k = static_cast<T>(mode);
            const T real_value = a*complex_access_type::real(np[mode]) - k*k*complex_access_type::real(up[mode]);
            const T imag_value = a*complex_access_type::imag(np[mode]) - k*k*complex_access_type::imag(up[mode]);
            const std::size_t offset = 2*static_cast<std::size_t>(i);
            dvp[offset] = real_value;
            dvp[offset + 1] = imag_value;
        }, static_cast<ordinal_type>(mode_count_));
    }

    void project_current_tangent(const T_vec& source, T_vec& destination)
    {
        symmetry_adapter.project_tangent(u_0, source, destination);
    }

    void project_tangent_at(const T_vec& state_on_slice, const T_vec& source, T_vec& destination)
    {
        symmetry_adapter.project_tangent(state_on_slice, source, destination);
    }

    void projected_F(const T_vec& u, const T lambda, T_vec& v)
    {
        symmetry_adapter.stabilize(u, projected_state);
        F(projected_state, lambda, projected_residual);
        symmetry_adapter.project_tangent(projected_state, projected_residual, v);
    }

    void projected_F_at_linearization(T_vec& v)
    {
        F(u_0, lambda_0, projected_residual);
        project_current_tangent(projected_residual, v);
    }

    void projected_jacobian_u(const T_vec& du, T_vec& dv)
    {
        symmetry_adapter.stabilizer_differential_from_last(du, projected_vector);
        jacobian_u(projected_vector, projected_residual);
        F(u_0, lambda_0, projected_field);
        symmetry_adapter.projected_vector_field_differential_from_last(
            projected_vector,
            projected_field,
            projected_residual,
            dv);
    }

    void projected_jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, lambda_0, projected_residual);
        project_current_tangent(projected_residual, dv);
    }

    void preconditioner_jacobian_u(T_vec& rhs_to_solution) const
    {
        solve_jacobian_system(rhs_to_solution);
    }

    void solve_jacobian_system(T_vec& rhs_to_solution) const
    {
        auto xp = access_type::data(rhs_to_solution);
        const T lambda = lambda_0;
        const T b = b_val;
        const T eps = T(128)*std::numeric_limits<T>::epsilon();
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const T k = static_cast<T>(i + 1);
            const T k2 = k*k;
            T diag = lambda*(-k2) + b*k2*k2;
            const T abs_diag = diag < T(0) ? -diag : diag;
            if(abs_diag < eps)
            {
                diag = diag < T(0) ? -eps : eps;
            }
            const std::size_t offset = 2*static_cast<std::size_t>(i);
            xp[offset] /= diag;
            xp[offset + 1] /= diag;
        }, static_cast<ordinal_type>(mode_count_));
    }

    void physical_solution(T_vec& u_in, T_vec& u_out)
    {
        physical_solution(static_cast<const T_vec&>(u_in), u_out);
    }

    void physical_solution(const T_vec& u_in, T_vec& u_out)
    {
        if(static_cast<std::size_t>(u_out.size()) != physical_size_)
        {
            throw std::runtime_error("kuramoto_sivashinskiy_1d_full::physical_solution: output vector has wrong size.");
        }
        reduced_to_complex(u_in, u_hat);
        inverse_to_physical(u_hat, u_out);
    }

    void project(T_vec& x)
    {
        symmetry_adapter.stabilize(x, x);
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
        projected_F(u, lambda_0, residual);
        const T quality = vec_ops->norm_l2(residual);
        vec_ops->stop_use_vector(residual);
        vec_ops->free_vector(residual);
        return quality;
    }

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res) const
    {
        std::vector<T> host_u(state_size_, T(0));
        vec_ops->get(u_in, host_u.data(), state_size_);
        res.clear();
        res.reserve(5);
        res.push_back(vec_ops->norm_l2(u_in));
        if(mode_count_ > 0)
        {
            const T re = host_u[0];
            const T im = host_u[1];
            res.push_back(ks1d_detail::sqrt(re*re + im*im));
            res.push_back(re);
            res.push_back(im);
        }
        else
        {
            res.push_back(T(0));
            res.push_back(T(0));
            res.push_back(T(0));
        }
        if(mode_count_ > 1)
        {
            const T re = host_u[2];
            const T im = host_u[3];
            res.push_back(ks1d_detail::sqrt(re*re + im*im));
        }
        else
        {
            res.push_back(T(0));
        }
    }

    std::vector<std::string> norm_bifurcation_diagram_labels() const
    {
        return {"l2_norm", "mode_1_abs", "mode_1_re", "mode_1_im", "mode_2_abs"};
    }

    void randomize_vector(T_vec& u_out)
    {
        std::vector<T> host_values(state_size_, T(0));
        const unsigned int profile_id = random_profile_counter++%8;
        const T amplitude = T(0.02) + T(0.01)*static_cast<T>(profile_id%8);
        for(std::size_t mode = 1; mode <= mode_count_; ++mode)
        {
            const T k = static_cast<T>(mode);
            const T sign_re = ((mode + profile_id)%2 == 0) ? T(1) : T(-1);
            const T sign_im = ((mode + profile_id)%3 == 0) ? T(1) : T(-1);
            const std::size_t offset = 2*(mode - 1);
            const T high_mode_filter = (mode <= 2) ? T(1) : T(0.05);
            host_values[offset] = sign_re*high_mode_filter*amplitude/(k*k);
            host_values[offset + 1] = sign_im*high_mode_filter*amplitude/(k*(k + T(1)));
        }
        vec_ops->set(host_values.data(), u_out, state_size_);
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
    std::size_t state_size_;
    fft_type fft_plan;
    symmetry_adapter_type symmetry_adapter;
    T a_val;
    T b_val;
    T lambda_0 = T(0);
    unsigned int random_profile_counter = 0;

    T_vec u_0;
    T_vec projected_state;
    T_vec projected_vector;
    T_vec projected_residual;
    T_vec projected_field;
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
        vec_ops->init_vector(projected_state);
        vec_ops->start_use_vector(projected_state);
        vec_ops->assign_scalar(T(0), projected_state);
        vec_ops->init_vector(projected_vector);
        vec_ops->start_use_vector(projected_vector);
        vec_ops->assign_scalar(T(0), projected_vector);
        vec_ops->init_vector(projected_residual);
        vec_ops->start_use_vector(projected_residual);
        vec_ops->assign_scalar(T(0), projected_residual);
        vec_ops->init_vector(projected_field);
        vec_ops->start_use_vector(projected_field);
        vec_ops->assign_scalar(T(0), projected_field);

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
            const std::size_t offset = 2*static_cast<std::size_t>(i);
            sp[mode] = complex_access_type::make(rp[offset], rp[offset + 1]);
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
            const std::size_t offset = 2*static_cast<std::size_t>(i);
            rp[offset] = linear*complex_access_type::real(up[mode])
                + lambda*a*complex_access_type::real(np[mode]);
            rp[offset + 1] = linear*complex_access_type::imag(up[mode])
                + lambda*a*complex_access_type::imag(np[mode]);
        }, static_cast<ordinal_type>(mode_count_));
    }

    __DEVICE_TAG__ static complex_type mode_value(const complex_type* spectrum, const int mode)
    {
        if(mode == 0)
        {
            return complex_access_type::make(T(0), T(0));
        }
        if(mode > 0)
        {
            return spectrum[static_cast<std::size_t>(mode)];
        }
        const complex_type value = spectrum[static_cast<std::size_t>(-mode)];
        return complex_access_type::make(complex_access_type::real(value), -complex_access_type::imag(value));
    }

    __DEVICE_TAG__ static complex_type mul_complex(const complex_type& left, const complex_type& right)
    {
        const T ar = complex_access_type::real(left);
        const T ai = complex_access_type::imag(left);
        const T br = complex_access_type::real(right);
        const T bi = complex_access_type::imag(right);
        return complex_access_type::make(ar*br - ai*bi, ar*bi + ai*br);
    }

    __DEVICE_TAG__ static complex_type derivative_coeff(const int mode, const complex_type& value)
    {
        const T k = static_cast<T>(mode);
        const T real = complex_access_type::real(value);
        const T imag = complex_access_type::imag(value);
        return complex_access_type::make(-k*imag, k*real);
    }

    void compute_nonlinearity_spectral(const complex_vector_type& spectrum, complex_vector_type& nonlinearity_spectrum)
    {
        assign_zero_complex(nonlinearity_spectrum);
        const auto up = spectrum.raw_ptr();
        auto np = nonlinearity_spectrum.raw_ptr();
        const int K = static_cast<int>(mode_count_);
        const T scale = T(1)/static_cast<T>(physical_size_);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const int m = static_cast<int>(i) + 1;
            T sum_re = T(0);
            T sum_im = T(0);
            for(int q = -K; q <= K; ++q)
            {
                const int p = m - q;
                if(p < -K || p > K || p == 0 || q == 0)
                {
                    continue;
                }
                const complex_type u_p = mode_value(up, p);
                const complex_type du_q = derivative_coeff(q, mode_value(up, q));
                const complex_type term = mul_complex(u_p, du_q);
                sum_re += complex_access_type::real(term);
                sum_im += complex_access_type::imag(term);
            }
            np[static_cast<std::size_t>(m)] = complex_access_type::make(scale*sum_re, scale*sum_im);
        }, static_cast<ordinal_type>(mode_count_));
    }

    void compute_jacobian_nonlinearity_spectral(
        const complex_vector_type& base_spectrum,
        const complex_vector_type& perturbation_spectrum,
        complex_vector_type& nonlinearity_spectrum)
    {
        assign_zero_complex(nonlinearity_spectrum);
        const auto up = base_spectrum.raw_ptr();
        const auto vp = perturbation_spectrum.raw_ptr();
        auto np = nonlinearity_spectrum.raw_ptr();
        const int K = static_cast<int>(mode_count_);
        const T scale = T(1)/static_cast<T>(physical_size_);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            const int m = static_cast<int>(i) + 1;
            T sum_re = T(0);
            T sum_im = T(0);
            for(int q = -K; q <= K; ++q)
            {
                const int p = m - q;
                if(p < -K || p > K || p == 0 || q == 0)
                {
                    continue;
                }
                const complex_type v_p = mode_value(vp, p);
                const complex_type u_p = mode_value(up, p);
                const complex_type du_q = derivative_coeff(q, mode_value(up, q));
                const complex_type dv_q = derivative_coeff(q, mode_value(vp, q));
                const complex_type term_left = mul_complex(v_p, du_q);
                const complex_type term_right = mul_complex(u_p, dv_q);
                sum_re += complex_access_type::real(term_left) + complex_access_type::real(term_right);
                sum_im += complex_access_type::imag(term_left) + complex_access_type::imag(term_right);
            }
            np[static_cast<std::size_t>(m)] = complex_access_type::make(scale*sum_re, scale*sum_im);
        }, static_cast<ordinal_type>(mode_count_));
    }
};

} // namespace nonlinear_operators

#endif
