#ifndef __STAR_SHAPED_TEST__
#define __STAR_SHAPED_TEST__

/**
*    Problem class for the star-shaped curve
*
*    F(x, lambda) = sqrt(x^2 + lambda^2) - 1
*                 - C * 4*x*lambda*(x^2 - lambda^2)/(x^2 + lambda^2)^2.
*
*    With x = r cos(theta), lambda = r sin(theta), this is
*    r = 1 + C sin(4 theta).  The implementation is scalar in x, but uses
*    SCFD-style backend for_each so it can be exercised on CPU/GPU backends.
*/

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <common/scalar_math.h>
#include <scfd/utils/device_tag.h>

namespace nonlinear_operators
{

namespace star_shaped_detail
{

template<class T>
__DEVICE_TAG__ T sqrt(const T& value)
{
    using std::sqrt;
    return sqrt(value);
}

template<class T>
__DEVICE_TAG__ T abs(const T& value)
{
    using std::abs;
    return abs(value);
}

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

template<class T>
T scalar_residual(const T& x, const T& lambda, const T& curvature)
{
    const T r2 = x*x + lambda*lambda;
    if(r2 <= T(0))
    {
        return T(-1);
    }
    return common::scalar_math::sqrt(r2) - T(1)
        - T(4)*curvature*x*lambda*(x*x - lambda*lambda)/(r2*r2);
}

} // namespace star_shaped_detail

template<class VectorOperations_R, unsigned int BLOCK_SIZE_x = 64>
class star_shaped
{
public:
    struct is_periodic_orbit_reprojected
    {
        static const bool value = false;
    };

    using vector_operations_real = VectorOperations_R;
    using T = typename VectorOperations_R::scalar_type;
    using T_vec = typename VectorOperations_R::vector_type;
    using ordinal_type = typename VectorOperations_R::ordinal_type;
    using access_type = star_shaped_detail::vector_access<VectorOperations_R>;

    explicit star_shaped(std::size_t Nx_, vector_operations_real* vec_ops_R_, const T& curvature_ = T(0.2)):
        vec_ops_R(vec_ops_R_),
        Nx(Nx_),
        curvature(curvature_)
    {
        common_constructor_operation();
    }

    ~star_shaped()
    {
        vec_ops_R->stop_use_vector(u_0);
        vec_ops_R->free_vector(u_0);
    }

    void F(const T_vec& u, const T lambda, T_vec& v)
    {
        const auto up = access_type::data(u);
        auto vp = access_type::data(v);
        const T lambda_l = lambda;
        const T curvature_l = curvature;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type)
        {
            const T x = up[0];
            const T r2 = x*x + lambda_l*lambda_l;
            const T r = star_shaped_detail::sqrt(r2);
            const T anisotropy = T(4)*curvature_l*x*lambda_l*(x*x - lambda_l*lambda_l)/(r2*r2);
            vp[0] = r - T(1) - anisotropy;
        }, ordinal_type(1));
    }

    void set_linearization_point(const T_vec& u_0_, const T lambda_0_)
    {
        vec_ops_R->assign(u_0_, u_0);
        lambda_0 = lambda_0_;
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        const auto u0p = access_type::data(u_0);
        const auto dup = access_type::data(du);
        auto dvp = access_type::data(dv);
        const T lambda_l = lambda_0;
        const T curvature_l = curvature;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type)
        {
            dvp[0] = jacobian_x_value(u0p[0], lambda_l, curvature_l)*dup[0];
        }, ordinal_type(1));
    }

    void jacobian_u_adjoint(const T_vec& w, T_vec& dv)
    {
        jacobian_u(w, dv);
    }

    void jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, lambda_0, dv);
    }

    void jacobian_alpha(const T_vec& u, const T& lambda, T_vec& dv)
    {
        const auto up = access_type::data(u);
        auto dvp = access_type::data(dv);
        const T curvature_l = curvature;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type)
        {
            dvp[0] = jacobian_lambda_value(up[0], lambda, curvature_l);
        }, ordinal_type(1));
    }

    void preconditioner_jacobian_u(T_vec& rhs_to_solution) const
    {
        solve_jacobian_system(rhs_to_solution);
    }

    void solve_jacobian_system(T_vec& rhs_to_solution) const
    {
        const auto u0p = access_type::data(u_0);
        auto xp = access_type::data(rhs_to_solution);
        const T lambda_l = lambda_0;
        const T curvature_l = curvature;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type)
        {
            const T jac = jacobian_x_value(u0p[0], lambda_l, curvature_l);
            xp[0] /= jac;
        }, ordinal_type(1));
    }

    void preconditioner_jacobian_affine_u(
        T_vec& rhs_to_solution,
        const T jacobian_scale,
        const T identity_shift) const
    {
        const auto u0p = access_type::data(u_0);
        auto xp = access_type::data(rhs_to_solution);
        const T lambda_l = lambda_0;
        const T curvature_l = curvature;
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type)
        {
            const T jac =
                jacobian_x_value(
                    u0p[0],
                    lambda_l,
                    curvature_l);
            xp[0] /=
                jacobian_scale*jac +
                identity_shift;
        }, ordinal_type(1));
    }

    void preconditioner_jacobian_affine_u_adjoint(
        T_vec& rhs_to_solution,
        const T jacobian_scale,
        const T identity_shift) const
    {
        preconditioner_jacobian_affine_u(
            rhs_to_solution,
            jacobian_scale,
            identity_shift);
    }

    void physical_solution(T_vec&, T_vec&)
    {
    }

    void project(T_vec&)
    {
    }

    void exact_solution(const T& lambda, T_vec& u_out)
    {
        const T value = x_root(lambda);
        vec_ops_R->set(&value, u_out, 1);
    }

    T check_solution_quality(const T_vec&)
    {
        return T(0);
    }

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res) const
    {
        T val = T(0);
        vec_ops_R->get(u_in, &val, 1);
        res.clear();
        res.reserve(2);
        res.push_back(val);
        res.push_back(common::scalar_math::abs(val));
    }

    std::vector<std::string> norm_bifurcation_diagram_labels() const
    {
        return {"x", "abs_x"};
    }

    void randomize_vector(T_vec& u_out)
    {
        vec_ops_R->assign_random(u_out);
        T val = T(0);
        vec_ops_R->get(u_out, &val, 1);
        val = T(2)*val - T(1);
        if(common::scalar_math::abs(val) < T(0.05))
        {
            val = val < T(0) ? T(-0.05) : T(0.05);
        }
        vec_ops_R->set(&val, u_out, 1);
    }

private:
    vector_operations_real* vec_ops_R;
    std::size_t Nx;
    T_vec u_0;
    T lambda_0 = T(0);
    T curvature = T(0.2);

    void common_constructor_operation()
    {
        if(Nx != 1)
        {
            throw std::runtime_error("nonlinear_operators::star_shaped expects vector size 1.");
        }
        vec_ops_R->init_vector(u_0);
        vec_ops_R->start_use_vector(u_0);
    }

    static __DEVICE_TAG__ T jacobian_x_value(const T& x, const T& lambda, const T& curvature)
    {
        const T r2 = x*x + lambda*lambda;
        const T r2_cubed = r2*r2*r2;
        return x/star_shaped_detail::sqrt(r2)
            + T(4)*curvature*(x*x*x*x*lambda - T(6)*x*x*lambda*lambda*lambda
            + lambda*lambda*lambda*lambda*lambda)/r2_cubed;
    }

    static __DEVICE_TAG__ T jacobian_lambda_value(const T& x, const T& lambda, const T& curvature)
    {
        const T r2 = x*x + lambda*lambda;
        const T r2_cubed = r2*r2*r2;
        return lambda/star_shaped_detail::sqrt(r2)
            - T(4)*curvature*(x*x*x*x*x - T(6)*x*x*x*lambda*lambda
            + x*lambda*lambda*lambda*lambda)/r2_cubed;
    }

    T positive_x_root(const T& lambda) const
    {
        if(common::scalar_math::abs(lambda) > T(1))
        {
            throw std::runtime_error("nonlinear_operators::star_shaped::positive_x_root expects |lambda| <= 1.");
        }

        T left = T(0);
        T right = T(2);
        T f_left = star_shaped_detail::scalar_residual(left, lambda, curvature);
        T f_right = star_shaped_detail::scalar_residual(right, lambda, curvature);
        while(f_right <= T(0))
        {
            right *= T(2);
            f_right = star_shaped_detail::scalar_residual(right, lambda, curvature);
            if(right > T(32))
            {
                throw std::runtime_error("nonlinear_operators::star_shaped::exact_solution failed to bracket the positive root.");
            }
        }
        if(f_left > T(0))
        {
            throw std::runtime_error("nonlinear_operators::star_shaped::exact_solution positive-root bracket is invalid.");
        }

        for(unsigned int iter = 0; iter < 160; ++iter)
        {
            const T mid = (left + right)/T(2);
            const T f_mid = star_shaped_detail::scalar_residual(mid, lambda, curvature);
            if(f_mid <= T(0))
            {
                left = mid;
            }
            else
            {
                right = mid;
            }
        }
        return (left + right)/T(2);
    }

    T bracketed_x_root(const T& lambda) const
    {
        const int intervals = 4096;
        const T x_min = T(-2);
        const T x_max = T(2);
        const T dx = (x_max - x_min)/T(intervals);
        T left = x_min;
        T f_left = star_shaped_detail::scalar_residual(left, lambda, curvature);

        for(int i = 1; i <= intervals; ++i)
        {
            const T right = (i == intervals) ? x_max : x_min + T(i)*dx;
            const T f_right = star_shaped_detail::scalar_residual(right, lambda, curvature);
            if(f_left == T(0))
            {
                return left;
            }
            if(f_left*f_right <= T(0))
            {
                T a = left;
                T b = right;
                T fa = f_left;
                for(unsigned int iter = 0; iter < 160; ++iter)
                {
                    const T mid = (a + b)/T(2);
                    const T fm = star_shaped_detail::scalar_residual(mid, lambda, curvature);
                    if(fa*fm <= T(0))
                    {
                        b = mid;
                    }
                    else
                    {
                        a = mid;
                        fa = fm;
                    }
                }
                return (a + b)/T(2);
            }
            left = right;
            f_left = f_right;
        }

        throw std::runtime_error("nonlinear_operators::star_shaped::exact_solution failed to bracket a root.");
    }

    T x_root(const T& lambda) const
    {
        if(common::scalar_math::abs(lambda) <= T(1))
        {
            return positive_x_root(lambda);
        }
        return bracketed_x_root(lambda);
    }
};

} // namespace nonlinear_operators

#endif
