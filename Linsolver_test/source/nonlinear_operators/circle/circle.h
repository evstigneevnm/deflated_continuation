#ifndef __CIRCLE_TEST_ND__
#define __CIRCLE_TEST_ND__

/**
*    Problem class for:
*    f(x, lambda) := x*x + lambda*lambda - R^2 = 0
*
*    This is a scalar continuation test problem.  The implementation is written
*    against SCFD-style vector operations and uses the backend for_each instead
*    of a hand-launched CUDA kernel.
*/

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <scfd/utils/device_tag.h>

namespace nonlinear_operators
{

template<class VectorOperations_R, unsigned int BLOCK_SIZE_x = 64>
class circle
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
    using for_each_type = typename VectorOperations_R::for_each_type;

    circle(T R_, std::size_t Nx_, vector_operations_real* vec_ops_R_):
        R(R_),
        vec_ops_R(vec_ops_R_),
        Nx(Nx_)
    {
        common_constructor_operation();
    }

    ~circle()
    {
        vec_ops_R->stop_use_vector(u_0);
        vec_ops_R->free_vector(u_0);
    }

    void F(const T_vec& u, const T alpha, T_vec& v)
    {
        const auto up = u.raw_ptr();
        auto vp = v.raw_ptr();
        const T R_l = R;
        for_each_([=] __DEVICE_TAG__ (ordinal_type)
        {
            vp[0] = up[0]*up[0] + alpha*alpha - R_l*R_l;
        }, ordinal_type(1));
        for_each_.wait();
    }

    void set_linearization_point(const T_vec& u_0_, const T alpha_0_)
    {
        vec_ops_R->assign(u_0_, u_0);
        alpha_0 = alpha_0_;
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        const auto u0p = u_0.raw_ptr();
        const auto dup = du.raw_ptr();
        auto dvp = dv.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type)
        {
            dvp[0] = T(2)*u0p[0]*dup[0];
        }, ordinal_type(1));
        for_each_.wait();
    }

    void jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, alpha_0, dv);
    }

    void jacobian_alpha(const T_vec&, const T& alpha, T_vec& dv)
    {
        auto dvp = dv.raw_ptr();
        for_each_([=] __DEVICE_TAG__ (ordinal_type)
        {
            dvp[0] = T(2)*alpha;
        }, ordinal_type(1));
        for_each_.wait();
    }

    void preconditioner_jacobian_u(T_vec&)
    {
    }

    void physical_solution(T_vec&, T_vec&)
    {
    }

    void project(T_vec&)
    {
    }

    void exact_solution(const T&, T_vec&)
    {
    }

    T check_solution_quality(const T_vec&)
    {
        return T(0);
    }

    void norm_bifurcation_diagram(const T_vec& u_in, std::vector<T>& res) const
    {
        T val = T(0);
        vec_ops_R->get(u_in, &val);
        res.clear();
        res.reserve(2);
        res.push_back(val);
        res.push_back(std::abs(val));
    }

    void randomize_vector(T_vec& u_out)
    {
        vec_ops_R->assign_random(u_out);
    }

private:
    T R;
    vector_operations_real* vec_ops_R;
    std::size_t Nx;
    T_vec u_0;
    T alpha_0 = T(0);
    mutable for_each_type for_each_;

    void common_constructor_operation()
    {
        if(Nx != 1)
        {
            throw std::runtime_error("nonlinear_operators::circle expects vector size 1.");
        }
        vec_ops_R->init_vector(u_0);
        vec_ops_R->start_use_vector(u_0);
    }
};

}

#endif
