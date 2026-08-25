#ifndef __NONLINEAR_OPERATORS_BRATU_H__
#define __NONLINEAR_OPERATORS_BRATU_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <common/scalar_math.h>
#include <nonlinear_operators/detail/linear_nonlinear_terms.h>
#include <scfd/utils/device_tag.h>

namespace nonlinear_operators
{

namespace bratu_detail
{

template<class T>
__DEVICE_TAG__ T exp(const T& value)
{
    using std::exp;
    return exp(value);
}

template<class T>
T log(const T& value)
{
    using std::log;
    return log(value);
}

template<class T>
T cosh(const T& value)
{
    using std::cosh;
    return cosh(value);
}

template<class T>
T acosh(const T& value)
{
    using std::acosh;
    return acosh(value);
}

template<class T>
T cos(const T& value)
{
    using std::cos;
    return cos(value);
}

template<class T>
T pi()
{
    return static_cast<T>(std::acos(-1.0));
}

template<class T>
T abs(const T& value)
{
    return common::scalar_math::abs(value);
}

template<class T>
T lambda_from_theta(const T& theta)
{
    if(theta == T(0))
    {
        return T(0);
    }
    const T c = cosh(theta/T(2));
    return T(2)*theta*theta/(c*c);
}

template<class T>
T fold_theta()
{
    T left = T(1);
    T right = T(10);
    for(unsigned int iter = 0; iter < 120; ++iter)
    {
        const T mid = (left + right)/T(2);
        const T value = mid*std::tanh(static_cast<double>(mid/T(2))) - T(2);
        if(value < T(0))
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

template<class T>
T lower_branch_theta_from_lambda(const T& lambda)
{
    if(lambda <= T(0))
    {
        return T(0);
    }

    const T theta_star = fold_theta<T>();
    const T lambda_star = lambda_from_theta(theta_star);
    if(lambda > lambda_star*(T(1) + T(1.0e-12)))
    {
        throw std::runtime_error("nonlinear_operators::bratu: lambda is above the Bratu fold.");
    }

    T left = T(0);
    T right = theta_star;
    for(unsigned int iter = 0; iter < 160; ++iter)
    {
        const T mid = (left + right)/T(2);
        if(lambda_from_theta(mid) < lambda)
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

template<class T>
T exact_value_from_theta(const T& x, const T& theta)
{
    if(theta == T(0))
    {
        return T(0);
    }
    return T(2)*log(cosh(theta/T(2))/cosh(theta*(x - T(0.5))));
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

} // namespace bratu_detail

template<class VectorOperations, unsigned int BLOCK_SIZE_x = 64>
class bratu
{
public:
    enum class spatial_discretization
    {
        chebyshev = 0,
        fd3 = 1
    };

    struct is_periodic_orbit_reprojected
    {
        static const bool value = false;
    };

    using vector_operations_real = VectorOperations;
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using ordinal_type = typename VectorOperations::ordinal_type;
    using access_type = bratu_detail::vector_access<VectorOperations>;

    bratu(
        std::size_t interior_size_,
        VectorOperations* vec_ops_,
        spatial_discretization discretization_ = spatial_discretization::chebyshev
    ):
        vec_ops(vec_ops_),
        interior_size(interior_size_),
        total_nodes(interior_size_ + 2),
        discretization(discretization_)
    {
        if(interior_size == 0)
        {
            throw std::runtime_error("nonlinear_operators::bratu expects at least one interior node.");
        }
        common_constructor_operation();
    }

    bratu(std::size_t interior_size_, VectorOperations* vec_ops_, int discretization_id):
        bratu(interior_size_, vec_ops_, discretization_from_int(discretization_id))
    {
    }

    ~bratu()
    {
        vec_ops->stop_use_vector(u_0);
        vec_ops->free_vector(u_0);
        vec_ops->stop_use_vector(d2_matrix);
        vec_ops->free_vector(d2_matrix);
        vec_ops->stop_use_vector(jacobian_matrix);
        vec_ops->free_vector(jacobian_matrix);
    }

    void F(const T_vec& u, const T lambda, T_vec& v)
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::all>(u, lambda, v);
    }

    void linear_residual(const T_vec& u, const T lambda, T_vec& v) const
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::linear>(u, lambda, v);
    }

    void nonlinear_residual(const T_vec& u, const T lambda, T_vec& v) const
    {
        evaluate_spatial_residual<detail::linear_nonlinear_terms::nonlinear>(u, lambda, v);
    }

    void set_linearization_point(const T_vec& u_0_, const T lambda_0_)
    {
        vec_ops->assign(u_0_, u_0);
        lambda_0 = lambda_0_;
        form_jacobian_matrix();
    }

    void jacobian_u(const T_vec& du, T_vec& dv)
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::all>(du, dv);
    }

    void linear_jacobian_u(const T_vec& du, T_vec& dv) const
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::linear>(du, dv);
    }

    void nonlinear_jacobian_u(const T_vec& du, T_vec& dv) const
    {
        apply_spatial_jacobian<detail::linear_nonlinear_terms::nonlinear>(du, dv);
    }

    void jacobian_alpha(T_vec& dv)
    {
        jacobian_alpha(u_0, lambda_0, dv);
    }

    void jacobian_alpha(const T_vec& u, const T&, T_vec& dv)
    {
        const auto up = access_type::data(u);
        auto dvp = access_type::data(dv);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type i)
        {
            dvp[i] = bratu_detail::exp(up[i]);
        }, static_cast<ordinal_type>(interior_size));
    }

    void preconditioner_jacobian_u(T_vec&)
    {
    }

    void solve_jacobian_system(T_vec& rhs_to_solution) const
    {
        ensure_host_work();
        vec_ops->get(jacobian_matrix, host_matrix_work.data(), interior_size*interior_size);
        vec_ops->get(rhs_to_solution, host_rhs_work.data(), interior_size);
        solve_dense_system(host_matrix_work, host_rhs_work);
        vec_ops->set(host_rhs_work.data(), rhs_to_solution, interior_size);
    }

    void preconditioner_jacobian_affine_u(
        T_vec& rhs_to_solution,
        const T jacobian_scale,
        const T identity_shift) const
    {
        ensure_host_work();
        vec_ops->get(
            jacobian_matrix,
            host_matrix_work.data(),
            interior_size*interior_size);
        for(std::size_t row = 0; row < interior_size; ++row)
        {
            for(std::size_t col = 0; col < interior_size; ++col)
            {
                host_matrix_work[index(
                    row,
                    col,
                    interior_size)] *= jacobian_scale;
            }
            host_matrix_work[index(
                row,
                row,
                interior_size)] += identity_shift;
        }
        vec_ops->get(
            rhs_to_solution,
            host_rhs_work.data(),
            interior_size);
        solve_dense_system(host_matrix_work, host_rhs_work);
        vec_ops->set(
            host_rhs_work.data(),
            rhs_to_solution,
            interior_size);
    }

    void physical_solution(T_vec&, T_vec&)
    {
    }

    void project(T_vec&)
    {
    }

    void exact_solution(const T& lambda, T_vec& u_out)
    {
        exact_solution_from_theta(bratu_detail::lower_branch_theta_from_lambda(lambda), u_out);
    }

    void exact_solution_from_theta(const T& theta, T_vec& u_out)
    {
        std::vector<T> host_values(interior_size);
        for(std::size_t i = 0; i < interior_size; ++i)
        {
            host_values[i] = bratu_detail::exact_value_from_theta(interior_points[i], theta);
        }
        if(discretization == spatial_discretization::fd3)
        {
            refine_host_solution(bratu_detail::lambda_from_theta(theta), host_values);
        }
        vec_ops->set(host_values.data(), u_out, interior_size);
    }

    T lambda_from_theta(const T& theta) const
    {
        return bratu_detail::lambda_from_theta(theta);
    }

    spatial_discretization get_spatial_discretization() const
    {
        return discretization;
    }

    static spatial_discretization discretization_from_int(int id)
    {
        switch(id)
        {
            case 0:
                return spatial_discretization::chebyshev;
            case 1:
                return spatial_discretization::fd3;
            default:
                throw std::runtime_error("nonlinear_operators::bratu: unknown spatial discretization id.");
        }
    }

    static const char* discretization_name(spatial_discretization discretization_)
    {
        switch(discretization_)
        {
            case spatial_discretization::chebyshev:
                return "chebyshev";
            case spatial_discretization::fd3:
                return "fd3";
        }
        return "unknown";
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
        std::vector<T> host_u(interior_size);
        vec_ops->get(u_in, host_u.data(), interior_size);
        T max_u = host_u.empty() ? T(0) : host_u.front();
        for(const auto& value: host_u)
        {
            if(value > max_u)
            {
                max_u = value;
            }
        }
        res.clear();
        res.reserve(3);
        res.push_back(max_u);
        res.push_back(vec_ops->norm_l2(u_in));
        res.push_back(host_u[interior_size/2]);
    }

    std::vector<std::string> norm_bifurcation_diagram_labels() const
    {
        return {"max_u", "l2_norm", "center_u"};
    }

    void randomize_vector(T_vec& u_out)
    {
        std::vector<T> host_values(interior_size);
        const unsigned int profile_id = random_profile_counter++%8;
        T amplitude = T(10);
        switch(profile_id)
        {
            case 0:
                amplitude = T(10);
                break;
            case 1:
                amplitude = T(8.75);
                break;
            case 2:
                amplitude = T(7.5);
                break;
            case 3:
                amplitude = T(6.25);
                break;
            case 4:
                amplitude = T(5);
                break;
            case 5:
                amplitude = T(3.75);
                break;
            case 6:
                amplitude = T(2.5);
                break;
            case 7:
                amplitude = T(1.25);
                break;
        }
        const T skew = (profile_id%3 == 0) ? T(-0.15) : ((profile_id%3 == 1) ? T(0) : T(0.15));
        for(std::size_t i = 0; i < interior_size; ++i)
        {
            const T x = interior_points[i];
            const T shape = T(4)*x*(T(1) - x);
            host_values[i] = amplitude*shape*(T(1) + skew*(T(2)*x - T(1)));
        }
        vec_ops->set(host_values.data(), u_out, interior_size);
    }

    const VectorOperations* get_vec_ops_ref() const
    {
        return vec_ops;
    }

    VectorOperations* get_vec_ops_ref()
    {
        return vec_ops;
    }

    std::size_t size() const
    {
        return interior_size;
    }

private:
    VectorOperations* vec_ops;
    std::size_t interior_size;
    std::size_t total_nodes;
    spatial_discretization discretization;
    T_vec u_0;
    T_vec d2_matrix;
    T_vec jacobian_matrix;

    __DEVICE_TAG__ static T scaled_exponential(const T value, const T lambda)
    {
        return lambda*bratu_detail::exp(value);
    }

    template <detail::linear_nonlinear_terms Terms>
    void evaluate_spatial_residual(
        const T_vec& u,
        const T lambda,
        T_vec& v) const
    {
        if constexpr(detail::includes_linear<Terms>())
        {
            matvec(d2_matrix, u, v);
        }

        if constexpr(detail::includes_nonlinear<Terms>())
        {
            const auto up = access_type::data(u);
            auto vp = access_type::data(v);
            const T lambda_l = lambda;
            access_type::for_each([up, vp, lambda_l] __DEVICE_TAG__ (ordinal_type i)
            {
                const T reaction = scaled_exponential(up[i], lambda_l);
                if constexpr(detail::includes_linear<Terms>())
                {
                    vp[i] += reaction;
                }
                else
                {
                    vp[i] = reaction;
                }
            }, static_cast<ordinal_type>(interior_size));
        }
    }

    template <detail::linear_nonlinear_terms Terms>
    void apply_spatial_jacobian(const T_vec& du, T_vec& dv) const
    {
        if constexpr(detail::includes_linear<Terms>())
        {
            matvec(d2_matrix, du, dv);
        }

        if constexpr(detail::includes_nonlinear<Terms>())
        {
            const auto u0p = access_type::data(u_0);
            const auto dup = access_type::data(du);
            auto dvp = access_type::data(dv);
            const T lambda_l = lambda_0;
            access_type::for_each([u0p, dup, dvp, lambda_l] __DEVICE_TAG__ (ordinal_type i)
            {
                const T reaction = scaled_exponential(u0p[i], lambda_l)*dup[i];
                if constexpr(detail::includes_linear<Terms>())
                {
                    dvp[i] += reaction;
                }
                else
                {
                    dvp[i] = reaction;
                }
            }, static_cast<ordinal_type>(interior_size));
        }
    }
    T lambda_0 = T(0);
    unsigned int random_profile_counter = 0;
    std::vector<T> interior_points;
    std::vector<T> d2_host;
    mutable std::vector<T> host_matrix_work;
    mutable std::vector<T> host_rhs_work;

    void common_constructor_operation()
    {
        vec_ops->init_vector(u_0);
        vec_ops->start_use_vector(u_0);
        vec_ops->assign_scalar(T(0), u_0);

        vec_ops->init_vector(d2_matrix);
        vec_ops->start_use_vector(d2_matrix, interior_size*interior_size);
        vec_ops->init_vector(jacobian_matrix);
        vec_ops->start_use_vector(jacobian_matrix, interior_size*interior_size);

        switch(discretization)
        {
            case spatial_discretization::chebyshev:
                build_chebyshev_d2_matrix();
                break;
            case spatial_discretization::fd3:
                build_fd3_d2_matrix();
                break;
        }
        vec_ops->assign(d2_matrix, jacobian_matrix);
    }

    void build_chebyshev_d2_matrix()
    {
        const std::size_t n_total = total_nodes;
        const std::size_t last = n_total - 1;
        std::vector<T> z(n_total);
        std::vector<T> c(n_total, T(1));
        std::vector<T> d(n_total*n_total, T(0));
        std::vector<T> d2_full(n_total*n_total, T(0));
        std::vector<T> d2_interior(interior_size*interior_size, T(0));
        interior_points.resize(interior_size);

        c.front() = T(2);
        c.back() = T(2);
        const T pi = bratu_detail::pi<T>();
        for(std::size_t i = 0; i < n_total; ++i)
        {
            z[i] = bratu_detail::cos(pi*static_cast<T>(i)/static_cast<T>(last));
        }

        for(std::size_t i = 0; i < n_total; ++i)
        {
            T row_sum = T(0);
            for(std::size_t j = 0; j < n_total; ++j)
            {
                if(i == j)
                {
                    continue;
                }
                const T sign = ((i + j)%2 == 0) ? T(1) : T(-1);
                const T value = sign*c[i]/(c[j]*(z[i] - z[j]));
                d[index(i, j, n_total)] = value;
                row_sum += value;
            }
            d[index(i, i, n_total)] = -row_sum;
        }

        for(std::size_t i = 0; i < n_total; ++i)
        {
            for(std::size_t j = 0; j < n_total; ++j)
            {
                T sum = T(0);
                for(std::size_t k = 0; k < n_total; ++k)
                {
                    sum += d[index(i, k, n_total)]*d[index(k, j, n_total)];
                }
                d2_full[index(i, j, n_total)] = T(4)*sum;
            }
        }

        for(std::size_t i = 0; i < interior_size; ++i)
        {
            const std::size_t full_i = i + 1;
            interior_points[i] = (T(1) - z[full_i])/T(2);
            for(std::size_t j = 0; j < interior_size; ++j)
            {
                d2_interior[index(i, j, interior_size)] = d2_full[index(full_i, j + 1, n_total)];
            }
        }
        d2_host = d2_interior;
        vec_ops->set(d2_host.data(), d2_matrix, d2_host.size());
    }

    void build_fd3_d2_matrix()
    {
        std::vector<T> d2_interior(interior_size*interior_size, T(0));
        interior_points.resize(interior_size);

        const T h = T(1)/static_cast<T>(interior_size + 1);
        const T inv_h2 = T(1)/(h*h);
        for(std::size_t i = 0; i < interior_size; ++i)
        {
            interior_points[i] = static_cast<T>(i + 1)*h;
            d2_interior[index(i, i, interior_size)] = T(-2)*inv_h2;
            if(i > 0)
            {
                d2_interior[index(i, i - 1, interior_size)] = inv_h2;
            }
            if(i + 1 < interior_size)
            {
                d2_interior[index(i, i + 1, interior_size)] = inv_h2;
            }
        }
        d2_host = d2_interior;
        vec_ops->set(d2_host.data(), d2_matrix, d2_host.size());
    }

    T host_residual_inf(const std::vector<T>& u, const T& lambda) const
    {
        T worst = T(0);
        for(std::size_t row = 0; row < interior_size; ++row)
        {
            T value = lambda*bratu_detail::exp(u[row]);
            for(std::size_t col = 0; col < interior_size; ++col)
            {
                value += d2_host[index(row, col, interior_size)]*u[col];
            }
            const T value_abs = bratu_detail::abs(value);
            if(value_abs > worst)
            {
                worst = value_abs;
            }
        }
        return worst;
    }

    void form_host_jacobian(const std::vector<T>& u, const T& lambda, std::vector<T>& jacobian) const
    {
        jacobian = d2_host;
        for(std::size_t row = 0; row < interior_size; ++row)
        {
            jacobian[index(row, row, interior_size)] += lambda*bratu_detail::exp(u[row]);
        }
    }

    void refine_host_solution(const T& lambda, std::vector<T>& u) const
    {
        std::vector<T> jacobian(interior_size*interior_size, T(0));
        std::vector<T> rhs(interior_size, T(0));
        const T tolerance = T(1.0e-13);
        for(unsigned int iteration = 0; iteration < 24; ++iteration)
        {
            T residual_inf = T(0);
            for(std::size_t row = 0; row < interior_size; ++row)
            {
                T value = lambda*bratu_detail::exp(u[row]);
                for(std::size_t col = 0; col < interior_size; ++col)
                {
                    value += d2_host[index(row, col, interior_size)]*u[col];
                }
                rhs[row] = -value;
                const T value_abs = bratu_detail::abs(value);
                if(value_abs > residual_inf)
                {
                    residual_inf = value_abs;
                }
            }
            if(residual_inf <= tolerance)
            {
                return;
            }

            form_host_jacobian(u, lambda, jacobian);
            solve_dense_system(jacobian, rhs);
            T update_inf = T(0);
            for(std::size_t row = 0; row < interior_size; ++row)
            {
                u[row] += rhs[row];
                const T update_abs = bratu_detail::abs(rhs[row]);
                if(update_abs > update_inf)
                {
                    update_inf = update_abs;
                }
            }
            if(update_inf <= tolerance)
            {
                return;
            }
        }

        const T residual_inf = host_residual_inf(u, lambda);
        if(residual_inf > T(1.0e-9))
        {
            throw std::runtime_error("nonlinear_operators::bratu: failed to refine FD3 exact solution.");
        }
    }

    void form_jacobian_matrix()
    {
        const auto d2p = access_type::data(d2_matrix);
        auto jp = access_type::data(jacobian_matrix);
        const auto u0p = access_type::data(u_0);
        const T lambda_l = lambda_0;
        const ordinal_type n = static_cast<ordinal_type>(interior_size);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type idx)
        {
            const ordinal_type row = idx/n;
            const ordinal_type col = idx - row*n;
            T value = d2p[idx];
            if(row == col)
            {
                value += scaled_exponential(u0p[row], lambda_l);
            }
            jp[idx] = value;
        }, static_cast<ordinal_type>(interior_size*interior_size));
    }

    void matvec(const T_vec& matrix, const T_vec& x, T_vec& y) const
    {
        const auto mp = access_type::data(matrix);
        const auto xp = access_type::data(x);
        auto yp = access_type::data(y);
        const ordinal_type n = static_cast<ordinal_type>(interior_size);
        access_type::for_each([=] __DEVICE_TAG__ (ordinal_type row)
        {
            T sum = T(0);
            for(ordinal_type col = 0; col < n; ++col)
            {
                sum += mp[row*n + col]*xp[col];
            }
            yp[row] = sum;
        }, n);
    }

    void ensure_host_work() const
    {
        host_matrix_work.resize(interior_size*interior_size);
        host_rhs_work.resize(interior_size);
    }

    static std::size_t index(std::size_t row, std::size_t col, std::size_t rows)
    {
        return row*rows + col;
    }

    static void solve_dense_system(std::vector<T>& matrix, std::vector<T>& rhs)
    {
        const std::size_t n = rhs.size();
        for(std::size_t k = 0; k < n; ++k)
        {
            std::size_t pivot = k;
            T pivot_abs = bratu_detail::abs(matrix[index(k, k, n)]);
            for(std::size_t row = k + 1; row < n; ++row)
            {
                const T value_abs = bratu_detail::abs(matrix[index(row, k, n)]);
                if(value_abs > pivot_abs)
                {
                    pivot = row;
                    pivot_abs = value_abs;
                }
            }
            if(pivot_abs <= T(0))
            {
                throw std::runtime_error("nonlinear_operators::bratu: singular Jacobian matrix.");
            }
            if(pivot != k)
            {
                for(std::size_t col = 0; col < n; ++col)
                {
                    std::swap(matrix[index(k, col, n)], matrix[index(pivot, col, n)]);
                }
                std::swap(rhs[k], rhs[pivot]);
            }
            for(std::size_t row = k + 1; row < n; ++row)
            {
                const T factor = matrix[index(row, k, n)]/matrix[index(k, k, n)];
                matrix[index(row, k, n)] = T(0);
                for(std::size_t col = k + 1; col < n; ++col)
                {
                    matrix[index(row, col, n)] -= factor*matrix[index(k, col, n)];
                }
                rhs[row] -= factor*rhs[k];
            }
        }

        for(std::size_t rr = n; rr-- > 0;)
        {
            T sum = rhs[rr];
            for(std::size_t col = rr + 1; col < n; ++col)
            {
                sum -= matrix[index(rr, col, n)]*rhs[col];
            }
            rhs[rr] = sum/matrix[index(rr, rr, n)];
        }
    }
};

} // namespace nonlinear_operators

#endif
