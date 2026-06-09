#ifndef __DEFLATION_SYMMETRY_SOLUTION_STORAGE_H__
#define __DEFLATION_SYMMETRY_SOLUTION_STORAGE_H__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <symmetry/stabilized_storage.h>

namespace deflation
{

namespace detail
{

template<class T>
void log_symmetry_solution_storage_message(void*, const T&)
{
}

template<class Log, class T>
auto log_symmetry_solution_storage_message(Log* log, const T& message) -> decltype(log->info_f("%s", message), void())
{
    if(log)
    {
        log->info_f("%s", message);
    }
}

template<class SymmetryAdapter, class Vector>
auto stabilize_canonical_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& source,
    Vector& destination,
    int) -> decltype(symmetry_adapter->stabilize_canonical(source, destination), void())
{
    symmetry_adapter->stabilize_canonical(source, destination);
}

template<class SymmetryAdapter, class Vector>
void stabilize_canonical_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& source,
    Vector& destination,
    long)
{
    symmetry_adapter->stabilize(source, destination);
}

template<class SymmetryAdapter, class Vector>
auto pullback_canonical_distance_gradient_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& source,
    const Vector& slice_state,
    const Vector& slice_gradient,
    Vector& gradient,
    int) -> decltype(symmetry_adapter->pullback_canonical_distance_gradient(source, slice_state, slice_gradient, gradient), void())
{
    symmetry_adapter->pullback_canonical_distance_gradient(source, slice_state, slice_gradient, gradient);
}

template<class SymmetryAdapter, class Vector>
void pullback_canonical_distance_gradient_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& source,
    const Vector& slice_state,
    const Vector& slice_gradient,
    Vector& gradient,
    long)
{
    symmetry_adapter->pullback_distance_gradient(source, slice_state, slice_gradient, gradient);
}

} // namespace detail

template<class VectorOperations>
class identity_symmetry_adapter
{
public:
    using vector_operations_type = VectorOperations;
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit identity_symmetry_adapter(VectorOperations* vec_ops_):
        vec_ops(vec_ops_)
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument("identity_symmetry_adapter got null vector operations");
        }
    }

    void stabilize(const vector_type& source, vector_type& destination) const
    {
        vec_ops->assign(source, destination);
    }

    void stabilize_canonical(const vector_type& source, vector_type& destination) const
    {
        stabilize(source, destination);
    }

    void pullback_distance_gradient(
        const vector_type&,
        const vector_type&,
        const vector_type& slice_gradient,
        vector_type& gradient) const
    {
        vec_ops->assign(slice_gradient, gradient);
    }

private:
    VectorOperations* vec_ops;
};

template<class VectorOperations, class SymmetryAdapter = identity_symmetry_adapter<VectorOperations>, class Log = void>
class symmetry_solution_storage
{
public:
    using vector_operations_type = VectorOperations;
    using symmetry_adapter_type = SymmetryAdapter;
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using host_state_type = std::vector<T>;
    using stabilized_host_storage_type = symmetry::stabilized_storage<host_state_type>;

    symmetry_solution_storage(
        VectorOperations* vec_ops_,
        const unsigned int number_of_solutions,
        const T norm_weight_,
        const T P_ = T(2),
        Log* log_ = nullptr):
        symmetry_solution_storage(
            vec_ops_,
            number_of_solutions,
            norm_weight_,
            P_,
            nullptr,
            default_duplicate_tolerance(),
            log_)
    {
    }

    symmetry_solution_storage(
        VectorOperations* vec_ops_,
        const unsigned int number_of_solutions,
        const T norm_weight_,
        const T P_,
        SymmetryAdapter* symmetry_adapter_,
        const double duplicate_tolerance_ = default_duplicate_tolerance(),
        Log* log_ = nullptr):
        vec_ops(vec_ops_),
        symmetry_adapter(symmetry_adapter_),
        log(log_),
        canonical_host_storage(duplicate_tolerance_),
        norm_weight(norm_weight_),
        P(P_),
        number_of_solutions_(number_of_solutions)
    {
        if(vec_ops == nullptr)
        {
            throw std::invalid_argument("symmetry_solution_storage got null vector operations");
        }
        if(norm_weight == T(0))
        {
            throw std::invalid_argument("symmetry_solution_storage got zero norm weight");
        }

        container.reserve(number_of_solutions);
        vec_ops->init_vector(distance_help);
        vec_ops->start_use_vector(distance_help);
        vec_ops->init_vector(slice_gradient);
        vec_ops->start_use_vector(slice_gradient);
        vec_ops->init_vector(x_hat);
        vec_ops->start_use_vector(x_hat);
        vec_ops->init_vector(x0_hat);
        vec_ops->start_use_vector(x0_hat);
        vec_ops->assign_scalar(T(0), x0_hat);
    }

    ~symmetry_solution_storage()
    {
        clear();
        vec_ops->stop_use_vector(x0_hat);
        vec_ops->free_vector(x0_hat);
        vec_ops->stop_use_vector(x_hat);
        vec_ops->free_vector(x_hat);
        vec_ops->stop_use_vector(slice_gradient);
        vec_ops->free_vector(slice_gradient);
        vec_ops->stop_use_vector(distance_help);
        vec_ops->free_vector(distance_help);
    }

    static double default_duplicate_tolerance()
    {
        return 64.0*static_cast<double>(std::numeric_limits<T>::epsilon());
    }

    void set_symmetry_adapter(SymmetryAdapter* symmetry_adapter_)
    {
        symmetry_adapter = symmetry_adapter_;
    }

    SymmetryAdapter* get_symmetry_adapter() const
    {
        return symmetry_adapter;
    }

    void set_known_solution(const T_vec& x0_p)
    {
        require_symmetry_adapter();
        detail::stabilize_canonical_if_available(symmetry_adapter, x0_p, x0_hat, 0);
        ignore_zero_ = false;
    }

    void set_ignore_zero()
    {
        ignore_zero_ = true;
    }

    void push_back(const T_vec& vect)
    {
        require_symmetry_adapter();
        detail::stabilize_canonical_if_available(symmetry_adapter, vect, x_hat, 0);
        const host_state_type host_state = get_host_state(x_hat);
        if(!canonical_host_storage.add_if_new(host_state))
        {
            detail::log_symmetry_solution_storage_message(log, "deflation::symmetry_solution_storage: skipped duplicate stabilized solution");
            return;
        }
        container.emplace_back(vec_ops, x_hat, log);
        elements_number++;
    }

    void clear()
    {
        container.clear();
        canonical_host_storage.clear();
        elements_number = 0;
    }

    unsigned int get_size() const
    {
        return elements_number;
    }

    unsigned int get_number_of_reserved_solutions() const
    {
        return number_of_solutions_;
    }

    double nearest_stabilized_distance(const T_vec& vect)
    {
        require_symmetry_adapter();
        detail::stabilize_canonical_if_available(symmetry_adapter, vect, x_hat, 0);
        return canonical_host_storage.nearest_distance(get_host_state(x_hat)).first;
    }

    void stabilize(const T_vec& source, T_vec& destination)
    {
        require_symmetry_adapter();
        detail::stabilize_canonical_if_available(symmetry_adapter, source, destination, 0);
    }

    void stabilize_in_place(T_vec& x)
    {
        require_symmetry_adapter();
        detail::stabilize_canonical_if_available(symmetry_adapter, x, x_hat, 0);
        vec_ops->assign(x_hat, x);
    }

    void calc_distance(const T_vec& x, T& beta, T_vec& c)
    {
        calc_distance_norms(x, c, P);
        beta = distance;
    }

    void calc_distance(const T_vec& x, const T_vec& x_translate, T& beta, T_vec& c)
    {
        (void)x_translate;
        calc_distance(x, beta, c);
    }

private:
    class internal_container
    {
    public:
        internal_container(VectorOperations* vec_ops_, const T_vec& vec_, Log* log_ = nullptr):
            vec_ops(vec_ops_),
            log(log_)
        {
            vec_ops->init_vector(array_);
            vec_ops->start_use_vector(array_);
            vec_ops->assign(vec_, array_);
            allocated = true;
            owned = true;
        }

        internal_container(const internal_container& that):
            vec_ops(that.vec_ops),
            log(that.log)
        {
            vec_ops->init_vector(array_);
            vec_ops->start_use_vector(array_);
            vec_ops->assign(that.array_, array_);
            allocated = true;
            owned = true;
        }

        internal_container(internal_container&& that):
            vec_ops(that.vec_ops),
            log(that.log),
            array_(that.array_),
            allocated(that.allocated),
            owned(that.owned)
        {
            that.owned = false;
            that.allocated = false;
        }

        internal_container& operator=(const internal_container&) = delete;

        internal_container& operator=(internal_container&& that)
        {
            if(this == &that)
            {
                return *this;
            }
            release();
            vec_ops = that.vec_ops;
            log = that.log;
            array_ = that.array_;
            allocated = that.allocated;
            owned = that.owned;
            that.owned = false;
            that.allocated = false;
            return *this;
        }

        ~internal_container()
        {
            release();
        }

        T_vec& get_ref()
        {
            return array_;
        }

    private:
        void release()
        {
            if(allocated && owned)
            {
                vec_ops->stop_use_vector(array_);
                vec_ops->free_vector(array_);
                detail::log_symmetry_solution_storage_message(log, "deflation::symmetry_solution_storage: removed solution from container");
            }
            allocated = false;
            owned = false;
        }

        VectorOperations* vec_ops = nullptr;
        Log* log = nullptr;
        T_vec array_;
        bool allocated = false;
        bool owned = false;
    };

    void require_symmetry_adapter() const
    {
        if(symmetry_adapter == nullptr)
        {
            throw std::runtime_error("symmetry_solution_storage requires a symmetry adapter before use");
        }
    }

    host_state_type get_host_state(const T_vec& x) const
    {
        host_state_type host(vec_ops->get_size(x), T(0));
        if(!host.empty())
        {
            vec_ops->get(x, host.data(), host.size());
        }
        return host;
    }

    T distance_contribution(const T norm, const T total_elements) const
    {
        return T(1)/(std::pow(norm, P)*total_elements);
    }

    T distance_derivative_factor(const T norm, const T total_elements) const
    {
        return P/(std::pow(norm, P + T(2))*total_elements);
    }

    void calc_distance_norms(const T_vec& x, T_vec& c, const T p)
    {
        require_symmetry_adapter();
        P = p;
        detail::stabilize_canonical_if_available(symmetry_adapter, x, x_hat, 0);
        vec_ops->assign_scalar(T(0), slice_gradient);

        unsigned int total_elements = elements_number + 1;
        distance = T(0);

        if(!ignore_zero_)
        {
            add_distance_from_reference(x_hat, x0_hat, total_elements);
        }
        else
        {
            total_elements = elements_number;
        }

        if(total_elements == 0)
        {
            distance = T(1);
            vec_ops->assign_scalar(T(0), c);
            return;
        }

        for(unsigned int j = 0; j < elements_number; ++j)
        {
            add_distance_from_reference(x_hat, container[j].get_ref(), total_elements);
        }

        detail::pullback_canonical_distance_gradient_if_available(symmetry_adapter, x, x_hat, slice_gradient, c, 0);
        distance += T(1);
    }

    void add_distance_from_reference(const T_vec& x, T_vec& reference, const unsigned int total_elements)
    {
        vec_ops->assign_mul(T(1), x, T(-1), reference, distance_help);
        const T norm = vec_ops->norm_l2(distance_help);
        distance += distance_contribution(norm, static_cast<T>(total_elements));
        const T factor = distance_derivative_factor(norm, static_cast<T>(total_elements));
        vec_ops->add_mul(factor/norm_weight, distance_help, T(1), slice_gradient);
    }

private:
    VectorOperations* vec_ops;
    SymmetryAdapter* symmetry_adapter;
    Log* log;
    stabilized_host_storage_type canonical_host_storage;
    T norm_weight;
    T P;
    bool ignore_zero_ = false;
    unsigned int number_of_solutions_;
    unsigned int elements_number = 0;
    T distance = T(1);
    T_vec distance_help;
    T_vec slice_gradient;
    T_vec x_hat;
    T_vec x0_hat;
    std::vector<internal_container> container;
};

} // namespace deflation

#endif
