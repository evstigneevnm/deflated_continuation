#ifndef __DEFLATION_SYMMETRY_SOLUTION_STORAGE_H__
#define __DEFLATION_SYMMETRY_SOLUTION_STORAGE_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

template<class SymmetryAdapter>
auto explicit_symmetry_definition_fingerprint(
    const SymmetryAdapter* symmetry_adapter,
    int) -> decltype(
        symmetry_adapter->symmetry_definition_fingerprint(),
        std::string())
{
    return symmetry_adapter->symmetry_definition_fingerprint();
}

template<class SymmetryAdapter>
auto explicit_symmetry_definition_fingerprint(
    const SymmetryAdapter* symmetry_adapter,
    long) -> decltype(
        symmetry_adapter->finite_registry()->has_explicit_definition_fingerprint(),
        std::string())
{
    const auto* registry = symmetry_adapter->finite_registry();
    return registry != nullptr &&
            registry->has_explicit_definition_fingerprint()
        ? registry->definition_fingerprint()
        : std::string{};
}

template<class SymmetryAdapter>
std::string explicit_symmetry_definition_fingerprint(
    const SymmetryAdapter*,
    ...)
{
    return {};
}

template<class SymmetryAdapter>
auto explicit_symmetry_action_names(
    const SymmetryAdapter* symmetry_adapter,
    int) -> decltype(
        symmetry_adapter->symmetry_action_names(),
        std::vector<std::string>())
{
    return symmetry_adapter->symmetry_action_names();
}

template<class SymmetryAdapter>
auto explicit_symmetry_action_names(
    const SymmetryAdapter* symmetry_adapter,
    long) -> decltype(
        symmetry_adapter->finite_registry()->action_names(),
        std::vector<std::string>())
{
    const auto* registry = symmetry_adapter->finite_registry();
    return registry != nullptr &&
            registry->has_explicit_definition_fingerprint()
        ? registry->action_names()
        : std::vector<std::string>{};
}

template<class SymmetryAdapter>
std::vector<std::string> explicit_symmetry_action_names(
    const SymmetryAdapter*,
    ...)
{
    return {};
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
auto align_orbit_closest_to_reference_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& reference,
    const Vector& source,
    Vector& destination,
    int) -> decltype(
        symmetry_adapter->align_orbit_closest_to_reference(
            reference,
            source,
            destination),
        void())
{
    symmetry_adapter->align_orbit_closest_to_reference(
        reference,
        source,
        destination);
}

template<class SymmetryAdapter, class Vector>
void align_orbit_closest_to_reference_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector&,
    const Vector& source,
    Vector& destination,
    long)
{
    stabilize_canonical_if_available(
        symmetry_adapter,
        source,
        destination,
        0);
}

template<class SymmetryAdapter, class Vector>
auto align_orbit_and_tangent_closest_to_reference_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& reference,
    const Vector& source,
    const Vector& source_tangent,
    Vector& destination,
    Vector& tangent_destination,
    int) -> decltype(
        symmetry_adapter->align_orbit_and_tangent_closest_to_reference(
            reference,
            source,
            source_tangent,
            destination,
            tangent_destination),
        bool())
{
    symmetry_adapter->align_orbit_and_tangent_closest_to_reference(
        reference,
        source,
        source_tangent,
        destination,
        tangent_destination);
    return true;
}

template<class SymmetryAdapter, class Vector>
bool align_orbit_and_tangent_closest_to_reference_if_available(
    SymmetryAdapter* symmetry_adapter,
    const Vector& reference,
    const Vector& source,
    const Vector& source_tangent,
    Vector& destination,
    Vector& tangent_destination,
    long)
{
    (void)symmetry_adapter;
    (void)reference;
    (void)source;
    (void)source_tangent;
    (void)destination;
    (void)tangent_destination;
    return false;
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

    void align_orbit_closest_to_reference(
        const vector_type&,
        const vector_type& source,
        vector_type& destination) const
    {
        stabilize(source, destination);
    }

    void align_orbit_and_tangent_closest_to_reference(
        const vector_type& reference,
        const vector_type& source,
        const vector_type& source_tangent,
        vector_type& destination,
        vector_type& tangent_destination) const
    {
        align_orbit_closest_to_reference(
            reference,
            source,
            destination);
        vec_ops->assign(source_tangent, tangent_destination);
    }

    void pullback_distance_gradient(
        const vector_type&,
        const vector_type&,
        const vector_type& slice_gradient,
        vector_type& gradient) const
    {
        vec_ops->assign(slice_gradient, gradient);
    }

    void pullback_canonical_distance_gradient(
        const vector_type& source,
        const vector_type& slice_state,
        const vector_type& slice_gradient,
        vector_type& gradient) const
    {
        pullback_distance_gradient(
            source,
            slice_state,
            slice_gradient,
            gradient);
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
        duplicate_tolerance_(duplicate_tolerance_),
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

    std::string symmetry_definition_fingerprint() const
    {
        require_symmetry_adapter();
        return detail::explicit_symmetry_definition_fingerprint(
            symmetry_adapter,
            0);
    }

    std::vector<std::string> symmetry_action_names() const
    {
        require_symmetry_adapter();
        return detail::explicit_symmetry_action_names(
            symmetry_adapter,
            0);
    }

    double duplicate_tolerance() const
    {
        return duplicate_tolerance_;
    }

    std::size_t duplicates_skipped_since_clear() const
    {
        return duplicates_skipped_since_clear_;
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
        for(const auto& stored: container)
        {
            if(orbit_distance(stored.get_ref(), vect) <=
               static_cast<T>(duplicate_tolerance()))
            {
                ++duplicates_skipped_since_clear_;
                detail::log_symmetry_solution_storage_message(log, "deflation::symmetry_solution_storage: skipped duplicate symmetry-orbit solution");
                return;
            }
        }

        detail::stabilize_canonical_if_available(symmetry_adapter, vect, x_hat, 0);
        container.emplace_back(vec_ops, x_hat, log);
        elements_number++;
    }

    void clear()
    {
        container.clear();
        elements_number = 0;
        duplicates_skipped_since_clear_ = 0;
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
        T nearest = std::numeric_limits<T>::infinity();
        for(const auto& stored: container)
        {
            nearest = std::min(
                nearest,
                orbit_distance(stored.get_ref(), vect));
        }
        return static_cast<double>(nearest);
    }

    double canonical_distance(const T_vec& left, const T_vec& right)
    {
        require_symmetry_adapter();
        return static_cast<double>(orbit_distance(left, right));
    }

    void align_endpoint_geometry(
        const T_vec& reference,
        const T_vec& source,
        const T_vec& source_tangent,
        T_vec& aligned_source,
        T_vec& aligned_tangent)
    {
        require_symmetry_adapter();
        if(!detail::align_orbit_and_tangent_closest_to_reference_if_available(
               symmetry_adapter,
               reference,
               source,
               source_tangent,
               aligned_source,
               aligned_tangent,
               0))
        {
            detail::align_orbit_closest_to_reference_if_available(
                symmetry_adapter,
                reference,
                source,
                aligned_source,
                0);
            vec_ops->assign(source_tangent, aligned_tangent);
        }
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

        const T_vec& get_ref() const
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

    T distance_contribution(const T norm, const T total_elements) const
    {
        return T(1)/(std::pow(norm, P)*total_elements);
    }

    T distance_derivative_factor(const T norm, const T total_elements) const
    {
        return P/(std::pow(norm, P + T(2))*total_elements);
    }

    T orbit_distance(const T_vec& reference, const T_vec& source)
    {
        detail::align_orbit_closest_to_reference_if_available(
            symmetry_adapter,
            reference,
            source,
            distance_help,
            0);
        vec_ops->assign_mul(
            T(1),
            distance_help,
            T(-1),
            reference,
            slice_gradient);
        return vec_ops->norm_l2(slice_gradient);
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
    double duplicate_tolerance_;
    T norm_weight;
    T P;
    bool ignore_zero_ = false;
    unsigned int number_of_solutions_;
    unsigned int elements_number = 0;
    std::size_t duplicates_skipped_since_clear_ = 0;
    T distance = T(1);
    T_vec distance_help;
    T_vec slice_gradient;
    T_vec x_hat;
    T_vec x0_hat;
    std::vector<internal_container> container;
};

} // namespace deflation

#endif
