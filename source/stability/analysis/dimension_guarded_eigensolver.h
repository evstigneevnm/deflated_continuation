#ifndef __STABILITY_ANALYSIS_DIMENSION_GUARDED_EIGENSOLVER_H__
#define __STABILITY_ANALYSIS_DIMENSION_GUARDED_EIGENSOLVER_H__

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <stability/eigensolvers/eigensolver_result.h>

namespace stability
{
namespace analysis
{

/**
 * Selects an exact small-system eigensolver when requested and otherwise
 * preserves the primary eigensolver path. The fallback is also available
 * transactionally after a primary failure. Both solvers operate through the
 * same vector-space and linear-operator abstractions.
 */
template<class PrimaryEigensolver, class SmallSystemEigensolver>
class dimension_guarded_eigensolver
{
public:
    using primary_type = PrimaryEigensolver;
    using small_system_type = SmallSystemEigensolver;
    using vector_type = typename primary_type::vector_type;
    using result_type = typename primary_type::result_type;

    static_assert(
        std::is_same<
            vector_type,
            typename small_system_type::vector_type>::value,
        "dimension-guarded eigensolvers require the same vector type");
    static_assert(
        std::is_same<
            result_type,
            typename small_system_type::result_type>::value,
        "dimension-guarded eigensolvers require the same result type");

    dimension_guarded_eigensolver(
        const primary_type& primary,
        const small_system_type& small_system,
        std::size_t dimension,
        std::size_t maximum_small_dimension,
        bool prefer_small_system)
        : primary_(primary),
          small_system_(small_system),
          dimension_(dimension),
          maximum_small_dimension_(maximum_small_dimension),
          prefer_small_system_(prefer_small_system)
    {
        if(dimension_ == 0)
        {
            throw std::invalid_argument(
                "dimension-guarded eigensolver requires a nonzero "
                "dimension");
        }
    }

    result_type execute(const vector_type& initial_vector) const
    {
        if(small_system_available() && prefer_small_system_)
        {
            result_type result =
                small_system_.execute(initial_vector);
            prefix_diagnostic(
                result,
                "small-system eigensolver selected");
            return result;
        }

        result_type primary_result =
            primary_.execute(initial_vector);
        if(primary_result.succeeded() || !small_system_available())
            return primary_result;

        result_type fallback_result =
            small_system_.execute(initial_vector);
        const std::string primary_diagnostic =
            primary_result.diagnostic.empty()
            ? std::string("primary eigensolver failed")
            : primary_result.diagnostic;
        prefix_diagnostic(
            fallback_result,
            "small-system recovery after {" +
                primary_diagnostic + "}");
        return fallback_result;
    }

    bool small_system_available() const
    {
        return
            maximum_small_dimension_ != 0 &&
            dimension_ <= maximum_small_dimension_;
    }

    bool classification_fallback_available() const
    {
        return
            small_system_available() &&
            !prefer_small_system_;
    }

    result_type execute_classification_fallback(
        const vector_type& initial_vector) const
    {
        if(!classification_fallback_available())
        {
            result_type result;
            result.status =
                eigensolvers::eigensolver_status::invalid_input;
            result.coverage_complete = false;
            result.diagnostic =
                "small-system classification fallback is unavailable";
            return result;
        }

        result_type result =
            small_system_.execute(initial_vector);
        prefix_diagnostic(
            result,
            "small-system recovery after incomplete classification");
        return result;
    }

    bool classification_confirmation_available() const
    {
        return small_system_available();
    }

    result_type execute_classification_confirmation(
        const vector_type& initial_vector) const
    {
        if(!classification_confirmation_available())
        {
            result_type result;
            result.status =
                eigensolvers::eigensolver_status::invalid_input;
            result.coverage_complete = false;
            result.diagnostic =
                "small-system classification confirmation is "
                "unavailable";
            return result;
        }

        result_type result =
            small_system_.execute(initial_vector);
        prefix_diagnostic(
            result,
            "small-system classification confirmation");
        return result;
    }

private:
    static void prefix_diagnostic(
        result_type& result,
        const std::string& prefix)
    {
        result.diagnostic =
            result.diagnostic.empty()
            ? prefix
            : prefix + ": " + result.diagnostic;
    }

    const primary_type& primary_;
    const small_system_type& small_system_;
    std::size_t dimension_;
    std::size_t maximum_small_dimension_;
    bool prefer_small_system_;
};

} // namespace analysis
} // namespace stability

#endif
