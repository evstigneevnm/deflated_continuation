#ifndef __STABILITY_ANALYSIS_STABILITY_POINT_RESULT_H__
#define __STABILITY_ANALYSIS_STABILITY_POINT_RESULT_H__

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <stability/eigensolvers/eigensolver_result.h>

namespace stability
{
namespace analysis
{

enum class stable_halfplane
{
    left,
    right
};

enum class spectrum_classification_status
{
    complete,
    incomplete,
    eigensolver_failure,
    invalid_input
};

inline const char* spectrum_classification_status_name(
    spectrum_classification_status status)
{
    switch(status)
    {
    case spectrum_classification_status::complete:
        return "complete";
    case spectrum_classification_status::incomplete:
        return "incomplete";
    case spectrum_classification_status::eigensolver_failure:
        return "eigensolver_failure";
    case spectrum_classification_status::invalid_input:
        return "invalid_input";
    }
    return "unknown";
}

struct unstable_dimension
{
    int real = 0;
    int complex_pairs = 0;

    int real_subspace_dimension() const
    {
        return real + 2*complex_pairs;
    }

    bool operator==(const unstable_dimension& other) const
    {
        return
            real == other.real &&
            complex_pairs == other.complex_pairs;
    }

    bool operator!=(const unstable_dimension& other) const
    {
        return !(*this == other);
    }

    std::pair<int, int> as_pair() const
    {
        return {real, complex_pairs};
    }
};

inline int unstable_subspace_dimension(
    const std::pair<int, int>& dimension)
{
    return dimension.first + 2*dimension.second;
}

template<class Real>
struct stability_point_result
{
    using real_type = Real;
    using eigenpair_type = eigensolvers::eigenpair_estimate<Real>;

    eigensolvers::eigensolver_status eigensolver_status =
        eigensolvers::eigensolver_status::invalid_input;
    spectrum_classification_status classification_status =
        spectrum_classification_status::invalid_input;
    std::vector<eigenpair_type> eigenpairs;

    unstable_dimension unstable;
    int stable_real = 0;
    int stable_complex_pairs = 0;
    int neutral_real = 0;
    int neutral_complex_pairs = 0;
    std::size_t unclassified_eigenvalues = 0;
    std::size_t unmatched_complex_eigenvalues = 0;
    std::size_t classification_attempts = 1;
    std::string diagnostic;

    bool classification_complete() const
    {
        return
            classification_status ==
            spectrum_classification_status::complete;
    }

    bool succeeded() const
    {
        return
            eigensolver_status ==
                eigensolvers::eigensolver_status::success &&
            classification_complete();
    }

    std::pair<int, int> unstable_dimension_pair() const
    {
        return unstable.as_pair();
    }
};

} // namespace analysis
} // namespace stability

#endif
