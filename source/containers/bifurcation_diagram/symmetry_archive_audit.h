#ifndef __BIFURCATION_DIAGRAM_SYMMETRY_ARCHIVE_AUDIT_H__
#define __BIFURCATION_DIAGRAM_SYMMETRY_ARCHIVE_AUDIT_H__

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace container
{

template<class Scalar>
struct symmetry_duplicate_curve_pair
{
    std::size_t first_curve = 0;
    std::size_t second_curve = 0;
    std::size_t matching_samples = 0;
    Scalar minimum_parameter = Scalar(0);
    Scalar maximum_parameter = Scalar(0);
    double maximum_state_distance = 0.0;
};

template<class Scalar>
struct symmetry_archive_audit_result
{
    std::size_t compared_state_pairs = 0;
    std::size_t read_failures = 0;
    std::vector<symmetry_duplicate_curve_pair<Scalar>> duplicate_curve_pairs;

    bool complete() const
    {
        return read_failures == 0;
    }

    bool has_duplicates() const
    {
        return !duplicate_curve_pairs.empty();
    }

    std::string summary() const
    {
        std::ostringstream output;
        output << compared_state_pairs
               << " coincident-parameter state pairs compared";
        if(read_failures != 0)
        {
            output << ", " << read_failures << " state reads failed";
        }
        for(const auto& duplicate: duplicate_curve_pairs)
        {
            output << "; curves " << duplicate.first_curve
                   << " and " << duplicate.second_curve
                   << " have " << duplicate.matching_samples
                   << " quotient-equal samples over lambda=["
                   << duplicate.minimum_parameter << ','
                   << duplicate.maximum_parameter << "]"
                   << " (max distance="
                   << duplicate.maximum_state_distance << ')';
        }
        return output.str();
    }
};

} // namespace container

#endif
