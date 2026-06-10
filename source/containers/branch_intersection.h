#ifndef __CONTAINER_BRANCH_INTERSECTION_H__
#define __CONTAINER_BRANCH_INTERSECTION_H__

#include <cstdint>
#include <string>

namespace container
{

template<class T>
struct branch_intersection_policy
{
    bool enabled = false;
    unsigned int signature_norm_index = 0;
    T signature_tolerance = T(1.0e-6);
    T state_tolerance = T(1.0e-8);
    T minimum_step_fraction_from_start = T(1.0e-3);
    bool verbose = false;
};

template<class T>
struct branch_intersection_result
{
    bool found = false;
    T lambda = T(0);
    T signature_distance = T(0);
    T state_distance = T(0);
    T state_tolerance = T(0);
    int curve_number = -1;
    uint64_t segment_id = 0;
    uint64_t semicurve_id = 0;
    uint64_t lower_point_index = 0;
    uint64_t upper_point_index = 0;
    std::string reason;
};

} // namespace container

#endif // __CONTAINER_BRANCH_INTERSECTION_H__
