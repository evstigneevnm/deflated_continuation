#ifndef __SYMMETRY_CONTINUATION_CONTINUATION_CHART_STATE_H__
#define __SYMMETRY_CONTINUATION_CONTINUATION_CHART_STATE_H__

#include <cstddef>

namespace symmetry
{
namespace continuation
{

template<class ChartData>
struct continuation_chart_state
{
    ChartData data;
    bool uses_lsq = false;
    bool initialized = false;
    std::size_t generation = 0;
};

struct identity_continuation_chart_state
{
    bool initialized = true;
    std::size_t generation = 0;
};

} // namespace continuation
} // namespace symmetry

#endif
