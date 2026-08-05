#ifndef __PENDING_BRANCH_EVENT_H__
#define __PENDING_BRANCH_EVENT_H__

#include <cstdint>

#include <containers/bifurcation_diagram/curve_provenance.h>

namespace continuation
{

template<class T>
class pending_branch_event
{
public:
    void clear()
    {
        active_ = false;
        refinements_ = 0;
        misses_ = 0;
        lambda_ = T(0);
        previous_lambda_ = T(0);
        previous_prediction_available_ = false;
        curve_number_ = -1;
        segment_id_ = 0;
        target_provenance_ = {};
    }

    void update(
        const T lambda,
        const int curve_number,
        const std::uint64_t segment_id,
        const container::curve_provenance& target_provenance = {})
    {
        const bool same_event = matches(curve_number, segment_id);
        if(!same_event)
        {
            refinements_ = 0;
            previous_prediction_available_ = false;
        }
        else
        {
            previous_lambda_ = lambda_;
            previous_prediction_available_ = true;
        }
        active_ = true;
        lambda_ = lambda;
        curve_number_ = curve_number;
        segment_id_ = segment_id;
        target_provenance_ = target_provenance;
        misses_ = 0;
    }

    bool note_miss(const unsigned int maximum_misses)
    {
        if(!active_)
        {
            return false;
        }
        ++misses_;
        if(misses_ > maximum_misses)
        {
            clear();
            return true;
        }
        return false;
    }

    void increment_refinements()
    {
        ++refinements_;
    }

    bool active() const
    {
        return active_;
    }

    bool matches(const int curve_number, const std::uint64_t segment_id) const
    {
        return active_ && curve_number_ == curve_number && segment_id_ == segment_id;
    }

    unsigned int refinements() const
    {
        return refinements_;
    }

    T lambda() const
    {
        return lambda_;
    }

    int curve_number() const
    {
        return curve_number_;
    }

    std::uint64_t segment_id() const
    {
        return segment_id_;
    }

    const container::curve_provenance& target_provenance() const
    {
        return target_provenance_;
    }

    bool prediction_is_stable(const T relative_tolerance) const
    {
        if(!previous_prediction_available_ || relative_tolerance <= T(0))
        {
            return false;
        }
        const T current_abs = lambda_ < T(0) ? -lambda_ : lambda_;
        const T previous_abs =
            previous_lambda_ < T(0) ? -previous_lambda_ : previous_lambda_;
        const T scale = current_abs > previous_abs
            ? (current_abs > T(1) ? current_abs : T(1))
            : (previous_abs > T(1) ? previous_abs : T(1));
        const T difference = lambda_ < previous_lambda_
            ? previous_lambda_ - lambda_
            : lambda_ - previous_lambda_;
        return difference <= relative_tolerance*scale;
    }

private:
    bool active_ = false;
    unsigned int refinements_ = 0;
    unsigned int misses_ = 0;
    T lambda_ = T(0);
    T previous_lambda_ = T(0);
    bool previous_prediction_available_ = false;
    int curve_number_ = -1;
    std::uint64_t segment_id_ = 0;
    container::curve_provenance target_provenance_;
};

} // namespace continuation

#endif // __PENDING_BRANCH_EVENT_H__
