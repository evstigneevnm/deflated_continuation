#ifndef __PENDING_BRANCH_EVENT_H__
#define __PENDING_BRANCH_EVENT_H__

#include <cstdint>

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
        curve_number_ = -1;
        segment_id_ = 0;
    }

    void update(const T lambda, const int curve_number, const std::uint64_t segment_id)
    {
        if(!matches(curve_number, segment_id))
        {
            refinements_ = 0;
        }
        active_ = true;
        lambda_ = lambda;
        curve_number_ = curve_number;
        segment_id_ = segment_id;
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

private:
    bool active_ = false;
    unsigned int refinements_ = 0;
    unsigned int misses_ = 0;
    T lambda_ = T(0);
    int curve_number_ = -1;
    std::uint64_t segment_id_ = 0;
};

} // namespace continuation

#endif // __PENDING_BRANCH_EVENT_H__
