#ifndef __INTERSECTION_STATUS_H__
#define __INTERSECTION_STATUS_H__

namespace container
{

struct intersection_status
{
    unsigned int added = 0;
    unsigned int failed = 0;
    unsigned int missing_data = 0;
    unsigned int skipped_discontinuous = 0;

    bool ok() const
    {
        return failed == 0 && missing_data == 0;
    }

    unsigned int incomplete() const
    {
        return failed + missing_data;
    }

    intersection_status& operator+=(const intersection_status& that)
    {
        added += that.added;
        failed += that.failed;
        missing_data += that.missing_data;
        skipped_discontinuous += that.skipped_discontinuous;
        return *this;
    }
};

}

#endif
