// Copyright © 2016-2018 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef _SYSTEM_TIMER_EVENT_H__
#define _SYSTEM_TIMER_EVENT_H__

//TODO windows realization

#include <sys/time.h>
#include <unistd.h>
#include "timer_event.h"

namespace utils
{

struct system_timer_event : public timer_event
{
    struct timeval tv;

    system_timer_event()
    {
    }
    ~system_timer_event()
    {
    }
    virtual void    init()
    {
    }
    virtual void    record()
    {
        gettimeofday(&tv, NULL);
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const system_timer_event *event = dynamic_cast<const system_timer_event*>(&e0);
        if (event == NULL) {
            throw std::logic_error("system_timer_event::elapsed_time: try to calc time from different type of timer");
        }
        double  res;
        long    seconds, useconds; 
        seconds  = tv.tv_sec  - event->tv.tv_sec;
        useconds = tv.tv_usec - event->tv.tv_usec;
        res = seconds*1000. + useconds/1000.0;
        return res;
    };
    virtual void    release()
    {
    }
};

}

#endif
