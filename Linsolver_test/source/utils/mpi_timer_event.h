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

#ifndef _MPI_TIMER_EVENT_H__
#define _MPI_TIMER_EVENT_H__

//TODO windows realization

#include <mpi.h>
#include "timer_event.h"

namespace utils
{

struct mpi_timer_event : public timer_event
{
    double  time;

    mpi_timer_event()
    {
    }
    ~mpi_timer_event()
    {
    }
    virtual void    init()
    {
    }
    virtual void    record()
    {
        time = MPI_Wtime();
    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const mpi_timer_event *event = dynamic_cast<const mpi_timer_event*>(&e0);
        if (event == NULL) {
            throw std::logic_error("mpi_timer_event::elapsed_time: try to calc time from different type of timer");
        }
        return (time - event->time)*1000.;
    };
    virtual void    release()
    {
    }
};

}

#endif
