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

#ifndef _CUDA_TIMER_EVENT_H__
#define _CUDA_TIMER_EVENT_H__

#include <utils/cuda_safe_call.h>
#include "timer_event.h"

namespace utils
{

struct cuda_timer_event : public timer_event
{
    cudaEvent_t     e;

    cuda_timer_event()
    {
    }
    ~cuda_timer_event()
    {
    }
    virtual void    init()
    {
        CUDA_SAFE_CALL( cudaEventCreate( &e ) );
    }
    virtual void    record()
    {
        cudaEventRecord( e, 0 );

    }
    virtual double  elapsed_time(const timer_event &e0)const
    {
        const cuda_timer_event *cuda_event = dynamic_cast<const cuda_timer_event*>(&e0);
        if (cuda_event == NULL) {
            throw std::logic_error("cuda_timer_event::elapsed_time: try to calc time from different type of timer (non-cuda)");
        }
        float   res;
        cudaEventSynchronize( e );
        cudaEventElapsedTime( &res, cuda_event->e, e );
        return (double)res;
    };
    virtual void    release()
    {
        cudaEventDestroy( e );
    }
};

}

#endif
