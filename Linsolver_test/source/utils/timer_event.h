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

#ifndef _TIMER_EVENT_H__
#define _TIMER_EVENT_H__

//#include <cuda_safe_call.h>

namespace utils
{

struct timer_event
{
    virtual void    init() = 0;
    virtual void    record() = 0;
    virtual double  elapsed_time(const timer_event &e0)const = 0;
    virtual void    release() = 0;

    /*~timer_event()
    {
        release();
    }*/
};

}

#endif
