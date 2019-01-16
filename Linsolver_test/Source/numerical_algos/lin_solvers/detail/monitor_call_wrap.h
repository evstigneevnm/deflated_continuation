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

#ifndef __SCFD_MONITOR_CALL_WRAP_H__
#define __SCFD_MONITOR_CALL_WRAP_H__

#include <cassert>

namespace numerical_algos
{
namespace lin_solvers 
{
namespace detail
{

template<class VectorOperations, class Monitor>
struct monitor_call_wrap
{
    typedef VectorOperations                        vector_operations_type;
    typedef typename VectorOperations::vector_type  vector_type;

    Monitor                 &monitor_;
    bool                    is_started_;

    monitor_call_wrap(Monitor &monitor) : 
        monitor_(monitor), is_started_(false) {}

    void start(const vector_type& rhs)
    {
        assert(!is_started_);
        monitor_.start(rhs);
        is_started_ = true;
    }
    void stop()
    {
        assert(is_started_);
        is_started_ = false;
        monitor_.stop();
    }

    ~monitor_call_wrap()
    {
        if (is_started_) stop();
    }
};

}
}
}

#endif
