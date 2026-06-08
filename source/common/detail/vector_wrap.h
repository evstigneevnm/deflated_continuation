// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_VECTOR_INIT_USE_WRAP_H__
#define __SCFD_VECTOR_INIT_USE_WRAP_H__

#include <cassert>

namespace scfd
{
namespace linspace
{
namespace detail
{

template<class VectorSpace, bool do_init_default = true, bool do_start_use_default = false>
struct vector_wrap
{
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    const vector_space_type &space_;
    vector_type             v_;
    bool                    is_inited_, is_using_;

    vector_wrap(
        const VectorSpace &space, bool do_init = do_init_default, bool do_start_use = do_start_use_default
    ) : space_(space), is_inited_(false), is_using_(false)
    {
        if (do_init) 
        {
            init();
        }
        if (do_start_use) 
        {
            start_use();
        }
    }
    ~vector_wrap()
    {
        stop_use();
        free();
    }

    vector_type         &operator*() { return v_; }
    const vector_type   &operator*()const { return v_; }

    void init()
    {
        if (is_inited_) return;
        space_.init_vector(v_);
        is_inited_ = true;
    }
    void free()
    {
        if (!is_inited_) return;
        is_inited_ = false;
        space_.free_vector(v_);
    }
    void start_use()
    {
        if (is_using_) return;
        space_.start_use_vector(v_);
        is_using_ = true;
    }
    void stop_use()
    {
        if (!is_using_) return;
        is_using_ = false;
        space_.stop_use_vector(v_);
    }
};

}   // namespace detail
}   // namespace linspace
}   // namespace scfd

#endif