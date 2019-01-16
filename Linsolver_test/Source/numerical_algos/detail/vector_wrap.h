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

#ifndef __SCFD_VECTOR_WRAP_H__
#define __SCFD_VECTOR_WRAP_H__

namespace numerical_algos
{
namespace detail
{


template<class VectorOperations>
struct vector_wrap
{
    typedef VectorOperations                        vector_operations_type;
    typedef typename VectorOperations::vector_type  vector_type;

    vector_type                     v_;
    bool                            is_inited_, is_using_;

    vector_wrap() : is_inited_(false), is_using_(false)
    {
    }

    vector_type         &vector() { return v_; }
    const vector_type   &vector()const { return v_; }
    bool                is_inited()const { return is_inited_; }
    bool                is_using()const { return is_using_; }

    void init(const vector_operations_type  &vec_ops)
    {
        if (is_inited_) return;
        vec_ops.init_vector(v_);
        is_inited_ = true;
    }
    void free(const vector_operations_type  &vec_ops)
    {
        if (!is_inited_) return;
        is_inited_ = false;
        vec_ops.free_vector(v_);
    }
    void start_use(const vector_operations_type  &vec_ops)
    {
        if (is_using_) return;
        vec_ops.start_use_vector(v_);
        is_using_ = true;
    }
    void stop_use(const vector_operations_type  &vec_ops)
    {
        if (!is_using_) return;
        is_using_ = false;
        vec_ops.stop_use_vector(v_);
    }  
};

}
}

#endif
