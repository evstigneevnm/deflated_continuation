// Copyright © 2020-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_MATRIX_OPERATOR_H__
#define __NMFD_MATRIX_OPERATOR_H__

#include <memory>

namespace nmfd
{
namespace operations
{

template<class Operations>
class matrix_operator : public Operations::matrix_type
{
    using parent_t = typename Operations::matrix_type;
    using ops_t = Operations;
    using vector_t = typename Operations::vector_type;
    using matrix_t = typename Operations::matrix_type;
    using vector_space_t = typename Operations::vector_space_type;
public:
    using scalar_type = typename Operations::scalar_type;
    using vector_type = vector_t;
    using vector_space_type = vector_space_t;
public:
    /// TODO
    /*matrix_operator(std::shared_ptr<Operations> ops) : 
        ops_(ops)
    {
    }*/
    matrix_operator(std::shared_ptr<Operations> ops, matrix_t &&mat) : 
        parent_t(std::move(mat)),
        ops_(ops)
    {
    }

    void apply(const vector_type &x, vector_type &y) const 
    {
        ops_->assign_scalar(scalar_type(0),y);
        ops_->add_matrix_vector_prod(scalar_type(1), *this, x, scalar_type(1), y);
    }

    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return ops_->get_matrix_im_space(*this);
    }
    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return ops_->get_matrix_dom_space(*this);
    }

protected:
    std::shared_ptr<Operations> ops_;
    //std::shared_ptr<matrix_t>   matrix_;
};

} // namespace operations 
} // namespace nmfd

#endif