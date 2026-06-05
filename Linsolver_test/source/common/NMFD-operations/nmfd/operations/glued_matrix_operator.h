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

#ifndef __NMFD_GLUED_MATRIX_OPERATOR_H__
#define __NMFD_GLUED_MATRIX_OPERATOR_H__

#include <memory>
#include <common/glued_vector_operations.h>
#include <common/glued_vector_space.h>
#include "glued_matrix.h"

namespace nmfd
{
namespace operations
{

template<class Operations, std::size_t n>
class glued_matrix_operator : public glued_matrix<typename Operations::matrix_type,n>
{
    using parent_t = glued_matrix<typename Operations::matrix_type,n>;
    using ops_t = Operations;
    using internal_vector_t = typename Operations::vector_type;
    using internal_matrix_t = typename Operations::matrix_type;
    using internal_vector_space_t = typename Operations::vector_space_type;
public:
    using scalar_type = typename Operations::scalar_type;
    using vector_type = glued_vector<internal_vector_t, n>;
    using vector_space_type = glued_vector_space<internal_vector_space_t, n>;
public:
    /// ISSUE pass internal matrices outside?
    glued_matrix_operator(std::shared_ptr<Operations> internal_ops) : 
        internal_ops_(internal_ops)
    {
    }
    glued_matrix_operator(std::shared_ptr<Operations> internal_ops, std::array<std::array<std::shared_ptr<internal_matrix_t>, n>, n> internal_matrices): 
        parent_t(internal_matrices),
        internal_ops_(internal_ops)
    {
    }
    glued_matrix_operator(std::shared_ptr<Operations> internal_ops, const parent_t &glued_matrix): 
        parent_t(glued_matrix),
        internal_ops_(internal_ops)
    {
    }

    using parent_t::comp_ptr;
    using parent_t::comp;

    void apply(const vector_type &x, vector_type &y) const 
    {
        for (std::size_t i = 0;i < n;++i)
        {
            internal_ops_->assign_scalar(scalar_type(0),y.comp(i));
            for (std::size_t j = 0;j < n;++j)
            {
                internal_ops_->add_matrix_vector_prod(scalar_type(1), comp(i,j), x.comp(j), scalar_type(1), y.comp(i));
            }
        }
    }

    std::shared_ptr<vector_space_type> get_im_space() const
    {
        std::array<std::shared_ptr<internal_vector_space_t>,n> internal_spaces;
        for (std::size_t i = 0;i < n;++i)
        {
            internal_spaces[i] = internal_ops_->get_matrix_im_space(comp(i,0));
        }
        return std::make_shared<vector_space_type>(std::move(internal_spaces));
    }
    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        std::array<std::shared_ptr<internal_vector_space_t>,n> internal_spaces;
        for (std::size_t j = 0;j < n;++j)
        {
            internal_spaces[j] = internal_ops_->get_matrix_dom_space(comp(0,j));
        }
        return std::make_shared<vector_space_type>(std::move(internal_spaces));
    }

protected:
    std::shared_ptr<Operations> internal_ops_;
    using parent_t::internal_matrices_;
};

} // namespace operations 
} // namespace nmfd

#endif