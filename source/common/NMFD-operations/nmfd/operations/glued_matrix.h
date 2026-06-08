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

#ifndef __NMFD_GLUED_MATRIX_H__
#define __NMFD_GLUED_MATRIX_H__

#include <memory>

namespace nmfd
{
namespace operations
{

template<class InternalMatrix, std::size_t n>
class glued_matrix
{
    using internal_matrix_t = InternalMatrix;
public:
    glued_matrix()
    {
        for (std::size_t i = 0;i < n;++i)
        {
            for (std::size_t j = 0;j < n;++j)
            {
                internal_matrices_[i][j] = std::make_shared<internal_matrix_t>();
            }
        }
    }
    explicit glued_matrix(std::array<std::array<std::shared_ptr<internal_matrix_t>, n>, n> internal_matrices): 
        internal_matrices_(internal_matrices)
    {
    }

    /// ISSUE not very consistent because internal_matrix_t is not const here
    const std::shared_ptr<internal_matrix_t> &comp_ptr(std::size_t comp_i, std::size_t comp_j)const
    {
        return internal_matrices_[comp_i][comp_j];
    }
    internal_matrix_t &comp(std::size_t comp_i, std::size_t comp_j)
    {
        return *internal_matrices_[comp_i][comp_j];
    }
    const internal_matrix_t &comp(std::size_t comp_i, std::size_t comp_j)const
    {
        return *internal_matrices_[comp_i][comp_j];
    }

protected:
    std::array<std::array<std::shared_ptr<internal_matrix_t>,n>,n> internal_matrices_;
};

} // namespace operations 
} // namespace nmfd

#endif