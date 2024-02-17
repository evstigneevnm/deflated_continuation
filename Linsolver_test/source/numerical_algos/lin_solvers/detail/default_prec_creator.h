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

#ifndef __SCFD_DEFAULT_PREC_CREATOR_H__
#define __SCFD_DEFAULT_PREC_CREATOR_H__

#include "../preconditioners/dummy.h"

namespace numerical_algos
{
namespace lin_solvers 
{

namespace detail
{

template<class VectorOperations,class LinearOperator,class Preconditioner>
struct default_prec_creator
{
    static std::shared_ptr<Preconditioner> get(std::shared_ptr<VectorOperations>)
    {
        return nullptr;
    }
};

template<class VectorOperations,class LinearOperator>
struct default_prec_creator<VectorOperations,LinearOperator,preconditioners::dummy<VectorOperations,LinearOperator>>
{
    static std::shared_ptr<preconditioners::dummy<VectorOperations,LinearOperator>> 
    get(std::shared_ptr<VectorOperations> vec_ops_)
    {
        return std::make_shared<preconditioners::dummy<VectorOperations,LinearOperator>>(std::move(vec_ops_));
    }
};

}

}
}

#endif
