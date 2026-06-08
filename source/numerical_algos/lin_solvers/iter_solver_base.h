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

#ifndef __SCFD_ITER_SOLVER_BASE_H__
#define __SCFD_ITER_SOLVER_BASE_H__

#include <memory>
#include <scfd/utils/logged_obj_base.h>
#include "detail/default_prec_creator.h"

namespace numerical_algos
{
namespace lin_solvers 
{


template
<
    class LinearOperator, class Preconditioner, class VectorOperations,class Monitor,class Log
>
class iter_solver_base : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using linear_operator_type = LinearOperator;
    using preconditioner_type = Preconditioner;
    using vector_operations_type = VectorOperations;
    using monitor_type = Monitor;
    using log_type = Log;

protected:
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;

    mutable monitor_type           monitor_;
    const vector_operations_type* vec_ops_;
    preconditioner_type* prec_;
    const linear_operator_type* A_;
public:
    iter_solver_base(const vector_operations_type* vec_ops, 
                     Log *log, int obj_log_lev, const std::string& log_msg_prefix):
        logged_obj_t(log, obj_log_lev, log_msg_prefix), 
        monitor_(*vec_ops, log),
        vec_ops_(vec_ops), prec_(nullptr)
    {
        monitor_.set_log_msg_prefix(log_msg_prefix + monitor_.get_log_msg_prefix());
    }

    Monitor         &monitor() { return monitor_; }
    const Monitor   &monitor()const { return monitor_; }

    void set_preconditioner(preconditioner_type *prec, bool own_prec = false) 
    { 
        prec_ = prec; 
    }

    virtual bool solve(const linear_operator_type &A, const vector_type &b, 
                       vector_type &x)const = 0;
    
    virtual void set_operator(const linear_operator_type* A)
    {
        A_ = A;
        if (prec_ != nullptr) 
        {
            prec_->set_operator(A_);
        }
    }
    // virtual bool solve(const vector_type &b, vector_type &x)const = 0;

    virtual ~iter_solver_base()
    {
    }
};

}
}

#endif