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

#ifndef __SCFD_CGS_H__
#define __SCFD_CGS_H__

#include <numerical_algos/detail/vectors_arr_wrap_static.h>
#include "detail/monitor_call_wrap.h"
#include "iter_solver_base.h"

namespace numerical_algos
{
namespace lin_solvers 
{

using numerical_algos::detail::vectors_arr_wrap_static;

//demands for template parameters:
//VectorOperations fits VectorOperations concept (see CONCEPTS.txt)
//LinearOperator and Preconditioner fit LinearOperator concept (see CONCEPTS.txt)
//VectorOperations, LinearOperator and Preconditioner have same vector_type

//Monitor concept:
//TODO

template<class LinearOperator,class Preconditioner,
         class VectorOperations,class Monitor,class Log>
class cgs : public iter_solver_base<LinearOperator,Preconditioner,
                                    VectorOperations,Monitor,Log>
{
public:
    typedef typename VectorOperations::scalar_type  scalar_type;
    typedef typename VectorOperations::vector_type  vector_type;
    typedef LinearOperator                          linear_operator_type;
    typedef Preconditioner                          preconditioner_type;
    typedef VectorOperations                        vector_operations_type;
    typedef Monitor                                 monitor_type;
    typedef Log                                     log_type;

private:
    typedef scalar_type                                         T;
    typedef utils::logged_obj_base<Log>                         logged_obj_t;
    typedef iter_solver_base<LinearOperator,Preconditioner,
                             VectorOperations,Monitor,Log>      parent_t;
    typedef vectors_arr_wrap_static<VectorOperations,6>         bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;
    typedef detail::monitor_call_wrap<VectorOperations,
                                      Monitor>                  monitor_call_wrap_t;


    mutable bufs_arr_t   bufs;
    vector_type          &ri, &r0, &pi, &ui, &qi, &theta;

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    cgs(const vector_operations_type *vec_ops, 
        Log *log = NULL, int obj_log_lev = 0) : 
        parent_t(vec_ops, log, obj_log_lev, "cgs::"),
        bufs(vec_ops),
        ri(bufs[0]),r0(bufs[1]),pi(bufs[2]),
        ui(bufs[3]),qi(bufs[4]),theta(bufs[5]) 
    {
        bufs.init();
    }

    virtual bool    solve(const linear_operator_type &A, const vector_type &b, 
                          vector_type &x)const
    {
        if (prec_ != NULL) prec_->set_operator(&A);
        
        bufs_arr_use_wrap_t     use_wrap(bufs);
        use_wrap.start_use_all();

        monitor_call_wrap_t     monitor_wrap(monitor_);
        monitor_wrap.start(b);

        //ri := b - A*x0;
        A.apply(x, ri);                                 //ri := A*x0
        vec_ops_->add_mul(T(1.f), b, -T(1.f), ri);      //ri := -ri + b = -A*x0 + b

        T       rho_i_2;
        bool    not_valid_coeff_faced = false;

        while (!monitor_.check_finished(x, ri))
        {
            if (prec_ != NULL) prec_->apply(ri);            //ri := P*ri

            if (monitor_.iters_performed() == 0) {
                //r0 := ri;
                vec_ops_->assign(ri, r0);
            } 

            T   rho_i_1 = vec_ops_->scalar_prod(ri, r0);
            //TODO check whether rho_k is zero
            if (monitor_.iters_performed() == 0) {
                vec_ops_->assign(ri, ui);
                vec_ops_->assign(ui, pi);
            } else {
                T   beta_i_1 = rho_i_1/rho_i_2;
                if (std::isnan(beta_i_1)) { logged_obj_t::info("solve: stop iterations because beta_i_1 is nan"); not_valid_coeff_faced = true; break; }
                if (std::isinf(beta_i_1)) { logged_obj_t::info("solve: stop iterations because beta_i_1 is inf"); not_valid_coeff_faced = true; break; }
                //ui := ri + beta_i_1 * qi
                vec_ops_->assign_mul(T(1.f), ri, beta_i_1, qi, ui);
                //pi := beta_i_1*beta_i_1 * p{i-1} + ui + beta_i_1 * qi
                vec_ops_->add_mul(   T(1.f), ui, beta_i_1, qi, beta_i_1*beta_i_1, pi);
            }
            //theta := P*A*pi
            A.apply(pi, theta); 
            if (prec_ != NULL) prec_->apply(theta);
            T   alpha_i = rho_i_1/vec_ops_->scalar_prod(theta, r0);
            if (std::isnan(alpha_i)) { logged_obj_t::info("solve: stop iterations because alpha_i is nan"); not_valid_coeff_faced = true; break; }
            if (std::isinf(alpha_i)) { logged_obj_t::info("solve: stop iterations because alpha_i is inf"); not_valid_coeff_faced = true; break; }
            //qi := ui - alpha_i*theta
            vec_ops_->assign_mul(T(1.f), ui, -alpha_i, theta, qi);
            //theta := ui + qi;
            vec_ops_->assign_mul(T(1.f), ui, T(1.f), qi, theta);
            //xi := x{i-1} + theta*alpha_i
            vec_ops_->add_mul(alpha_i, theta, T(1.f), x);
            
            //ri := b - A*xi;
            A.apply(x, ri);                             //ri := A*xi
            vec_ops_->add_mul(T(1.f), b, -T(1.f), ri);  //ri := -ri + b = -A*xi + b

            rho_i_2 = rho_i_1;

            ++monitor_;
        }
        
        if (monitor_.out_min_resid_norm()) vec_ops_->assign(monitor_.min_resid_norm_x(), x);

        if (not_valid_coeff_faced)
            return true;
        else
            return monitor_.converged();
    }
};

}
}

#endif
