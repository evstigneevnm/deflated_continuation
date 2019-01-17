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

#ifndef __SCFD_BICGSTAB_H__
#define __SCFD_BICGSTAB_H__

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
class bicgstab : public iter_solver_base<LinearOperator,Preconditioner,
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
    typedef vectors_arr_wrap_static<VectorOperations,7>         bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;
    typedef detail::monitor_call_wrap<VectorOperations,
                                      Monitor>                  monitor_call_wrap_t;


    mutable bufs_arr_t   bufs;
    vector_type          &ri, &r_, &pi, &s, &t, &nu_i,
                         &nonprecond_ri, &real_ri;

    bool                 use_real_resid_, use_precond_resid_;

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    bicgstab(const vector_operations_type *vec_ops, 
             Log *log = NULL, int obj_log_lev = 0) : 
        parent_t(vec_ops, log, obj_log_lev, "bicgstab::"),
        bufs(vec_ops),
        ri(bufs[0]),r_(bufs[1]),pi(bufs[2]),
        s(bufs[3]),t(bufs[4]),nu_i(bufs[5]),
        nonprecond_ri(bufs[6]), real_ri(bufs[6]), 
        use_real_resid_(false), use_precond_resid_(true)
    {
        bufs.init();
    }

    void    set_use_precond_resid(bool use_precond_resid) { use_precond_resid_ = use_precond_resid; }
    void    set_use_real_resid(bool use_real_resid) { use_real_resid_ = use_real_resid; }

    virtual bool    solve(const linear_operator_type &A, const vector_type &b, 
                          vector_type &x)const
    {
        if (prec_ != NULL) prec_->set_operator(&A);

        bufs_arr_use_wrap_t     use_wrap(bufs);
        use_wrap.start_use_all();

        monitor_call_wrap_t     monitor_wrap(monitor_);
        if (use_precond_resid_) {
            vec_ops_->assign(b, ri);
            if (prec_ != NULL) prec_->apply(ri);
            monitor_wrap.start(ri);            
        } else {
            monitor_wrap.start(b);
        }

        vector_type             *checked_ri;
        if (use_real_resid_) {
            checked_ri = &real_ri;
        } else if (use_precond_resid_) {
            checked_ri = &ri;
        } else {
            checked_ri = &nonprecond_ri;
        }

        //ri := P*b - P*A*x0;
        A.apply(x, ri);                               //ri := A*x0
        vec_ops_->add_mul(T(1), b, -T(1), ri);        //ri := -ri + b = -A*x0 + b
        if (!use_precond_resid_ &&  use_real_resid_) vec_ops_->assign(ri, real_ri);
        if (!use_precond_resid_ && !use_real_resid_) vec_ops_->assign(ri, nonprecond_ri);
        if (prec_ != NULL) prec_->apply(ri);          //ri := P*ri = -P*A*x0 + P*b
        if ( use_precond_resid_ &&  use_real_resid_) vec_ops_->assign(ri, real_ri);
        //r_ := ri;
        vec_ops_->assign(ri, r_);

        T   rho_i_1(1), alpha(1), omega_i_1(1);

        vec_ops_->assign_scalar(T(0), nu_i);
        vec_ops_->assign_scalar(T(0), pi);

        bool not_valid_coeff_faced = false;

        while (!monitor_.check_finished(x, *checked_ri))
        {
            //nu_i, pi, ri are nu_{i-1}, p_{i-1}, r_{i-1}
            T   rho_i = vec_ops_->scalar_prod(ri, r_),
                beta = (rho_i/rho_i_1)*(alpha/omega_i_1);
            if (isnan(beta)) { logged_obj_t::info("solve: stop iterations because beta is ind"); not_valid_coeff_faced = true; break; }
            if (isinf(beta)) { logged_obj_t::info("solve: stop iterations because beta is inf"); not_valid_coeff_faced = true; break; }

            //pi := ri - beta*omega_{i_1}*nu_i + beta*p_{i-1}
            vec_ops_->add_mul(T(1), ri, -beta*omega_i_1, nu_i, beta, pi);
            //pi now is pi
            //nu_i := P*A*pi
            A.apply(pi, nu_i);
            if (!use_precond_resid_ && !use_real_resid_) {
                //use t as buffer 4 nonprecond nu_i
                vec_ops_->assign(nu_i, t);
            }
            if (prec_ != NULL) prec_->apply(nu_i);
            //nu_i now is nu_i

            alpha = rho_i/vec_ops_->scalar_prod(nu_i, r_);
            if (isnan(alpha)) { logged_obj_t::info("solve: stop iterations because alpha is nan"); not_valid_coeff_faced = true; break; }
            if (isinf(alpha)) { logged_obj_t::info("solve: stop iterations because alpha is inf"); not_valid_coeff_faced = true; break; }

            //s := ri - alpha*nu_i
            vec_ops_->assign_mul(T(1), ri, -alpha, nu_i, s);
            if (!use_precond_resid_ && !use_real_resid_) {
                //NOTE t here is nonpreconditioned nu_i
                vec_ops_->add_mul(-alpha, t, T(1), nonprecond_ri);
                //nonprecond_ri here is nonpreconditioned s
            }

            //theta := P*A*pi
            A.apply(s, t);
            if (!use_precond_resid_ && !use_real_resid_) {
                //use ri as buffer 4 nonprecond t
                vec_ops_->assign(t, ri);
            }
            if (prec_ != NULL) prec_->apply(t);

            T omega_i = vec_ops_->scalar_prod(t, s)/vec_ops_->scalar_prod(t, t);
            if (isnan(omega_i)) { logged_obj_t::info("solve: stop iterations because omega_i is nan"); not_valid_coeff_faced = true; break; }
            if (isinf(omega_i)) { logged_obj_t::info("solve: stop iterations because omega_i is inf"); not_valid_coeff_faced = true; break; }

            //x := x + alpha*pi + omega_i*s
            vec_ops_->add_mul(alpha, pi, omega_i, s, T(1), x);

            if (!use_precond_resid_ && !use_real_resid_) {
                //NOTE ri here is nonpreconditioned t
                //NOTE nonprecond_ri here is nonpreconditioned s
                vec_ops_->add_mul(-omega_i, ri, T(1), nonprecond_ri);
                //nonprecond_ri now is nonpreconditioned ri
            }

            //ri := s - omega_i*t
            vec_ops_->assign_mul(T(1), s, -omega_i, t, ri);
            //ri now is ri

            if (use_real_resid_) {
                //real_ri := P*b - P*A*x0;
                A.apply(x, real_ri);                                                //real_ri := A*x
                vec_ops_->add_mul(T(1), b, -T(1), real_ri);                         //real_ri := -real_ri + b = -A*x + b
                if (use_precond_resid_ && (prec_ != NULL)) prec_->apply(real_ri);   //real_ri := P*real_ri = -P*A*x + P*b
            }

            omega_i_1 = omega_i; rho_i_1 = rho_i;
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
