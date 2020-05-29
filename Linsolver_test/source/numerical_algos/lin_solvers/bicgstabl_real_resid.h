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

#ifndef __SCFD_BICGSTABL_H__
#define __SCFD_BICGSTABL_H__

#include <numerical_algos/detail/vectors_arr_wrap_static.h>
#include "detail/monitor_call_wrap.h"
#include "iter_solver_base.h"

#ifndef SCFD_BICGSTABL_MAX_BASIS_SIZE
#define SCFD_BICGSTABL_MAX_BASIS_SIZE 10
#endif

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
class bicgstabl : public iter_solver_base<LinearOperator,Preconditioner,
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
    static const int                                            max_basis_sz_ = SCFD_BICGSTABL_MAX_BASIS_SIZE;
    typedef scalar_type                                         T;
    typedef utils::logged_obj_base<Log>                         logged_obj_t;
    typedef iter_solver_base<LinearOperator,Preconditioner,
                             VectorOperations,Monitor,Log>      parent_t;
    typedef vectors_arr_wrap_static<VectorOperations, 2>        buf_t;
    typedef typename buf_t::vectors_arr_use_wrap_type           buf_use_wrap_t;
    typedef vectors_arr_wrap_static<VectorOperations,
                                    max_basis_sz_ + 1>          bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;
    typedef detail::monitor_call_wrap<VectorOperations,
                                      Monitor>                  monitor_call_wrap_t;

    mutable buf_t        buf;
    mutable bufs_arr_t   r;
    mutable bufs_arr_t   u;
    vector_type          &rtilde, &real_ri;

    bool                 use_real_resid_, use_precond_resid_;
    int                  basis_sz_;

    mutable int          flag; 

    void    calc_residual_(const linear_operator_type &A, const vector_type &x, const vector_type &b, vector_type &r)const
    {
        A.apply(x, r);
        vec_ops_->add_mul(T(1.f), b, -T(1.f), r);
        if (prec_ != NULL) prec_->apply(r);
    }
    int     normalize_(vector_type &v)const
    {
        T norm2 = std::sqrt( vec_ops_->scalar_prod(v, v) );
        vec_ops_->assign_mul(T(1.f)/norm2, v, v);
    }

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    bicgstabl(const vector_operations_type *vec_ops, 
              Log *log = NULL, int obj_log_lev = 0) : 
        parent_t(vec_ops, log, obj_log_lev, "bicgstabl::"),
        buf(vec_ops), r(vec_ops), u(vec_ops), rtilde(buf[0]), real_ri(buf[1]),
        use_real_resid_(false), use_precond_resid_(true), basis_sz_(2) 
    {
        buf.init(); r.init(); u.init();
    }

    void    set_basis_size(int basis_sz) { basis_sz_ = basis_sz; }
    void    set_use_precond_resid(bool use_precond_resid) { use_precond_resid_ = use_precond_resid; }
    void    set_use_real_resid(bool use_real_resid) { use_real_resid_ = use_real_resid; }

    virtual bool    solve(const linear_operator_type &A, const vector_type &b, 
                          vector_type &x)const
    {                
        if ((!use_precond_resid_)&&(prec_ != NULL)) 
            throw std::logic_error("bicgstabl::solve: use_precond_resid_ == false with non-empty preconditioner is not supported");
        if (prec_ != NULL) prec_->set_operator(&A);

        bufs_arr_use_wrap_t     use_wrap_r(r), use_wrap_u(u);
        use_wrap_r.start_use_all(); use_wrap_u.start_use_all();

        buf_use_wrap_t          use_wrap_buf(buf);
        use_wrap_buf.start_use_all();

        monitor_call_wrap_t     monitor_wrap(monitor_);
        if ((use_precond_resid_)&&(prec_ != NULL)) {
            vec_ops_->assign(b, r[0]);
            prec_->apply(r[0]);
            monitor_wrap.start(r[0]);
        } else {
            monitor_wrap.start(b);
        }

        vector_type             *checked_ri;
        if (use_real_resid_) {
            checked_ri = &real_ri;
        } else {
            checked_ri = &r[0];
        } 

        vec_ops_->assign_scalar(T(0.f), u[0]);  //u[0] := 0.;

        T       tau[basis_sz_][basis_sz_];      //Hessenberg matrix
        T       sigma[basis_sz_ + 1];
        T       gamma[basis_sz_ + 1];
        T       gamma_p[basis_sz_ + 1];
        T       gamma_pp[basis_sz_ + 1];

        calc_residual_(A, x, b, r[0]);
        vec_ops_->assign_mul(T(1.f), r[0], rtilde);
        if (use_real_resid_) vec_ops_->assign(r[0], real_ri);
        normalize_(rtilde);

        T       rho = T(1.f), alpha = T(0.f), omega = T(1.f);

        bool    res = true;

        flag = 1;                               //default termiantion flag
        
        while (!monitor_.check_finished(x, *checked_ri))
        {       
            rho = -omega * rho;
            for (int j = 0; j < basis_sz_; ++j) {                                       //j=0,...basis_sz_-1
                if ( rho == T(0.f) ) {                                                  //check rho break
                    flag=-1;
                    break; 
                }
            
                T rho1 = vec_ops_->scalar_prod(r[j], rtilde);                           //rho1=(r[j],rtilde)
                T beta = alpha * rho1 / rho; 
                rho = rho1;     
                for (int i = 0; i <= j; ++i){
                    vec_ops_->add_mul(T(1.f), r[i], -beta, u[i]);                       //u[i] := r[i] - beta * u[i]
                }
                A.apply(u[j], u[j+1]);                                                  //u[j+1] := A*u[j]
                if (prec_ != NULL) prec_->apply(u[j+1]);
                alpha = rho / vec_ops_->scalar_prod(u[j+1], rtilde);                    //alpha=rho/(u[j+1],rtilde)
                for (int i = 0; i <= j; ++i){
                    vec_ops_->add_mul(-alpha, u[i+1], T(1.f), r[i]);                    //r[i] := r[i] - alpha * u[i+1]
                }
                A.apply(r[j], r[j+1]);                                                  //r[j+1] := A*r[j] Krylov subspace
                if (prec_ != NULL) prec_->apply(r[j+1]);
                vec_ops_->add_mul(alpha, u[0], T(1.f), x);                              //x := x + alpha * u[0]
            }
            for (int j = 1; j <= basis_sz_; ++j) {
                for (int i = 1; i < j; ++i) {
                    tau[i][j] = vec_ops_->scalar_prod(r[j], r[i]) / sigma[i];
                    vec_ops_->add_mul(-tau[i][j], r[i], T(1.f), r[j]);                  //r[j] := r[j] - tau[i,j] * r[i]
                }
                sigma[j] = vec_ops_->scalar_prod(r[j],r[j]);                            //sigma[j]=(r[j],r[j]);
                gamma_p[j] = vec_ops_->scalar_prod(r[0], r[j]) / sigma[j];              //gamma_p[j]=(r[0],r[j])/sigma[j];
            }
            gamma[basis_sz_] = gamma_p[basis_sz_];                                      //gamma[basis_sz_]=gamma_p[basis_sz_];
            omega = gamma[basis_sz_];                                                   //gamma=gamma[basis_sz_];
            for (int j = basis_sz_-1; j >= 1; --j) {
                gamma[j] = gamma_p[j];                                                  //gamma[j]=gamma_p[j]
                for (int i = j+1; i <= basis_sz_; ++i) {
                    gamma[j] -= tau[j][i] * gamma[i];                                   //gamma[j]=gamma[j]-tau[j,i].*gamma[i];
                }
            }
            for (int j = 1; j < basis_sz_; ++j) {
                gamma_pp[j] = gamma[j+1];                                               //gamma_pp[j]=gamma[j+1]
                for (int i = j+1; i < basis_sz_; ++i){
                    gamma_pp[j] += tau[j][i] * gamma[i+1];                              //gamma_pp[j]=gamma_pp[j]+tau[j,i]*gamma[i+1]
                }
            }
            vec_ops_->add_mul(gamma[1], r[0], T(1.f), x);                               //x := x + gamma[1] * r[0];
            vec_ops_->add_mul(-gamma_p[basis_sz_], r[basis_sz_], T(1.f), r[0]);         //r[0] := r[0] - gamma_p[basis_sz_] * r[basis_sz_]
            vec_ops_->add_mul(-gamma[basis_sz_], u[basis_sz_], T(1.f), u[0]);           //u[0] := u[0] - gamma[basis_sz_] * u[basis_sz_];
            
            for (int j = 1; j < basis_sz_; ++j) {                                       //j=1,..basis_sz_-1
                vec_ops_->add_mul(gamma_pp[j], r[j], T(1.f), x);                        //x := x + gamma_pp[j] * r[j]
                vec_ops_->add_mul(-gamma_p[j], r[j], T(1.f), r[0]);                     //r[0] := r[0] - gamma_p[j] * r[j]
                vec_ops_->add_mul(-gamma[j], u[j], T(1.f), u[0]);                       //u[0] := u[0] - gamma[j] * u[j]
            }

            if (use_real_resid_) calc_residual_(A, x, b, real_ri);

            monitor_ += basis_sz_;
        }

        return res;

    }
};

}
}

#endif
