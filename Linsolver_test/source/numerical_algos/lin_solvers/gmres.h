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

#ifndef __SCFD_GMRES_H__
#define __SCFD_GMRES_H__

#include <stdexcept>
#include <cmath>
#include <numerical_algos/detail/vectors_arr_wrap_static.h>
#include "detail/monitor_call_wrap.h"
#include "iter_solver_base.h"

#ifndef SCFD_GMRES_MAX_BASIS_SIZE
#define SCFD_GMRES_MAX_BASIS_SIZE 100
#endif

namespace numerical_algos
{
namespace lin_solvers 
{

using numerical_algos::detail::vectors_arr_wrap_static;

//demands for template parameters:
//VectorOperations fits VectorOperations concept (see CONCEPTS.txt)
//LinearOperator and Preconditioner fit LinearOperator concept (see CONCEPTS.txt)
//VectorOperations, LinearOperator and Preconditioner have same T_vec

//Monitor concept:
//TODOjj

template<class LinearOperator, class Preconditioner,
     class VectorOperations, class Monitor, class Log>
class gmres : public iter_solver_base<LinearOperator, Preconditioner,
                                          VectorOperations, Monitor, Log>
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
    static const int max_basis_sz_ = SCFD_GMRES_MAX_BASIS_SIZE;
    typedef scalar_type T;
    typedef vector_type T_vec;
    typedef utils::logged_obj_base<Log> logged_obj_t;
    typedef iter_solver_base<
            LinearOperator,
            Preconditioner,
            VectorOperations,
            Monitor,
            Log> parent_t;

    typedef vectors_arr_wrap_static<VectorOperations, 1>        buf_t;
    typedef typename buf_t::vectors_arr_use_wrap_type           buf_use_wrap_t;
    
    typedef vectors_arr_wrap_static<VectorOperations,
                                    max_basis_sz_ + 1>          bufs_arr_t;
    typedef typename bufs_arr_t::vectors_arr_use_wrap_type      bufs_arr_use_wrap_t;
    typedef detail::monitor_call_wrap<VectorOperations,
                                      Monitor>                  monitor_call_wrap_t;

    mutable buf_t buf_r;
    mutable bufs_arr_t V;
    mutable bufs_arr_t buf_other;

    T_vec &w;
    T_vec &r_tilde;
    T_vec &V0;
    T_vec &r;
    
    //host arrays
    mutable T_vec H = nullptr;
    mutable T_vec s = nullptr;
    mutable T_vec cs = nullptr;
    mutable T_vec sn = nullptr;


    bool                 use_precond_resid_;
    int                  resid_recalc_freq_;

    int                  restart_;

    mutable int flag; 

    void calc_residual_(const linear_operator_type &A, const T_vec &x, const T_vec &b, T_vec &r)const
    {
        A.apply(x, r);
        vec_ops_->add_mul(T(1.f), b, -T(1.f), r);
        if (prec_ != NULL) 
        {
            prec_->apply(r);
        }
    }
    
    // normilizes and returns the norm that was used to normilize
    // not L2 norm!
    // TODO: if T is complex, then we have a problem! Fix this!!!
    // For by vector_operations i have a type called ''norm_type''. Should we use it as a must???
    T normalize_(T_vec &v, const T& sign_ = T(1.0))const
    {
        T norm2 = vec_ops_->norm(v);
        vec_ops_->assign_mul(sign_/norm2, v, v);
        return norm2;
    }

    void calc_Krylov_vector_(const linear_operator_type &A, const T_vec &x, T_vec &r)const
    {
        A.apply(x, r);
        if (prec_ != NULL) 
        {
            prec_->apply(r);
        }
    }


    void init_host_()
    {
        if(H==nullptr)
        {
            H = (T_vec) malloc(sizeof(T)*restart_*(restart_+1));
            s = (T_vec) malloc(sizeof(T)*(restart_+1));
            sn = (T_vec) malloc(sizeof(T)*restart_);
            cs = (T_vec) malloc(sizeof(T)*restart_);
            if( (H == NULL)||(s==NULL)||(sn==NULL)||(cs==NULL) )
                throw;
        }
        else
        {
            // realloc didn't work!
            free(H);
            free(s);
            free(sn);
            free(cs);                        
            H = (T_vec) malloc(sizeof(T)*restart_*(restart_+1));
            s = (T_vec) malloc(sizeof(T)*(restart_+1));
            sn = (T_vec) malloc(sizeof(T)*restart_);
            cs = (T_vec) malloc(sizeof(T)*restart_);
            if( (H == NULL)||(s==NULL)||(sn==NULL)||(cs==NULL) )
                throw;
        }
    }

    void zero_host_H_() const
    {
        for(int k=0;k<restart_;k++)
        {
            cs[k] = T(0.0);
            sn[k] = T(0.0);
            for(int j=0;j<=restart_;j++)
            {

                H[j*(restart_) + k] = T(0.0);
            }
        }
    }
    void zero_host_s_() const
    {
        for(int j=0;j<=restart_;j++)
        {
            s[j] = T(0.0);
        }
    }

    // can be replaced by LAPACK, for better quality.
    // we assume that we take H-matrix and s-vector as an RHS.
    // and store the solution in s-vector, which is overwritten.
    void solve_triangular_system_(const T_vec& H, const int ind, T_vec& s) const 
    {
        for (int j = ind; j >= 0; j--) {
            s[j] /= H[j*(restart_) + j];//(j, j);
            // S(0:j) = s(0:j) - s[j] H(0:j,j)
            for (int k = j - 1; k >= 0; k--) {
                s[k] -= H[k*(restart_) + j] * s[j]; //H(k, j)
            }
            //std::cout << std::endl;
        }        

    }

    // *** Givens rotations ***

    //TODO - problem with complex type again! Should return complex conjugate for complex T
    T conj_(const T& val) const
    {   
        return val;
    }
    void apply_plane_rotation_(T& dx, T& dy, const T& cs, const T& sn) const
    {
        T temp = cs * dx + sn * dy;
        dy = -conj_(sn) * dx + cs * dy;
        dx = temp;
    }

    void generate_plane_rotation_(const T& dx, const T& dy, T& cs, T& sn) const
    {
        if (dy == T(0.0))
        {
            cs = T(1.0);
            sn = T(0.0);
        }
        // else
        // {
        //     //TODO: norm type problem again!
        //     T scale = std::abs(dx) + std::abs(dy);
        //     T norm = scale * std::sqrt(std::abs(dx / scale) * std::abs(dx / scale) +
        //                                       std::abs(dy / scale) * std::abs(dy / scale));
        //     T alpha = dx / std::abs(dx);
        //     cs = std::abs(dx) / norm;
        //     sn = alpha * conj_(dy) / norm;
        // }
        else if(std::abs(dy) > std::abs(dx))
        {
            T tmp = dx / dy;
            sn = T(1.0) / std::sqrt(T(1.0) + tmp*tmp);
            cs = tmp*sn;
        } 
        else
        {   
            T tmp = dy / dx;
            cs = T(1.0) / std::sqrt(T(1.0) + tmp*tmp);
            sn = tmp*cs;
        }    
    }
   

    void plane_rotation_(T_vec& H, T_vec& cs_, T_vec& sn_, T_vec& s, const int i) const
    {
        for (int k = 0; k < i; k++)
        {
            apply_plane_rotation_(H[ k*restart_ + i], H[ (k+1)*restart_ + i], cs_[k], sn_[k]);
        }

        generate_plane_rotation_(H[ (i)*restart_ + i], H[ (i+1)*restart_ + i], cs_[i], sn_[i]);
        apply_plane_rotation_(H[ (i)*restart_ + i], H[ (i+1)*restart_ + i], cs_[i], sn_[i]);
        H[ (i+1)*restart_ + i] = T(0.0); //remove numerical noice below diagonal
        apply_plane_rotation_(s[i], s[i + 1], cs_[i], sn_[i]);
    }
    // *** Givens rotations ends ***

    //constructs solution of the linear system
    void construct_solution_(const int i, T_vec& x) const
    {
        // x= V(1:N,0:i)*s(0:i)+x
        for (int j = 0; j <= i; j++) 
        {
            // x = x + s[j] * V(j)
            //add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)
            vec_ops_->add_mul(s[j], V[j], x);
        }        
    }

    //used for debug!
    // void print_matrix_(const T_vec& H) const
    // {
    //     for(int j=0;j<=restart_;j++)
    //     {
    //         for(int k=0;k<restart_;k++)
    //         {
    //             std::cout << H[(j)*restart_ + k] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "============" << std::endl;
    // }
    // void print_vector_(const T_vec& vvv) const
    // {
    //     for(int j=0;j<=restart_;j++)
    //     {
    //         std::cout << vvv[j] << " ";
    //     }
    //     std::cout << "**********" << std::endl;       
    // }   

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    ~gmres()
    {
        if(H!=nullptr) free(H);
        if(s!=nullptr) free(s);
        if(sn!=nullptr) free(sn);
        if(cs!=nullptr) free(cs);      
    }

    gmres(const vector_operations_type *vec_ops, 
              Log *log = NULL, int obj_log_lev = 3): 

        parent_t(vec_ops, log, obj_log_lev, "gmres::"),
        buf_r(vec_ops), 
        buf_other(vec_ops),
        V(vec_ops), 
        r_tilde(buf_other[0]),
        w(buf_other[1]),
        V0(buf_other[2]),
        r(buf_r[0]),
        use_precond_resid_(true), 
        resid_recalc_freq_(0), 
        restart_(2) //using default restarts
    {
        try
        {
            init_host_();
        }
        catch(...)
        {
            throw;
        }

        buf_r.init(); 
        buf_other.init(3);
        V.init(restart_+1); 
        
    }

    void set_restarts(unsigned int restart)
    { 
        restart_ = (int)restart; 

        V.init(restart_+1); 
        try
        {
            init_host_();
        }
        catch(...)
        {
            throw;
        }        
    }
    void    set_use_precond_resid(bool use_precond_resid) { use_precond_resid_ = use_precond_resid; }
    void    set_resid_recalc_freq(int resid_recalc_freq) { resid_recalc_freq_ = resid_recalc_freq; }

    virtual bool solve(const linear_operator_type &A, const T_vec &b, T_vec &x)const
    {                
        if ((!use_precond_resid_)&&(prec_ != NULL)) 
            throw std::logic_error("gmres::solve: use_precond_resid_ == false with non-empty preconditioner is not supported");

        if (prec_ != NULL) 
        {
            prec_->set_operator(&A);
        }

        bufs_arr_use_wrap_t use_wrap_V(V);
        bufs_arr_use_wrap_t use_wrap_other(buf_other);
        use_wrap_V.start_use_range(restart_+1);
        use_wrap_other.start_use_range(3);

        buf_use_wrap_t use_wrap_buf(buf_r);
        use_wrap_buf.start_use_all();
        zero_host_H_();
        monitor_call_wrap_t monitor_wrap(monitor_);
        
        if ((use_precond_resid_)&&(prec_ != NULL)) 
        {
            vec_ops_->assign(b, r);
            prec_->apply(r);
            monitor_wrap.start(r);
        }
        else
        {
            monitor_wrap.start(b);
        }

        bool res = true;
        int i;
        calc_residual_(A, x, b, r);
        if( !monitor_.check_finished(x, r) )
        {            
            do
            {
                T beta = normalize_(r, T(1.0));
                vec_ops_->assign( r, V[0]);
                zero_host_s_(); // s(:) = 0;
                s[0] = beta;
                i = -1;
                do
                {
                    ++i;
                    ++monitor_;
                    vec_ops_->assign(r, w);
                    calc_Krylov_vector_(A, w, r);
                    // Gram-Schmidt with iterative correction
                    for( int k = 0; k <= i; k++)
                    {
                        T alpha = vec_ops_->scalar_prod(V[k], r); // H(k,i) = (V[k],V[i+1])

                        vec_ops_->add_mul(-alpha, V[k], r); // V(i+1) -= H(k, i) * V(k)

                        // T c_norm = alpha;
                        // int correction_iterations = 0;
                        // while(c_norm > 1.0e-10*std::sqrt(vec_ops_->sz_)) //iterative correction
                        // {
                        //     correction_iterations++;
                        //     T c = vec_ops_->scalar_prod(V[k], r); // H(k,i) = (V[k],V[i+1])
                        //     c_norm = std::abs(c);
                        //     vec_ops_->add_mul(-c, V[k], r);
                        //     alpha += c;
                        //     if(correction_iterations>10)
                        //     {
                        //         break;
                        //         //if we are here, then the method will probably diverge
                        //     }
                        // }

                        H[k*restart_+i] = alpha;

                    }
                    T h_ip = normalize_(r);
                    H[(i + 1)*restart_ + i] = h_ip;
                    vec_ops_->assign(r, V[i+1]);
                    
                    plane_rotation_(H, cs, sn, s, i); //QR via Givens rotations
                    
                    // TODO: this is bad, converence fails. Think about it!
                    // T resid_estimate = std::abs(s[i + 1]);
                    
                    // if (resid_estimate < monitor_.rel_tol())
                    // {
                    //     // TODO: check real solution? 
                    //     // Ritz value may not be acurate in approx arithmetics
                    //     solve_triangular_system_(H, i, s);

                    //     construct_solution_(i, x);
                        
                    //     calc_residual_(A, x, b, r);   
                    //     resid_estimate = vec_ops_->norm(r);
                    //     if(resid_estimate < monitor_.rel_tol())                   
                    //         break;
                    // }

                } while( i + 1 < restart_);
                
                solve_triangular_system_(H, i, s);

                construct_solution_(i, x);
                
                calc_residual_(A, x, b, r);

            } while (!monitor_.check_finished(x, r));

        }

        res = monitor_.converged();
        if(!res)
            logged_obj_t::error_f("solve: linear solver failed to converge");
        
        return res;
    
    }
};

}
}

#endif //__SCFD_GMRES_H__
