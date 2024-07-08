// Copyright © 2016-2023 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

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
#include <limits>
#include "detail/monitor_call_wrap.h"
// #include "detail/algo_utils_hierarchy.h"
// #include "detail/algo_params_hierarchy.h"
// #include "detail/algo_hierarchy_creator.h"
#include "iter_solver_base.h"
#include "detail/dense_operations.h"
#include "detail/residual_regularization_dummy.h"
// #include "preconditioners/dummy.h"

namespace numerical_algos
{
namespace lin_solvers 
{

//demands for template parameters:
//VectorOperations fits VectorOperations concept (see CONCEPTS.txt)
//LinearOperator and Preconditioner fit LinearOperator concept (see CONCEPTS.txt)
//VectorOperations, LinearOperator and Preconditioner have same T_vec

//Monitor concept:
//TODOjj

// template
// <
//      class VectorOperations, class Monitor, class Log, 
//      class LinearOperator, class Preconditioner = preconditioners::dummy<VectorOperations,LinearOperator>,
//      class ResidualRegulariation = detail::residual_regularization_dummy,
//      class DenseOperations = detail::dense_operations<typename VectorOperations::Ord, typename VectorOperations::scalar_type> 
// >
template<
        class LinearOperator,
        class Preconditioner,
        class VectorOperations,
        class Monitor,
        class Log,
        class ResidualRegulariation = detail::residual_regularization_dummy,
        class DenseOperations = detail::dense_operations<
            std::ptrdiff_t, 
            typename VectorOperations::scalar_type
            > 
        >
class gmres : public iter_solver_base<LinearOperator, Preconditioner, VectorOperations, Monitor, Log>
{
    using parent_t = iter_solver_base<LinearOperator, Preconditioner, VectorOperations, Monitor, Log>;
    using logged_obj_t = typename parent_t::logged_obj_t;
    using logged_obj_params_t = typename parent_t::logged_obj_params_t;

public:
    using scalar_type =  typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using multivector_type = typename VectorOperations::multivector_type;
    using linear_operator_type =  LinearOperator;
    using preconditioner_type = Preconditioner;
    using vector_operations_type = VectorOperations;
    using dense_operations_t = DenseOperations;
    using monitor_type = Monitor;
    using log_type = Log;
    using residual_regulaization_t = ResidualRegulariation;


    struct params: public logged_obj_params_t
    {
        unsigned basis_size; //size of the krylov basis
        unsigned batch_size; //size of the batch size s.t. Ritz vector converhence is checked each batch_size inside the krylov bass size
        char preconditioner_side; //can be L for left and R for right
        bool reorthogonalization; //apply additional reorthogonalization in Gram-Schmidt process
        // typename Monitor::params monitor;

        params(const std::string &log_prefix = "", const std::string &log_name = "gmres::"):
            logged_obj_params_t(0, log_prefix+log_name),
            basis_size(15), 
            batch_size(5), 
            preconditioner_side('R'), 
            reorthogonalization(false)
            // monitor( typename Monitor::params(this->log_msg_prefix) )
        { }

    };

private:
    using T = scalar_type;
    using T_vec = vector_type;
    using T_mvec = multivector_type;

    using D_vec = typename dense_operations_t::vector_type;
    using D_mat = typename dense_operations_t::matrix_type;


    using monitor_call_wrap_t = detail::monitor_call_wrap<VectorOperations, Monitor>;


    T error_L2_basic_type_;
    mutable T_mvec V_;
    mutable T_vec r_;
    mutable T_vec y_;
    mutable T_vec x_tmp_;
    
    //parameters:
    params prms_;

    //host dense operations vectors and matrices
    mutable D_mat H_;
    mutable D_vec s_;
    mutable D_vec cs_;
    mutable D_vec sn_;
    mutable D_vec s_h_;
    mutable D_vec rr;


    void calc_left_preconditioned_residual(const linear_operator_type &A, const T_vec &x, const T_vec &b, T_vec &r)const
    {
        
        A.apply(x, r);
        vec_ops_->add_lin_comb(static_cast<T>(1.0), b, static_cast<T>(-1.0), r);
        
        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'L'))
        {
            prec_->apply(r);
        }
    }

    void calc_right_precond_solution(T_vec &x) const
    {
        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'R'))
        {
            prec_->apply(x);
        }
    }


    void calc_krylov_vector(const linear_operator_type &A, const T_vec &x, T_vec &r)const
    {
        if(prec_ == nullptr)
        {
            A.apply(x, r);
        }
        else
        {
            if(prms_.preconditioner_side == 'L') 
            {
                A.apply(x, r);
                prec_->apply(r);
            }
            else if(prms_.preconditioner_side == 'R') 
            {
                vec_ops_->assign(x, y_);
                prec_->apply(y_);
                A.apply(y_, r);
            }
        }
    }


    void init_host()
    {
        dense_ops_->init_matrix(H_);
        dense_ops_->init_col_vectors(s_, cs_, sn_, s_h_, rr);
    }

    void free_host_()
    {
        dense_ops_->free_matrix(H_);
        dense_ops_->free_col_vectors(s_, cs_, sn_, s_h_, rr);
    }

    void init_all() const
    {
        vec_ops_->init_vector( r_ );
        vec_ops_->init_vector( y_ );
        vec_ops_->init_vector( x_tmp_ );
        vec_ops_->init_multivector( V_, prms_.basis_size+1);
    }

    void init_error_L2_basic_type()
    {
        vec_ops_->start_use_vector(y_);
        vec_ops_->assign_scalar(T(1), y_);
        error_L2_basic_type_ = (std::numeric_limits<T>::epsilon() )*std::sqrt(vec_ops_->scalar_prod(y_,y_));
        vec_ops_->stop_use_vector(y_);
    }

    void start_use_all() const
    {
        vec_ops_->start_use_vector( r_ );
        vec_ops_->start_use_vector( y_ );
        vec_ops_->start_use_vector( x_tmp_ );
        vec_ops_->start_use_multivector( V_, prms_.basis_size+1 );    
    }

    void stop_use_all() const
    {
        vec_ops_->stop_use_vector( r_ );
        vec_ops_->stop_use_vector( y_ );
        vec_ops_->stop_use_vector( x_tmp_ );
        vec_ops_->stop_use_multivector( V_, prms_.basis_size+1 );        
    }
    void free_all() const
    {
        vec_ops_->free_vector( r_ );
        vec_ops_->free_vector( y_ );
        vec_ops_->free_vector( x_tmp_ );
        vec_ops_->free_multivector( V_, prms_.basis_size+1 );
    }

    void zero_host_H() const
    {
        dense_ops_->assign_scalar_matrix(0, H_);
    }
    void zero_host_s() const
    {
        dense_ops_->assign_scalar_col_vector(0, s_);
    }


     //constructs solution of the linear system
    void construct_solution(const int i, const D_vec& s, T_vec& x) const
    {
        // x= V(1:N,0:i)*s(0:i)+x
        vec_ops_->assign_scalar(0, x);
        for (int j = 0; j <= i; j++) 
        {
            // x = x + s[j] * V(j)
            //add_lin_comb(scalar_type mul_x, const vector_type& x, vector_type& y)
            T_vec& Vj = vec_ops_->at(V_, prms_.basis_size+1, j);
            vec_ops_->add_lin_comb(s(j), (const T_vec)Vj, 1.0, x);          
        }
    }


protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;
    std::shared_ptr<dense_operations_t> dense_ops_;
    std::shared_ptr<residual_regulaization_t> residual_reg_;

public:
    ~gmres()
    {
        free_host_();
        free_all();
    }

    gmres(  
        const vector_operations_type *vec_ops,
        Log *log = nullptr,
        const params& prm = params(),
        int obj_log_lev = 3,
        std::shared_ptr<residual_regulaization_t> residual_reg = std::make_shared<residual_regulaization_t>(),
        std::shared_ptr<dense_operations_t> dense_ops = std::make_shared<dense_operations_t>()
    ) : 
        parent_t(vec_ops, log, obj_log_lev, "gmres::" ), 
        prms_(prm),
        residual_reg_(std::move(residual_reg)),
        dense_ops_(std::move(dense_ops))
    {
        dense_ops_->init(prm.basis_size+1, prm.basis_size);
        init_host();
        init_all();
        init_error_L2_basic_type();
    }

    virtual bool solve(const linear_operator_type &A, const T_vec &b, T_vec &x)const
    {                
    
        auto restart_ = prms_.basis_size;
        start_use_all();
        // if (prec_ != nullptr)
        // {
        //     throw std::logic_error("gmres::solve: use_precond_resid_ == false with non-empty preconditioner is not supported");
        // }

        /*if (prec_ != nullptr) 
        {
            prec_->set_operator(&A);
        }*/

        zero_host_H();
        monitor_call_wrap_t monitor_wrap(monitor_);
        
        if ((prec_ != nullptr)&&(prms_.preconditioner_side == 'L')) 
        {
            vec_ops_->assign(b, r_);
            prec_->apply(r_);
            residual_reg_->apply(r_);
            monitor_wrap.start(r_);
        }
        else
        {
            monitor_wrap.start(b);
            
        }

        bool res = true;
        int i;

        // calc_preconditioned_residual(A, x, b, r_);
        calc_left_preconditioned_residual(A, x, b, r_);
        residual_reg_->apply(r_);
        // logged_obj_t::info_f("||r|| = %e", vec_ops_->norm(r_) );
        bool converged_by_checked_ritz_norm = false;
        std::size_t total_iterations = 0;
        
        T previous_res = vec_ops_->norm(r_);
        std::vector<scalar_type> reduction_rates; 
        reduction_rates.reserve(prms_.batch_size);

        if( !monitor_.check_finished(x, r_) )
        {            
            do
            {
                T beta = vec_ops_->norm(r_);
                T_vec& V_0 = vec_ops_->at(V_, restart_+1, 0);
                vec_ops_->scale(1.0/beta, r_);
                vec_ops_->assign(r_, V_0);
                zero_host_s(); // s(:) = 0;
                dense_ops_->vector_at(s_, 0) = beta;
                // std::cout << "s[0] = " << dense_ops_->vector_at(s_, 0) << std::endl;
                // std::cout << V_0[0] << " <-> " << vec_ops_->at(V_, restart_+1, 0)[0] << std::endl;
                i = -1;
                do
                {
                    ++i;
                    ++monitor_;
                    vec_ops_->assign(r_, y_);
                    calc_krylov_vector(A, y_, r_);
                    residual_reg_->apply(r_);
                    T next_r_norm = vec_ops_->norm(r_);
                    // std::cout << "||r|| = " << next_r_norm << std::endl;
                    // Gram-Schmidt with iterative correction
                    for( int k = 0; k <= i; k++)
                    {
                        T_vec& V_k = vec_ops_->at(V_, restart_+1, k);
                        T alpha = vec_ops_->scalar_prod(V_k, r_); // H(k,i) = (V[k],V[i+1])
                        // std::cout << "i = " << i << ", k = " << k << ", (v_k,r_) = " << alpha  << ", ||v_k|| = " << vec_ops_->norm(V_k) << std::endl;
                        vec_ops_->add_lin_comb(-alpha, V_k, 1.0, r_); // V(i+1) -= H(k, i) * V(k)

                        T c_norm = alpha;
                        int correction_iterations = 0;
                        while( (prms_.reorthogonalization)&&(c_norm > error_L2_basic_type_*next_r_norm )) //iterative correction
                        {
                            correction_iterations++;
                            T c = vec_ops_->scalar_prod(V_k, r_); // H(k,i) = (V[k], V[i+1])
                            c_norm = std::abs(c);
                            vec_ops_->add_lin_comb(-c, V_k, 1.0, r_);
                            alpha += c;
                            if(correction_iterations>10)
                            {
                                //if we are here, then the method will probably diverge.
                                logged_obj_t::warning_f("failed in Gram-Schmidt reorthogonalization in iteration %i, restart %i with error %e", k, i, c_norm); 
                                break;
                            }
                        }
                        dense_ops_->matrix_at(H_, k, i) = alpha;
                    }

                    // for(int ll=0;ll<=i;ll++)
                    // {
                    //     T_vec V_ll = vec_ops_->at(V_, restart_+1, ll);
                    //     T alpha = vec_ops_->scalar_prod(V_ll, r_);
                    //     T V_ll_norm = vec_ops_->norm(V_ll);
                    //     std::cout << "i = " << i << ", ll = " << ll << ", (v_ll,r_) = " << alpha  << ", ||v_ll|| = " << V_ll_norm << std::endl;
                    // }


                    T h_ip = vec_ops_->norm(r_);
                    // H_[(i + 1)*restart_ + i] = h_ip;
                    dense_ops_->matrix_at(H_, i+1, i) = h_ip;

                    T_vec& V_ip1 = vec_ops_->at(V_, restart_+1, i+1);
                    
                    vec_ops_->scale(1.0/h_ip, r_);
                    vec_ops_->assign(r_, V_ip1);
                    
                    // plane_rotation_(H_, cs_, sn_, s_, i); //QR via Givens rotations
                    dense_ops_->plane_rotation_col(H_, cs_, sn_, s_, i);

                    T resid_estimate = std::abs(dense_ops_->vector_at(s_, i+1));

                    reduction_rates.push_back(resid_estimate/previous_res);
                    if((total_iterations%prms_.batch_size == 0)||(i+1 == restart_))
                    {
                        auto reduction_rate_prod = std::accumulate(reduction_rates.begin(), reduction_rates.end(), 1.0, std::multiplies<T>() );
                        logged_obj_t::info_f("iter = %i(%i), resid_estimate = %e, reduction = %.03f", total_iterations+1, i+1, resid_estimate, std::pow(reduction_rate_prod, 1.0/reduction_rates.size() ) );
                    }
                    if(total_iterations % prms_.batch_size == 0)
                    {
                        reduction_rates.clear();
                    }
                    total_iterations++;
                    previous_res = resid_estimate;


                    if ( resid_estimate < monitor_.tol() ) //fix to tol
                    {
                        vec_ops_->assign(x, x_tmp_);
                        logged_obj_t::info_f("resid_estimate = %e monitor_.tol() = %e", resid_estimate, monitor_.tol());
                    //      check real solution
                    // Ritz value may not be acurate in approx arithmetics
                        dense_ops_->solve_upper_triangular_subsystem(H_, s_, s_h_, i+1);

                        // std::cout << "s_:" << std::endl;
                        // dense_ops_->print_col_vector(s_, 9);
                        // std::cout << "s_h_:" << std::endl;
                        // dense_ops_->print_col_vector(s_h_, 9);
                        // dense_ops_->print_matrix(H_,2);

                        // for(int jj=0;jj<i+1;jj++)
                        // {
                        //     rr(jj) = 0;
                        //     for(int kk=0;kk<i+1;kk++)
                        //     {
                        //         rr(jj) += H_(jj,kk)*s_h_(kk);
                        //     }
                        //     rr(jj) -= s_(jj);
                        // }
                        // std::cout << "residual: " << std::endl;
                        // dense_ops_->print_col_vector(rr,2);
                        // std::cout << "H_:" << std::endl;
                        // dense_ops_->print_matrix(H_,15);            

                        construct_solution(i, s_h_, y_);
                        calc_right_precond_solution(y_);
                        vec_ops_->add_lin_comb(1.0, y_, 1.0, x);
                        calc_left_preconditioned_residual(A, x, b, r_); 
                        residual_reg_->apply(r_);

                        resid_estimate = vec_ops_->norm(r_);
                        logged_obj_t::info_f("actual_residual = %e", resid_estimate);
                        if(resid_estimate < monitor_.tol())
                        {
                            converged_by_checked_ritz_norm = true;
                            break;
                        }
                        else
                        {
                            vec_ops_->assign(x_tmp_, x);
                        }
                    }
                }
                while( i + 1 < restart_);
                
                if(!converged_by_checked_ritz_norm)
                {
                    // std::cout << "s_:" << std::endl;
                    // dense_ops_->print_col_vector(s_, 16);
                    dense_ops_->solve_upper_triangular_subsystem(H_, s_, s_h_, i+1);
                    // std::cout << "s_h_:" << std::endl;
                    // dense_ops_->print_col_vector(s_h_, 16);
                    construct_solution(i, s_h_, y_);
                    calc_right_precond_solution(y_);
                    vec_ops_->add_lin_comb(1.0, y_, 1.0, x);
                    calc_left_preconditioned_residual(A, x, b, r_);
                    residual_reg_->apply(r_);   
                    // std::cout << "norm r = " << vec_ops_->norm(r_);
                    // exit(-1);
                }

            } 
            while(!monitor_.check_finished(x, r_) );

        }
        res = monitor_.converged();
        if(!res)
        {
            logged_obj_t::error_f("solve: linear solver failed to converge");
        }
        
        stop_use_all();

        return res;
    
    }


};

}
}

#endif //__SCFD_GMRES_H__