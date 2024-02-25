/**
General wrapper solver for the exact solver that can be used as a stand alone and in rank-1 perturbated form
We need to incorporate an exact solver into the set of iterative solvers, since the consept should be genral.
One way to do it is to use a preconditioner as an exact solver.
In this case the consept is as follows:
1. Preconditioner P should take care of an exact soltion of the linear system, presented by matrix A,
  which is stored in the linear operator L.
    - In this case there is a method P.apply(y) which treats y as a RHS of the linear system and
        solves: x=A\y, y<-x.
    - The method P.set_operator(const LinearOperator *L) sets a linear operator that contains a matrix A.
    - The LinearOperator should have a method L.get_matrix_ref() which returns the matrix refernce
    - The method in the Preconditioner P.update_matrix(L) updates the reference to the matrix from L.
    - P.is_constant_matrix() (true/fasle) defines if the pecondtioner CAN ruin the matrix 
        that is referenced from the L. 

2. LinearOperator L contains a reference to the matrix A and has a method L.apply(x,y).
    - This method essentially y<-Ax and is used to calculate the residual.
    - The LinearOperator must have a method L.get_matrix_ref() which returns the matrix reference.
    - The method L.is_constant_matrix() (true/false) defines if the preconditioner CAN ruin the matrix 
        which reference is stored in the L, default is false.

*/


#ifndef __EXACT_WRAPPER_LINEAR_SYSTEM_SOLVE_H__
#define __EXACT_WRAPPER_LINEAR_SYSTEM_SOLVE_H__

#include "detail/monitor_call_wrap.h"
#include "iter_solver_base.h"

namespace numerical_algos
{
namespace lin_solvers 
{

template<class LinearOperator, class Preconditioner, class VectorOperations, class Monitor, class Log>
class exact_wrapper: public iter_solver_base<LinearOperator,Preconditioner, VectorOperations,Monitor,Log>
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using linear_operator_type = LinearOperator;
    using preconditioner_type = Preconditioner;
    using vector_operations_type = VectorOperations;
    using monitor_type = Monitor;
    using log_type = Log;
private:
    using monitor_call_wrap_t = detail::monitor_call_wrap<VectorOperations, Monitor>;
    using parent_t = iter_solver_base<LinearOperator,Preconditioner, VectorOperations,Monitor,Log>;
    using logged_obj_t = typename parent_t::logged_obj_t;
protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:

    exact_wrapper(const vector_operations_type *vec_ops_p, Log *log = nullptr, int obj_log_lev = 3):
    parent_t(vec_ops_p, log, obj_log_lev, "exact::")
    {
        vec_ops_->init_vector(r_);
        vec_ops_->start_use_vector(r_);
    }
    ~exact_wrapper()
    {
        vec_ops_->stop_use_vector(r_);
        vec_ops_->free_vector(r_);
    }

    void set_basis_size(int basis_sz) 
    {
        //dummy
    }   
    void set_use_precond_resid(bool use_precond_resid)
    {
        //dummy
    }
    void set_resid_recalc_freq(int resid_recalc_freq)
    {
        //dummy
    }     
    virtual bool solve(const linear_operator_type &A, const vector_type &b, vector_type &x)const
    {
        if (prec_ == nullptr)
        {
            throw std::logic_error("exact::solve: preconditioner can't be NULL since preconditioner is used to solve the system exactly! Use 'set_preconditioner' method before running an exact solver.");
        }


        monitor_call_wrap_t monitor_wrap(monitor_);
        monitor_wrap.start(b);

        prec_->set_operator(&A);
        vec_ops_->assign(b, x);
        prec_->apply(x);
        ++monitor_;
        calc_residual(A, x, b, r_);
        monitor_.check_finished(x, r_);
        bool res = monitor_.converged();
        return res;

    }


private:
    mutable vector_type r_;
    void calc_residual(const linear_operator_type &A, const vector_type &x, const vector_type &b, vector_type &r)const
    {
        vec_ops_->assign_scalar(0, r);
        A.apply(x, r);
        vec_ops_->add_mul( static_cast<scalar_type>(1.0), b, static_cast<scalar_type>(-1.0), r);
    }


};


}
}


#endif
